"""
Authors :
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import os
from math import ceil

from tqdm.auto import tqdm
import numpy as np


from CNN.build_dataset_chameleon import highpass
import h5py

# Unsupervised components (vectorization + clustering)
from unsupervised_learning.vectorization import (
    BaseVectorizer,
    TSFreshVectorizer,
)
from unsupervised_learning.classification import (
    BaseClassifier,
    KMeansClassifier,
)


def saveClassification(segmentation: np.ndarray, output_file: str) -> None:
    """
    Save the segmentation to a file.

    Parameters
    ----------
    segmentation : np.ndarray
        The segmentation to save.
    output_file : str
        The file where the segmentation will be saved.
    """

    np.save(output_file, segmentation)


def _dataLoader(chunk_file):
    with h5py.File(chunk_file, "r", libver="latest") as hf_chunk:
        chunk_len = len(hf_chunk["metadata/ciphers/"].keys())
        for n in range(0, chunk_len):
            traces = hf_chunk[f"data/traces/trace_{n}"]
            yield traces[:]


# ======================================================================
# Unsupervised sliding-window pipeline:
#   1. extract fixed-size windows
#   2. vectorize (TSFresh / autoencoder / other BaseVectorizer)
#   3. cluster or classify vectors (KMeansClassifier / other BaseClassifier)
# ======================================================================


def _extract_fixed_windows(
    trace: np.ndarray,
    window_size: int,
    stride: int,
    batch_size: int,
):
    """
    Extract fixed-length windows with given stride, yielding them in batches.

    Unlike _cutSubWindows (used by CNNs), this function ONLY yields
    full windows of length `window_size` and drops any final incomplete
    window. All windows inside each yielded batch therefore have shape
    (window_size,).

    Parameters
    ----------
    trace : np.ndarray
        1D trace array of shape (T,).
    window_size : int
        Number of samples per window.
    stride : int
        Stride between consecutive window starts.
    batch_size : int
        Maximum number of windows per yielded batch.

    Yields
    ------
    list[np.ndarray]
        A list of windows, each of shape (window_size,).
    """
    T = trace.shape[0]
    if T < window_size:
        # No full window can be taken
        return

    windows_batch = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        # This is guaranteed to be a full window
        windows_batch.append(trace[start:end])

        if len(windows_batch) == batch_size:
            yield windows_batch
            windows_batch = []

    if windows_batch:
        yield windows_batch


def _vectorize_in_batches(
    windows: np.ndarray, vectorizer: BaseVectorizer, batch_size: int
) -> np.ndarray:
    """
    Apply vectorizer.transform to windows in batches for memory efficiency.
    windows: (N, L)
    returns: (N, D)
    """
    N = windows.shape[0]
    if N == 0:
        return np.empty((0, 0), dtype=float)

    feats = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = windows[start:end]
        feats_batch = vectorizer.transform(batch)
        feats.append(feats_batch)
    return np.vstack(feats)


def classifyTrace_unsupervised(
    trace_file: str,
    vectorizer: BaseVectorizer,
    classifier: BaseClassifier,
    stride: int,
    window_size: int,
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Unsupervised sliding-window classification using a vectorizer and
    an unsupervised classifier, modeling the original workflow as close as possible


    Parameters
    ----------
    trace_file : str
        Path to HDF5 trace file (same format used in CNN pipeline).
    vectorizer : BaseVectorizer
        Window-level feature extractor
    classifier : BaseClassifier
        Unsupervised classifier
    stride : int
    window_size : int
    batch_size : int, optional

    Returns
    -------
    np.ndarray
        For each trace, per-window scores.
        Shape: (num_traces, num_windows, num_classes).
        If classifier has predict_proba, scores are soft probabilities.
        Otherwise, scores are one-hot encodings of predicted labels. (Not sure if this works with the further code)
    """

    with h5py.File(trace_file, "r", libver="latest") as hf_chunk:
        total_traces = len(hf_chunk["metadata/ciphers/"].keys())

    # Optional first pass to train vectorizer
    if vectorizer.needs_two_pass:
        # Collect all raw windows, vectorizer.partial_fit on batches
        for trace in tqdm(_dataLoader(trace_file), total=total_traces, colour="yellow", desc="Vectorizer first pass"):
            trace = highpass(trace, 0.001)
            trace = (trace - np.mean(trace, axis=0)) / np.std(trace, axis=0)

            for windows_batch in _extract_fixed_windows(trace, window_size, stride, batch_size):
                windows_arr = np.asarray(windows_batch)
                vectorizer.partial_fit(windows_arr)


    per_trace_feats = []
    per_trace_scores = []
    # Vectorize all traces (and classify if you can't store all vectors)
    for trace in tqdm(_dataLoader(trace_file), total=total_traces, colour="green", desc="Vectorizing and classifying traces"):
        trace = highpass(trace, 0.001)
        trace = (trace - np.mean(trace, axis=0)) / np.std(trace, axis=0)

        feats_batches = []

        for windows_batch in _extract_fixed_windows(trace, window_size, stride, batch_size):
            windows_arr = np.asarray(windows_batch)
            feats_batch = vectorizer.transform(windows_arr)  # shape (batch, D)
            feats_batches.append(feats_batch)

        X = np.vstack(feats_batches) if feats_batches else np.zeros((0, 0))

        if classifier.fit_per_trace:
            if X.shape[0] == 0:
                # consistent empty output shape
                scores = np.zeros((0,3))
            else:
                classifier.fit(X)
                if hasattr(classifier, "predict_proba"):
                    scores = classifier.predict_proba(X)
                else:
                    scores = classifier.predict(X)

            per_trace_scores.append(scores)

        else:
            per_trace_feats.append(X)

    # Train classifier on all features from file, should be default
    if not classifier.fit_per_trace:
        all_X = np.vstack(per_trace_feats) if len(per_trace_feats) else np.zeros((0, 0))

        if all_X.shape[0] > 0:
            classifier.fit(all_X)


        for X in tqdm(per_trace_feats, colour="cyan", desc="Classifying globally-fitted"):
            if X.shape[0] == 0:
                per_trace_scores.append(np.zeros((0, 3)))
                continue

            if hasattr(classifier, "predict_proba"):
                scores = classifier.predict_proba(X)
            else:
                labels = classifier.predict(X)
                scores = labels

            per_trace_scores.append(scores)


    return np.stack(per_trace_scores, axis=0)