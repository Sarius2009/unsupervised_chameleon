"""
Authors :
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""
import gc
import os
from time import sleep

import torch
from torch import nn
from tqdm import tqdm
import numpy as np


from CNN.build_dataset_chameleon import highpass
import h5py

from inference_pipeline.debug import loaderGt
# Unsupervised components (vectorization + clustering)
from unsupervised_learning.vectorization import BaseVectorizer
from unsupervised_learning.classification import BaseClassifier

import sys
import numpy as np



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
    epochs,
    tmp_folder,
    batch_size: int = 2000,
    stored_features_path: str = None
) -> np.ndarray:
    """
    Unsupervised sliding-window classification using a vectorizer and
    an unsupervised classifier, modeling the original workflow as close as possible


    Parameters
    ----------
    stored_features_path
    tmp_folder
    skip_feature_extraction
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

    per_trace_scores = []
    feat_files = []          # list of npy file paths for each trace
    if stored_features_path is None:
        # Optional first pass to train vectorizer
        if vectorizer.needs_two_pass:
            # Collect all raw windows, vectorizer.partial_fit on batches
            _train_autoencoder(
                vectorizer=vectorizer,
                trace_file=trace_file,
                window_size=window_size,
                stride=stride,
                batch_size=batch_size,
                epochs=epochs,        # or 2, 3, ... as you like
            )



        feat_dim = None          # feature dimension (D)
        # Vectorize all traces (and classify if you can't store all vectors)
        for trace_idx, trace in enumerate(tqdm(_dataLoader(trace_file), total=total_traces, colour="green", desc="Vectorizing traces" + (" and classifying" if classifier.fit_per_trace else ""))):
            trace = highpass(trace, 0.001)
            trace = (trace - np.mean(trace, axis=0)) / np.std(trace, axis=0)

            feats_batches = []

            for windows_batch in _extract_fixed_windows(trace, window_size, stride, batch_size):
                windows_arr = np.asarray(windows_batch)
                feats_batch = vectorizer.transform(windows_arr)  # shape (batch, D)
                feats_batches.append(feats_batch)

            X = np.vstack(feats_batches)

            if classifier.fit_per_trace:
                raise NotImplementedError('Per trace fitting was not updated')
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
                # Store features for this trace on disk instead of in RAM
                trace_feat_path = os.path.join(
                    tmp_folder, f"trace_{trace_idx}_feats.npy"
                )
                np.save(trace_feat_path, X)

                feat_files.append(trace_feat_path)
                if X.shape[0] > 0:
                    feat_dim = X.shape[1] if feat_dim is None else feat_dim
            del X, feats_batch, trace, windows_batch
            gc.collect()
            sleep(20)
            gc.collect()
            sleep(20)
            gc.collect()
        if hasattr(vectorizer, "global_avg_error"):
            print('Avg. reconstruction error:', vectorizer.global_avg_error)
            with open('loss.txt', 'w') as f:
                f.write(str(vectorizer.global_avg_error))
    else:
        for i in range(0, 16):
            trace_feat_path = os.path.join(
                tmp_folder, stored_features_path, f"trace_{i}_feats.npy"
            )
            feat_files.append(trace_feat_path)

    # Train classifier on all features from file (global fit)
    if not classifier.fit_per_trace:

        # Create a disk-backed memmap for all features
        # Before creating the memmap

        total_rows = 0
        feat_dim = None
        feat_dtype = None

        for path in feat_files:
            Xi = np.load(path, mmap_mode='r')  # memory-mapped, cheap
            if Xi.size == 0:
                continue

            n_rows, d = Xi.shape
            total_rows += n_rows

            if feat_dim is None:
                feat_dim = d
            if feat_dtype is None:
                feat_dtype = Xi.dtype

        # Now you have total_rows, feat_dim and dtype
        all_feats_path = os.path.join(tmp_folder, "all_feats.dat")
        all_X = np.memmap(
            all_feats_path,
            dtype=feat_dtype,   # matches saved features
            mode="w+",
            shape=(total_rows, feat_dim),
        )


        # Fill memmap by streaming per-trace features from disk
        offset = 0
        for path in tqdm(
                feat_files,
                total=len(feat_files),
                colour="magenta",
                desc="Building all_X memmap",
        ):
            Xi = np.load(path).astype(np.float32, copy=False)  # load per-trace features
            n_rows = Xi.shape[0]
            all_X[offset : offset + n_rows, :] = Xi
            offset += n_rows

        # Fit  on disk-backed array
        all_labels = []
        offset = 0
        for trace_idx, path in enumerate(feat_files):
            Xi = np.load(path)              # shape (n_windows_i, D)

            # load pinpoints for this trace
            with h5py.File(trace_file, "r", libver="latest") as hf:
                pinpoints = hf[f"metadata/pinpoints/pinpoints_{trace_idx}"][:]
                trace_len = hf[f"data/traces/trace_{trace_idx}"].shape[0]

            labels_i = window_labels_from_pinpoints(
                trace_len=trace_len,
                pinpoints=pinpoints,
                window_size=window_size,
                stride=stride,
            )
            assert labels_i.shape[0] == Xi.shape[0]

            all_labels.append(labels_i)

        labels = np.concatenate(all_labels).astype(int)
        classifier.fit(all_X, labels)

        metrics = []
# Second pass: classify per trace by loading back each X from file
        metrics = []

        for trace_idx, path in enumerate(tqdm(
                feat_files,
                total=len(feat_files),
                colour="cyan",
                desc="Classifying globally-fitted",
        )):
            X = np.load(path)

            if hasattr(classifier, "predict_proba"):
                scores = classifier.predict_proba(X)
            else:
                scores = classifier.predict(X)

            per_trace_scores.append(scores)
            metrics.append(window_metrics_from_scores(scores, all_labels[trace_idx]))


    print(f'Precicion: {np.mean([m["precision"] for m in metrics], axis=0):.4f}')
    print(f'Recall: {np.mean([m["recall"] for m in metrics], axis=0):.4f}')
    print(f'F1: {np.mean([m["f1"] for m in metrics], axis=0):.4f}')
    print(f'Balanced accuracy: {np.mean([m["balanced_accuracy"] for m in metrics], axis=0):.4f}')
    if hasattr(classifier, "cluster_to_class_"):
        print(f'Cluster-to-class mapping: {classifier.cluster_to_class_}')

    return np.stack(per_trace_scores, axis=0)



def window_labels_from_pinpoints(
        trace_len: int,
        pinpoints: np.ndarray,  # shape (N, 2) with columns [start, end]
        window_size: int,
        stride: int,
) -> np.ndarray:
    """
    Output binary window labels:
    - 1 if at least 50% of the window is continuously inside [start, end]
      for any pinpoint interval
    - 0 otherwise
    """
    num_windows = (trace_len - window_size) // stride + 1
    labels = np.zeros(num_windows, dtype=int)

    win_starts = np.arange(num_windows) * stride
    win_ends = win_starts + window_size
    min_overlap = 0.45 * window_size

    for s, e in pinpoints:
        s = int(s)
        e = int(e)

        # overlap length between each window and [s, e]
        overlap = np.minimum(win_ends, e) - np.maximum(win_starts, s)

        # continuous overlap must be >= 50% of window
        mask = overlap >= min_overlap
        labels[mask] = 1

    #print(np.count_nonzero(labels == 0) / len(labels), 'noise windows found')

    return labels


def _train_autoencoder(
        vectorizer,
        trace_file: str,
        window_size: int,
        stride: int,
        batch_size: int,
        epochs: int = 1,
):
    with h5py.File(trace_file, "r", libver="latest") as hf_chunk:
        total_traces = len(hf_chunk["metadata/ciphers/"].keys())

    runs = 1
    for epoch in range(epochs):
        if hasattr(vectorizer, "on_epoch_start"):
            vectorizer.on_epoch_start(epoch)

        pbar = tqdm(
            _dataLoader(trace_file),
            total=total_traces,
            colour="yellow",
            desc=f"Autoencoder epoch {epoch+1}/{epochs}",
            dynamic_ncols=True
        )
        if hasattr(vectorizer, "global_avg_error"):
            old = vectorizer.global_avg_error

        for trace in pbar:

            # -----------------------------
            # Preprocessing
            # -----------------------------
            trace = highpass(trace, 0.001)
            trace = (trace - np.mean(trace, axis=0)) / np.std(trace, axis=0)

            # -----------------------------
            # Train on all windows in trace
            # -----------------------------
            subsample_mul = 1
            if hasattr(vectorizer, "subsample_fraction"):
                subsample_mul = 1/vectorizer.subsample_fraction
            for windows_batch in _extract_fixed_windows(trace, window_size, stride, int(batch_size*subsample_mul)):
                windows_arr = np.asarray(windows_batch, dtype="float32")
                for i in range(runs):
                    vectorizer.partial_fit(windows_arr)
                    if runs > 1 and (i%100 == 0 or i == runs-1):
                        print(f'Error at {i+1}/{runs}: {vectorizer.global_avg_error:.3f}')
                runs = max(1, int(runs*0.9))
                if runs < 200:
                    runs = 1


            # -----------------------------
            # Store the error after this trace
            # -----------------------------
            if hasattr(vectorizer, "global_avg_error"):

                pbar.set_postfix({
                    "old_avg_err": f"{old:.4f}",
                    "avg_err": f"{vectorizer.global_avg_error:.4f}",
                })
                old = vectorizer.global_avg_error

        if hasattr(vectorizer, "on_epoch_end"):
            vectorizer.on_epoch_end(epoch)


def window_metrics_from_scores(scores, y_true, positive_class=1, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores)

    if scores.ndim == 2:
        # scores are probabilities (N,2) in class order [0,1]
        y_prob_pos = scores[:, positive_class]

        # hard predictions from probability threshold
        y_pred = (y_prob_pos >= threshold).astype(int)

        y_score = y_prob_pos  # keep for PR/AUC later
    else:
        # scores are hard labels (N,)
        y_pred = scores.astype(int)
        y_score = None

    TP = np.count_nonzero((y_true == 1) & (y_pred == 1))
    FP = np.count_nonzero((y_true == 0) & (y_pred == 1))
    TN = np.count_nonzero((y_true == 0) & (y_pred == 0))
    FN = np.count_nonzero((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    bal_acc   = 0.5 * (
            (TP / (TP + FN) if (TP + FN) else 0.0) +
            (TN / (TN + FP) if (TN + FP) else 0.0)
    )

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "y_score": y_score,
    }




