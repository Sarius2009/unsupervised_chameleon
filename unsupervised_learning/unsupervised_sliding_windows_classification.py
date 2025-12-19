"""
Authors :
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""
import gc
import os
from time import sleep

import h5py
from tqdm import tqdm
import numpy as np

from CNN.build_dataset_chameleon import highpass

# Unsupervised components (vectorization + clustering)
from unsupervised_learning.vectorization import BaseVectorizer
from unsupervised_learning.classification import BaseClassifier


def loaderGt(chunk_file):
    with h5py.File(chunk_file, 'r', libver='latest') as hf_chunk:
        chunk_len = len(hf_chunk['metadata/ciphers/'].keys())
        for n in range(0, chunk_len):
            labels = hf_chunk[f'metadata/pinpoints/pinpoints_{n}']
            yield labels[:]


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


def classify_trace_unsupervised(
        trace_file: str,
        vectorizer: BaseVectorizer,
        classifier: BaseClassifier,
        stride: int,
        window_size: int,
        epochs,
        tmp_folder,
        batch_size: int = 2000,
        stored_features_path: str = None,
        limit_traces: int = 0 #For debugging, 0 = use all
) -> np.ndarray:
    """
    Unsupervised sliding-window classification using a vectorizer and
    an unsupervised classifier, modeling the original workflow as close as possible


    Parameters
    ----------
    limit_traces
    single_trace
    epochs
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

    total_traces = total_traces if not limit_traces else min(limit_traces, total_traces)

    per_trace_scores = []
    feat_files = []          # list of npy file paths for each trace
    #-------------------------------------------
    # Skip vectorization a feature path is given
    #-------------------------------------------
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
                epochs=epochs,
                total_traces=total_traces,
            )

        #---------------------
        # Vectorize all traces
        #---------------------
        for trace_idx, trace in enumerate(tqdm(_dataLoader(trace_file), total=total_traces, colour="green", desc="Vectorizing traces")):
            trace = highpass(trace, 0.001)
            trace = (trace - np.mean(trace, axis=0)) / np.std(trace, axis=0)

            feats_batches = []

            for windows_batch in _extract_fixed_windows(trace, window_size, stride, batch_size):
                windows_arr = np.asarray(windows_batch).astype(np.float32)
                feats_batch = vectorizer.transform(windows_arr)
                feats_batches.append(feats_batch)

            X = np.vstack(feats_batches)

            #--------------------------------------------------------
            # Store features for this trace on disk instead of in RAM
            #--------------------------------------------------------
            trace_feat_path = os.path.join(
                tmp_folder, f"trace_{trace_idx}_feats.npy"
            )
            np.save(trace_feat_path, X)

            feat_files.append(trace_feat_path)
            del X, feats_batch, trace, windows_batch
            gc.collect()
            sleep(10)
            if trace_idx >= total_traces-1:
                break


        if hasattr(vectorizer, "global_avg_error"):
            print('Avg. reconstruction error:', vectorizer.global_avg_error)
            with open('loss.txt', 'w') as f:
                f.write(str(vectorizer.global_avg_error))
    #-------------------------------
    # Read stored features from disk
    #-------------------------------
    else:
        for i in range(0, total_traces):
            trace_feat_path = os.path.join(
                tmp_folder, stored_features_path, f"trace_{i}_feats.npy"
            )
            feat_files.append(trace_feat_path)

    #---------------------------------------------
    # Create a disk-backed memmap for all features
    #---------------------------------------------
    total_windows = 0
    feat_dim = None
    trace_windows = []  # windows per trace, aligned with feat_files order

    for path in feat_files:
        X = np.load(path, mmap_mode="r")
        trace_windows.append(X.shape[0])
        total_windows += X.shape[0]
        if feat_dim is None and X.shape[0] > 0:
            feat_dim = X.shape[1]


    all_X_path = os.path.join(tmp_folder, "all_X.mmap")
    all_X = np.memmap(all_X_path, dtype=np.float32, mode="w+", shape=(total_windows, feat_dim))

    offset = 0
    for path in feat_files:
        X = np.load(path, mmap_mode="r")
        n = X.shape[0]
        all_X[offset:offset+n] = X
        offset += n

    all_X.flush()

    #--------------
    # Fit classifier
    #---------------
    labels_path = os.path.join(tmp_folder, "all_labels.mmap")
    labels = np.memmap(labels_path, dtype=np.int64, mode="w+", shape=(total_windows,))

    trace_offsets = [0]
    offset = 0

    with h5py.File(trace_file, "r", libver="latest") as hf:
        for trace_idx in range(total_traces):
            trace_len = hf[f"data/traces/trace_{trace_idx}"].shape[0]
            pinpoints = hf[f"metadata/pinpoints/pinpoints_{trace_idx}"][:]


            labels_i = _window_labels_from_pinpoints(trace_len, pinpoints, window_size, stride)
            n = labels_i.shape[0]

            if n != trace_windows[trace_idx]:
                raise ValueError(
                    f"Trace {trace_idx}: labels windows {n} != feature windows {trace_windows[trace_idx]}"
                )


            labels[offset:offset+n] = labels_i
            offset += n
            trace_offsets.append(offset)
            if trace_idx >= total_traces-1:
                break

    labels.flush()

    classifier.fit(all_X, labels)

    #-----------------
    # Classify windows
    #-----------------
    metrics = []

    for trace_idx, path in enumerate(tqdm(
            feat_files,
            total=len(feat_files),
            colour="cyan",
            desc="Classifying globally-fitted",
    )):
        X = np.load(path, mmap_mode="r")

        scores = classifier.predict_proba(X) if hasattr(classifier, "predict_proba") else classifier.predict(X)
        per_trace_scores.append(scores)

        start, end = trace_offsets[trace_idx], trace_offsets[trace_idx + 1]
        y_true = labels[start:end]
        metrics.append(_window_metrics_from_scores(scores, y_true))
        if trace_idx >= total_traces-1:
            break

    print(f'Precision (CO): {np.mean([m["precision"] for m in metrics]):.4f}')
    print(f'TPR (CO recall): '
          f'{np.mean([m["TP"] / (m["TP"] + m["FN"]) if (m["TP"] + m["FN"]) else 0.0 for m in metrics]):.4f}')

    print(f'TNR (boundary recall): '
          f'{np.mean([m["TN"] / (m["TN"] + m["FP"]) if (m["TN"] + m["FP"]) else 0.0 for m in metrics]):.4f}')


    print(f'Balanced accuracy: {np.mean([m["balanced_accuracy"] for m in metrics]):.4f}')

    if hasattr(classifier, "cluster_to_class_"):
        print(f'Cluster-to-class mapping: {classifier.cluster_to_class_}')

    return np.stack(per_trace_scores, axis=0)


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



def _window_labels_from_pinpoints(
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
    min_overlap = 0.5 * window_size

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
        epochs: int,
        total_traces: int,

):
    for epoch in range(epochs):
        if hasattr(vectorizer, "on_epoch_start"):
            vectorizer.on_epoch_start(epoch)
        if hasattr(vectorizer, "global_avg_error"):
            old = vectorizer.global_avg_error

        pbar = tqdm(
            _dataLoader(trace_file),
            total=total_traces,
            colour="yellow",
            desc=f"Autoencoder epoch {epoch+1}/{epochs}",
            dynamic_ncols=True
        )

        for trace_idx, trace in enumerate(pbar):

            # -------------
            # Preprocessing
            # -------------
            trace = highpass(trace, 0.001)
            trace = (trace - np.mean(trace, axis=0)) / np.std(trace, axis=0)

            subsample_mul = 1
            if hasattr(vectorizer, "subsample_fraction"):
                subsample_mul = 1/vectorizer.subsample_fraction

            # -----------------------------
            # Train on all windows in trace
            # -----------------------------
            for windows_batch in _extract_fixed_windows(trace, window_size, stride, int(batch_size*subsample_mul)):
                windows_arr = np.asarray(windows_batch, dtype="float32")
                vectorizer.partial_fit(windows_arr)


            # --------------------------------
            # Store the error after this trace
            # --------------------------------
            if hasattr(vectorizer, "global_avg_error"):
                pbar.set_postfix({
                    "old_avg_err": f"{old:.4f}",
                    "avg_err": f"{vectorizer.global_avg_error:.4f}",
                })
                old = vectorizer.global_avg_error
            if trace_idx >= total_traces-1:
                break

        if hasattr(vectorizer, "on_epoch_end"):
            vectorizer.on_epoch_end(epoch)

    if hasattr(vectorizer, "save_autoencoder"):
        vectorizer.save_autoencoder()


def _window_metrics_from_scores(scores, y_true, positive_class=1, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_true_bin = (y_true == positive_class).astype(int)
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

    TP = np.count_nonzero((y_true_bin == 1) & (y_pred == 1))
    FP = np.count_nonzero((y_true_bin == 0) & (y_pred == 1))
    TN = np.count_nonzero((y_true_bin == 0) & (y_pred == 0))
    FN = np.count_nonzero((y_true_bin == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    recall_neg = TN / (TN + FP) if (TN + FP) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    bal_acc   = 0.5 * (
            (TP / (TP + FN) if (TP + FN) else 0.0) +
            (TN / (TN + FP) if (TN + FP) else 0.0)
    )

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision,
        "recall": recall,  # positive-class recall
        "recall_neg": recall_neg,  # negative-class recall (specificity)
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "y_score": y_score,
    }

