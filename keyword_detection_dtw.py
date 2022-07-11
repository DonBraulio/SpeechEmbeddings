# %%
import os
import torch
import time
import numpy as np
import pandas as pd
import torchaudio
import numba

from typing import Callable, Tuple
from torch.utils.data import Dataset
from rich.progress import track

from scipy.spatial import distance
import matplotlib.pyplot as plt

# Use this on a real jupyter notebook:
# from IPython.display import Audio
# Use this on vs-code notebook
from vscode_audio import Audio
from IPython.display import display

from pathlib import Path

from speechbrain.lobes.features import (
    STFT,
    Filterbank,
    DCT,
    spectral_magnitude,
)

# %%
class WordSTFTDataset(Dataset):
    def __init__(
        self,
        path_audio: Path,
        path_labels: Path,
        sort_labels: bool = False,
        fs: int = 16000,
        window_size: int = 1024,
        hop_size: int = 512,
    ):
        audio, self.fs = torchaudio.load(path_audio)
        assert self.fs == fs, f"File {path_audio} sample rate is {self.fs} != {fs}"

        self.audio = audio.squeeze()
        self.window_size = window_size
        self.hop_size = hop_size
        self.audio_stft = self.compute_stft(self.audio)

        self._df_labels = pd.read_csv(
            str(path_labels), sep="\t", names=["start", "end", "label"]
        )
        if sort_labels:
            self._df_labels = self._df_labels.sort_values(by="label").reset_index()

    def __len__(self) -> int:
        return len(self._df_labels)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        """
        Given a sample number, get STFT [win_length x n_fft x 2] and label
        """
        label = self.get_item_label(index)
        start_win, end_win = self.get_item_limits(index)
        return self.audio_stft[0, start_win:end_win, :, :], label

    def get_item_audio(self, index: int) -> np.ndarray:
        """
        Given a sample number, return its audio fragment
        """
        start_win, end_win = self.get_item_limits(index)
        n_sample_start = start_win * self.hop_size
        n_sample_end = end_win * self.hop_size + self.window_size
        return self.audio[n_sample_start:n_sample_end]

    def get_item_limits(self, index: int) -> Tuple[int, int]:
        """
        Given a sample number, return first and last window numbers
        """
        t_start, t_end = self._df_labels.iloc[index][["start", "end"]]
        return self.window_at(t_start), self.window_at(t_end)

    def get_item_label(self, index: int) -> str:
        return self._df_labels.iloc[index]["label"]

    def window_at(self, time: float) -> int:
        """
        Return number of the last window that covers this time position
        """
        return round(time * self.fs) // self.hop_size

    def compute_stft(self, audio):
        hop_length_ms = (self.hop_size / self.fs) * 1e3  # in ms
        win_length_ms = (
            self.window_size / self.fs
        ) * 1e3  # in ms, keep in range 20-30ms for voice
        # Time domain -> time/freq domain (STFT)
        compute_STFT = STFT(
            sample_rate=self.fs,
            n_fft=self.window_size,
            win_length=win_length_ms,
            hop_length=hop_length_ms,
        )
        with torch.no_grad():
            return compute_STFT(audio.unsqueeze(0))


class WordMFCCDataset(Dataset):
    """
    Wrapper for WordSTFTDataset that returns MFCC instead of STFT for each sample
    """

    def __init__(
        self, stft_dataset: WordSTFTDataset, f_min=60, f_max=8000, n_mels=60, n_mfcc=12
    ):
        self.stft_dataset = stft_dataset
        self.fs = self.stft_dataset.fs
        self.window_size = stft_dataset.window_size
        self.hop_size = stft_dataset.hop_size
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        audio_fbanks = self.compute_fbanks(self.stft_dataset.audio_stft)
        self.audio_mfcc = self.compute_mfcc(audio_fbanks).squeeze()

    def __len__(self) -> int:
        return len(self.stft_dataset)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        label = self.stft_dataset.get_item_label(index)
        win_start, win_end = self.stft_dataset.get_item_limits(index)
        return self.audio_mfcc[win_start:win_end, :], label

    def get_item_audio(self, index: int) -> np.ndarray:
        return self.stft_dataset.get_item_audio(index)

    def compute_fbanks(self, audio_stft):
        # magnitude(stft) -> Mel filterbank
        compute_fbanks = Filterbank(
            sample_rate=self.fs,
            n_fft=self.window_size,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            log_mel=True,
            filter_shape="triangular",
        )
        with torch.no_grad():
            return compute_fbanks(spectral_magnitude(audio_stft))

    def compute_mfcc(self, audio_fbanks):
        # Mel -> Cosine transform
        compute_dct = DCT(input_size=self.n_mels, n_out=self.n_mfcc)
        with torch.no_grad():
            return compute_dct(audio_fbanks)


# def cos_sim_matrix(x1, x2):
#     return (x1 @ x2.T) / (x1.norm(dim=1).unsqueeze(1) @ x2.norm(dim=1).unsqueeze(0))

# def euclidean_dist_matrix(x1, x2):
#     return (x1 @ x2.T) / (x1.norm(dim=1).unsqueeze(1) @ x2.norm(dim=1).unsqueeze(0))

# sim_matrix = cos_sim_matrix(mfcc_1, mfcc_2)


# %%
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"


@numba.jit(nopython=True)
def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.float64:
    return 1 - (x @ y) / np.sqrt((x @ x) * (y @ y))


@numba.jit(nopython=True)
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.float64:
    return np.sqrt(np.sum(np.power(x - y, 2)))


@numba.jit(nopython=True)
def dtw_table_numba(
    x: np.ndarray,
    y: np.ndarray,
    distance,
    accumulative: bool = False,
    diagonal_weight: float = 1.0,
):
    nx = len(x)
    ny = len(y)
    table = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    # Trick to avoid handling first col and row separately
    table[1:, 0] = np.inf
    table[0, 1:] = np.inf

    # Fill in the rest.
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            d = distance(x[i - 1], y[j - 1])
            table[i, j] = d
            if accumulative:
                table[i, j] += min(
                    table[i - 1, j],
                    table[i, j - 1],
                    diagonal_weight * table[i - 1, j - 1],
                )
    return table[1:, 1:]


def dtw_table(
    x: np.ndarray,
    y: np.ndarray,
    distance,
    accumulative: bool = False,
    diagonal_weight: float = 1,
):
    return dtw_table_numba(
        x.numpy(),
        y.numpy(),
        distance,
        accumulative=accumulative,
        diagonal_weight=diagonal_weight,
    )


# %%


def dtw(x, y, table):
    i = len(x) - 1
    j = len(y) - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        minval = table[i - 1, j - 1]
        step = (i - 1, j - 1)
        if i > 0 and table[i - 1, j] < minval:
            minval = table[i - 1, j]
            step = (i - 1, j)
        if j > 0 and table[i][j - 1] < minval:
            step = (i, j - 1)
        path.insert(0, step)
        i, j = step
    return np.array(path)


def plot_dtw_matrix(features, titles, distance, accumulative):
    x1, x2 = features
    title_1, title_2 = titles
    sim_matrix = dtw_table(x1, x2, distance, accumulative)

    # plot the best path on top of local similarity matrix
    fig = plt.figure(figsize=(9, 8))
    fig.patch.set_facecolor("white")

    # bottom right plot
    ax1 = plt.axes([0.2, 0, 0.8, 0.20])
    ax1.imshow(x1.T, origin="lower", aspect="auto", cmap="coolwarm")
    ax1.set_xlabel(title_1)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # top left plot
    ax2 = plt.axes([0, 0.2, 0.20, 0.8])
    ax2.imshow(x2.flip(dims=(1,)), origin="lower", aspect="auto", cmap="coolwarm")
    ax2.set_ylabel(title_2)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # top right plot
    ax3 = plt.axes([0.2, 0.2, 0.8, 0.8])
    ax3.imshow(
        sim_matrix.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="gray",
    )
    ax3.set_xticks([])
    ax3.set_yticks([])

    # DTW path
    if accumulative:
        dtw_path = dtw(x1, x2, sim_matrix)
        ax3.plot(dtw_path[:, 0], dtw_path[:, 1], "r")
    return sim_matrix


def compare_utterances(
    ds1,
    ds2,
    idx1,
    idx2,
    distance=cosine_distance,
    accumulative=True,
    compare_stft=False,
    path_save_wav: Path = None,
):
    # Similarity matrix using STFT
    if compare_stft:
        stft_1, label_1 = ds1.stft_dataset[idx1]
        stft_2, label_2 = ds2.stft_dataset[idx2]
        features_1 = np.log10(spectral_magnitude(stft_1))
        features_2 = np.log10(spectral_magnitude(stft_2))
        title_1 = f'Spectrogram of sample {idx1}: "{label_1}"'
        title_2 = f'Spectrogram of sample {idx2}: "{label_2}"'
        _ = plot_dtw_matrix(
            [features_1, features_2], [title_1, title_2], distance, accumulative
        )

    # DTW using MFCC
    mfcc_1, label_1 = ds1[idx1]
    mfcc_2, label_2 = ds2[idx2]
    skip_coef = 1  # check this coef
    features_1 = mfcc_1[:, skip_coef:]
    features_2 = mfcc_2[:, skip_coef:]
    title_1 = f'MFCCs of sample {idx1}: "{label_1}"'
    title_2 = f'MFCCs of sample {idx2}: "{label_2}"'
    sim_matrix = plot_dtw_matrix(
        [features_1, features_2], [title_1, title_2], distance, accumulative
    )

    if accumulative:
        # Normalize using diagonal size of the matrix
        normalizator = np.sqrt(np.sum(np.power(sim_matrix.shape, 2)))
        dtw_dist = sim_matrix[-1][-1]
        print(f"DTW distance: {dtw_dist} | Normalized: {dtw_dist / normalizator}")
    print(f"labels={ds1[idx1][1]},{ds2[idx2][1]}")
    audio_comparison = torch.concat(
        [ds1.get_item_audio(idx1), ds2.get_item_audio(idx2)]
    )
    display(Audio(audio_comparison, rate=fs))

    # Save comparison if required
    if path_save_wav is not None:
        path_save_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(path_save_wav, audio_comparison.unsqueeze(0), fs)

    return sim_matrix


def concat_samples(ds, idxs, path_save_wav: Path = None):
    silence = torch.zeros(fs // 2)
    concat_list = []
    for idx in idxs:
        concat_list.append(ds.get_item_audio(idx))
        concat_list.append(silence)
    concat_audio = torch.concat(concat_list)
    if path_save_wav is not None:
        path_save_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(path_save_wav, concat_audio.unsqueeze(0), fs)
    display(Audio(concat_audio, rate=fs))


# %%
path_audios = Path("data/audios")
path_labels = Path("data/labels")

# Datasets for both speakers
fs = 16000
n_mfcc = 8
ds_stft_sp1 = WordSTFTDataset(
    path_audios / "english_1.wav",
    path_labels / "english_1.txt",
    sort_labels=True,
    window_size=512,
    hop_size=128,
    fs=fs,
)
ds_stft_sp2 = WordSTFTDataset(
    path_audios / "english_2.wav",
    path_labels / "english_2.txt",
    sort_labels=True,
    window_size=512,
    hop_size=128,
    fs=fs,
)
ds_sp1 = WordMFCCDataset(ds_stft_sp1, n_mfcc=n_mfcc)
ds_sp2 = WordMFCCDataset(ds_stft_sp2, n_mfcc=n_mfcc)

# %%
# Show speaker 1 samples
all_utters = []
silence = torch.zeros(fs // 2)
for idx, (mfccs, label) in enumerate(ds_sp1):
    print(f"sp1 {idx}\t| shape={tuple(mfccs.shape)}\t| {label} ")
    all_utters.append(ds_sp1.get_item_audio(idx))
    all_utters.append(silence)
torchaudio.save("speaker_1_samples.wav", torch.concat(all_utters).unsqueeze(0), fs)
# %%
all_utters = []
# Show speaker 2 samples
for idx, (mfccs, label) in enumerate(ds_sp2):
    print(f"sp2 {idx}\t| shape={tuple(mfccs.shape)}\t| {label} ")
    all_utters.append(ds_sp2.get_item_audio(idx))
    all_utters.append(silence)
torchaudio.save("speaker_2_samples.wav", torch.concat(all_utters).unsqueeze(0), fs)

# %% [markdown]
# %%
def dtw_distance(
    x1, x2, feature_distance, max_ratio, normalize=True, diagonal_weight=1.0
):
    # If frame lengths are too different, don't compare them
    shape_ratio = x1.shape[0] / x2.shape[0]
    if max_ratio > shape_ratio or shape_ratio > 1 / max_ratio:
        return np.inf

    dtw_matrix = dtw_table(
        x1,
        x2,
        distance=feature_distance,
        accumulative=True,
        diagonal_weight=diagonal_weight,
    )
    dist = dtw_matrix[-1][-1]  # farthest corner of dtw matrix

    # Normalize by the hypothenuse length
    if normalize:
        dist /= np.sqrt(np.sum(np.power(dtw_matrix.shape, 2)))

    return dist


def calculate_dtw_distances(
    ds1,
    ds2=None,
    f_distance=cosine_distance,
    skip_coef=1,
    max_ratio=0.001,
    diagonal_weight=1.0,
):
    same_dataset = ds2 is None
    if same_dataset:
        ds2 = ds1
    t0 = time.time()
    dtw_distances = np.zeros((len(ds1), len(ds2)))
    labels_1 = []
    labels_2 = []
    total_ops = 0
    for idx1, (mfccs1, label1) in track(enumerate(ds1), total=len(ds1)):
        labels_1.append(f"{label1}_{idx1}[{mfccs1.shape[0]}]")
        for idx2, (mfccs2, label2) in enumerate(ds2):
            if idx2 == len(labels_2):
                labels_2.append(f"{label2}_{idx2}[{mfccs2.shape[0]}]")
            # same_dataset: symmetric matrix, avoid recalculating above diagonal
            if same_dataset and idx2 < idx1:
                dtw_distances[idx1, idx2] = dtw_distances[idx2, idx1]
                continue

            # Elements below diagonal: calculate dtw distance and normalize
            dtw_distances[idx1, idx2] = dtw_distance(
                mfccs1[:, skip_coef:],
                mfccs2[:, skip_coef:],
                f_distance,
                max_ratio,
                diagonal_weight=diagonal_weight,
            )
            total_ops += 1

    total_time = time.time() - t0
    avg_time = total_time / total_ops
    print(
        f"Calculated {total_ops} DTW distances | Total time: {total_time:.1f}s |  Avg time: {avg_time:.4f}s"
    )
    return dtw_distances, np.array(labels_1), np.array(labels_2)


# %%
def plot_distance_matrix(distances_matrix, labels_1, labels_2):
    labels_1 = np.array(labels_1)
    labels_2 = np.array(labels_2)
    sort_1 = np.argsort(labels_1)
    sort_2 = np.argsort(labels_2)
    plt.figure(figsize=(10, 10))
    plt.imshow(
        distances_matrix[sort_1, :][:, sort_2].T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
    )
    _ = plt.xticks(ticks=range(len(labels_1)), labels=labels_1[sort_1], rotation=90)
    _ = plt.yticks(ticks=range(len(labels_2)), labels=labels_2[sort_2])


# %%
# feature distance (between STFT or MFCCs)
# f_distance = euclidean_distance
f_distance = cosine_distance
diagonal_weight = 1.0

# Speaker 1 against himself
dtw_distances_sp1, labels_sp1, _ = calculate_dtw_distances(
    ds_sp1, f_distance=f_distance, max_ratio=0.001, diagonal_weight=diagonal_weight
)
plot_distance_matrix(dtw_distances_sp1, labels_sp1, labels_sp1)

# %%
# Speaker 2 against himself
dtw_distances_sp2, labels_sp2, _ = calculate_dtw_distances(
    ds_sp2, f_distance=f_distance, max_ratio=0.001, diagonal_weight=diagonal_weight
)
plot_distance_matrix(dtw_distances_sp2, labels_sp2, labels_sp2)

# %%[markdown]
"""
### Some examples for report
"""
# %%

# Select a pair of audios from same speaker
# Same utterance (calibrate matrix)
dtw_matrix = compare_utterances(
    ds_sp1, ds_sp1, 0, 0, f_distance, accumulative=False, compare_stft=True
)

# %%
samples_path = Path("report_audio_samples")
# %%
# Report: STFT vs. MFCCs (same word, different tone)
# Both listen(s)
dtw_matrix = compare_utterances(
    ds_sp1,
    ds_sp1,
    19,
    21,
    f_distance,
    accumulative=False,
    compare_stft=True,
    path_save_wav=samples_path / "sp1_listen19_listen21.wav",
)

# %%
# Report: illustrative of why paths can be quite warped
# 9: greeeeaaat
_ = compare_utterances(
    ds_sp2,
    ds_sp2,
    5,
    9,
    compare_stft=False,
    path_save_wav=samples_path / "sp2_great5_great9.wav",
)


# %% [markdown]
"""
### Analysis of some particular cases
"""
# %%
# Report: why are hello(s) so different/similar between them
# hello 13 is similar to 10, 11, 15 and 17, but quite different from 12 and 16
# Answer: 12 and 16 have a lot of background noise (gritos niños alta frecuencia)
concat_samples(
    ds_sp1, [12, 16, 13, 15], path_save_wav=samples_path / "sp1_hellos_12_16_13_15.wav"
)

# %%
# Report: what about great 7 and 8
# Answer: great 7 has background noise
concat_samples(ds_sp1, [7, 8], path_save_wav=None)

# %%
# Report: why on earth listen_22 and good 6 are near?
# good_6 is similar to listen 22, hello 13, hello 15, ok 34, and 28, but different from 29 and 27
concat_samples(
    ds_sp1, [6, 22, 15, 34], path_save_wav=samples_path / "sp1_good_6_similars.wav"
)
# Answer: all examples above seem to have low frequency background noise


# %%
# Speaker 1: Good and bad examples of "excellent"
# Best examples of listen
dtw_matrix = compare_utterances(
    ds_sp1,
    ds_sp1,
    20,
    21,
    f_distance,
    accumulative=True,
    path_save_wav=samples_path / "sp1_listen20_listen21.wav",
)
# %%
# Worse examples of listen
dtw_matrix = compare_utterances(
    ds_sp1,
    ds_sp1,
    18,
    24,
    f_distance,
    accumulative=True,
    path_save_wav=samples_path / "sp1_listen24_listen18.wav",
)

# %%
# Report: noisy samples for sp2
concat_samples(
    ds_sp2, [3, 16, 17, 35], path_save_wav=samples_path / "sp2_3_16_17_35.wav"
)

# %%
# Speaker 2: Compare noisy excellent_3 vs. excellent 4
# Different lengths
dtw_matrix = compare_utterances(ds_sp2, ds_sp2, 3, 4, f_distance, accumulative=True)
# %%
# Very good from different speakers, low distance
dtw_matrix = compare_utterances(
    ds_sp1,
    ds_sp2,
    43,
    36,
    f_distance,
    accumulative=True,
    compare_stft=False,
    path_save_wav=None,
)

# %%
# However, very good and ok are also low distance
dtw_matrix = compare_utterances(
    ds_sp1,
    ds_sp2,
    28,
    36,
    f_distance,
    accumulative=True,
    compare_stft=False,
    path_save_wav=None,
)
# %%
# Speaker 1 against speaker 2
dtw_distances_sp12, labels_sp121, labels_sp122 = calculate_dtw_distances(ds_sp1, ds_sp2)
plot_distance_matrix(dtw_distances_sp12, labels_sp121, labels_sp122)


# %%
def clean_labels(labels):
    return np.array(["_".join(label.split("_")[:-1]) for label in labels])


def nearest_matches(dtw_distances, labels_1, labels_2=None, kth=3):
    single_speaker = labels_2 is None
    if single_speaker:
        labels_2 = labels_1
    assert (len(labels_1), len(labels_2)) == dtw_distances.shape
    labels_1_clean = clean_labels(labels_1)
    labels_2_clean = clean_labels(labels_2)

    rows = []
    rows_match = np.zeros((len(labels_1), kth), dtype=bool)
    kth_l = list(range(kth))
    for idx, label in enumerate(labels_1):
        dists = dtw_distances[idx]
        if single_speaker:
            dists = np.copy(dists)
            dists[idx] = np.inf  # Don't match aganist itself
        k_nearest = np.argpartition(dists, kth=kth_l)[:kth]
        row = {"label": label}
        row.update(
            {k: f"{labels_2[k_nearest][k]} ({dists[k_nearest][k]:.2f})" for k in kth_l}
        )
        rows.append(row)

        # Boolean matrix with 1's where labels match for the k-nearest elements
        rows_match[idx, :] = labels_2_clean[k_nearest] == labels_1_clean[idx]

    df_nearest = pd.DataFrame(data=rows)
    return df_nearest, rows_match


def plot_matches(rows_match, labels):
    plt.figure(figsize=(10, 0.3 + 0.3 * rows_match.shape[1]))
    colors = np.array([[255, 91, 66], [148, 255, 66]])  # Red, green
    # rows_match_colored = np.zeros(rows_match.shape + (3,))
    rows_match = rows_match.T
    rows_match_colored = colors[rows_match.flatten().astype(int)].reshape(
        rows_match.shape + (3,)
    )
    plt.imshow(
        rows_match_colored,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdYlGn",
    )
    _ = plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90)
    plt.yticks(range(0, rows_match.shape[0]))
    plt.ylabel("Nearest words match")


def extract_label(label):
    # Extract label_n[length].. -> label, n, length
    clean_label = "_".join(label.split("_")[:-1])
    n = int(label.split("_")[-1].split("[")[0])
    length = int(label.split("[")[1].split("]")[0])
    return clean_label, n, length


def df_distances(dtw_distances, labels_1, labels_2=None, max_length_ratio=np.inf):
    single_speaker = labels_2 is None
    if single_speaker:
        labels_2 = labels_1
    assert (len(labels_1), len(labels_2)) == dtw_distances.shape

    rows = []
    for idx, label in enumerate(labels_1):
        for idx2, label2 in enumerate(labels_2):
            if single_speaker and idx == idx2:
                continue
            dist = dtw_distances[idx, idx2]
            word_1, n_samp1, length_1 = extract_label(label)
            word_2, n_samp2, length_2 = extract_label(label2)
            row = {
                "label_1": word_1,
                "label_2": word_2,
                "distance": dist,
                "n_sample_1": n_samp1,
                "n_sample_2": n_samp2,
                "length_1": length_1,
                "length_2": length_2,
            }
            rows.append(row)
    df_dist = pd.DataFrame(data=rows)

    # matching labels
    df_dist["is_match"] = df_dist["label_1"] == df_dist["label_2"]

    # length ratios
    lengths = df_dist[["length_1", "length_2"]]
    df_dist["length_ratio"] = lengths.max(axis=1) / lengths.min(axis=1)
    df_dist = df_dist[df_dist["length_ratio"] <= max_length_ratio]

    # match ranking: for each label_1, assign label_2 ranking by min distance
    df_dist = (
        df_dist.groupby(["label_1", "n_sample_1"])
        .apply(
            lambda x: x.sort_values(by="distance").assign(match_ranking=range(len(x)))
        )
        .reset_index(drop=True)
    )
    return df_dist


def plot_distance_hist(df_dist, title):
    plt.figure()
    plt.hist(
        df_dist[df_dist["is_match"]]["distance"], alpha=0.5, label="Match", density=True
    )
    plt.hist(
        df_dist[~df_dist["is_match"]]["distance"],
        alpha=0.5,
        label="No match",
        density=True,
    )
    plt.legend(loc="upper right")
    plt.title(title)


def accuracy(df_eval):
    return (df_eval["label_1"] == df_eval["label_2"]).sum() / len(df_eval)


def print_accuracy(df_eval):
    global_accuracy = accuracy(df_eval)
    print(f"Global accuracy: {global_accuracy}")
    print(df_eval.groupby("label_1").apply(accuracy))


# %%
# ## Accuracy por speakers
df_dist_1 = df_distances(dtw_distances_sp1, labels_sp1, max_length_ratio=np.inf)

df_dist_2 = df_distances(dtw_distances_sp2, labels_sp2, max_length_ratio=np.inf)

# Considerando sólo top-1
print("Speaker 1 accuracy")
print_accuracy(df_dist_1[df_dist_1["match_ranking"] == 0])

print("\n\nSpeaker 2 accuracy")
print_accuracy(df_dist_2[df_dist_2["match_ranking"] == 0])


# %%
## Precision-recall
def precision_recall(df_dist):
    # Only take smallest distance and check if it's above threshold
    df = df_dist[df_dist["match_ranking"] == 0]

    curve_pr = []  # precision-recall curve
    for threshold in df["distance"].sort_values():
        df_valid = df[df["distance"] <= threshold]
        precision = accuracy(df_valid)
        recall = len(df_valid) / len(df)
        curve_pr.append(
            {"threshold": threshold, "precision": precision, "recall": recall}
        )
    return pd.DataFrame(curve_pr)


def plot_precision_recall_curve(curve_pr, label, f=None):
    if f is None:
        f = plt.figure()
    plt.plot(curve_pr["recall"], curve_pr["precision"], label=label)
    plt.ylim([0, 1.1])
    plt.xlim([0, 1.1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    return f


curve_pr1 = precision_recall(df_dist_1)
fig = plot_precision_recall_curve(curve_pr1, "Speaker 1")

curve_pr2 = precision_recall(df_dist_2)
plot_precision_recall_curve(curve_pr2, "Speaker 2", fig)


# %%
df_dist_1["length_ratio"].hist()
# %%
plot_distance_hist(df_dist_1, "Speaker 1: Histograma de distancias")
plot_distance_hist(
    df_dist_1[df_dist_1["match_ranking"] < 1],
    "Speaker 1: Histograma de distancias top-1",
)
plot_distance_hist(
    df_dist_1[df_dist_1["match_ranking"] < 3],
    "Speaker 1: Histograma de distancias top-3",
)
plot_distance_hist(
    df_dist_1[df_dist_1["match_ranking"] < 5],
    "Speaker 1: Histograma de distancias top-5",
)

# %%
df_nearest_1, rows_match_1 = nearest_matches(dtw_distances_sp1, labels_sp1, kth=5)
df_nearest_1
# %%
plot_matches(rows_match_1, labels_sp1)

# %% [markdown]
# ## Evaluation of speaker 2
# %%
df_nearest_2, rows_match_2 = nearest_matches(dtw_distances_sp2, labels_sp2, kth=5)
df_nearest_2

# %%
plot_matches(rows_match_2, labels_sp2)

# %%

# %%
plot_distance_hist(df_dist_2, "Speaker 2: Histograma de distancias")
plot_distance_hist(
    df_dist_2[df_dist_2["match_ranking"] < 1],
    "Speaker 2: Histograma de distancias top-1",
)
plot_distance_hist(
    df_dist_2[df_dist_2["match_ranking"] < 3],
    "Speaker 2: Histograma de distancias top-3",
)
plot_distance_hist(
    df_dist_2[df_dist_2["match_ranking"] < 5],
    "Speaker 2: Histograma de distancias top-5",
)

# %%
