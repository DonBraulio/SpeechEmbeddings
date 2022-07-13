# %%[markdown]
"""
# Speaker embeddings similarity matrix

Notebook to test Embedding similarity

Goal: given an audio (and optionally labels), generate similarity
matrix

Speaker embeddings based on:
https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
"""
# %%
import torch
import torchaudio
import numpy as np
import pandas as pd
import numba
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier

from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from rich.progress import track

from speechbrain.lobes.features import (
    STFT,
    Filterbank,
    DCT,
    spectral_magnitude,
)

# Use this on a real jupyter notebook:
# from IPython.display import Audio
# Use this on vs-code notebook
from vscode_audio import Audio
from IPython.display import display


# %%
class AudioDataset(Dataset):
    """
    Given one or multiple audio files, and optional labels for each audio,
    concatenate all audios and generate samples with the given window and hop sizes,
    where each window is a sample with its corresponding text label (or None).

    Each audio file is trimmed to fit last window before concatenation.
    Label is set to None on windows that don't fit entirely within a label [start, end].
    """

    def __init__(
        self,
        folder_audios: Path | str,
        folder_labels: Path | str | None = None,
        glob_audios: str = "*.wav",
        fs: int = 16000,
        window_size: int = 1024,
        hop_size: int = 512,
    ):
        self.window_size = window_size
        self.hop_size = hop_size
        self.fs = fs

        self.total_windows = 0
        audios = []
        dfs_labels = []
        time_offset = 0.0
        for idx_audio, path_audio in enumerate(Path(folder_audios).glob(glob_audios)):

            # Load audio
            audio, fs = torchaudio.load(path_audio)
            assert self.fs == fs, f"File {path_audio} sample rate is {fs} != {self.fs}"

            # Load labels for this audio file (if exists)
            df_labels = None
            if folder_labels is not None:
                path_labels = Path(folder_labels) / path_audio.with_suffix(".txt").name
                if path_labels.is_file():
                    df_labels = pd.read_csv(
                        str(path_labels), sep="\t", names=["start", "end", "label"]
                    )
                    df_labels[["start", "end"]] += time_offset
                    dfs_labels.append(df_labels)

            # Fit audio to integer number of windows and prepare for next audio
            audio = audio.squeeze()
            n_windows = (len(audio) - window_size) // hop_size + 1
            n_samples = n_windows * window_size
            audio = audio[:n_samples]  # trim to windows
            duration = n_samples / fs
            time_offset += duration
            print(
                f"Loaded {idx_audio}, labels: "
                f"{'None' if df_labels is None else len(df_labels)}, "
                f"duration: {duration:.2f}s, "
                f"#windows: {n_windows}, from: {path_audio}"
            )
            audios.append(audio)
            self.total_windows += n_windows  # Total length

        # Concatenate all loaded audios after trimming
        self.audio = torch.concat(audios, axis=0)
        total_duration = len(self.audio) / fs
        print(f"Total duration: {total_duration:.2f}s ({self.total_windows} windows)")

        # Convert labels from dataframe[start_time, end_time, label] -> list[idx, label]
        self.labels = [""] * self.total_windows
        if dfs_labels:
            df_labels = pd.concat(dfs_labels, axis=0, ignore_index=True)
            assert total_duration > df_labels["start"].max()
            assert total_duration > df_labels["end"].max()

            # Iterate over windows and set their labels when fit
            df_idx = 0
            window_size_secs = window_size / fs
            for idx in range(self.total_windows):
                # Current window limits
                start_time = (idx * hop_size) / fs
                end_time = start_time + window_size_secs

                # Advance index in labels df when end time is reached
                while end_time > df_labels.iloc[df_idx]["end"] and df_idx < len(
                    df_labels
                ):
                    df_idx += 1

                # Set label for this window if it fits entirely
                if (
                    start_time >= df_labels.iloc[df_idx]["start"]
                    and end_time <= df_labels.iloc[df_idx]["end"]
                ):
                    self.labels[idx] = df_labels.iloc[df_idx]["label"]

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        """
        Get the audio fragment of the window at position 'index'
        and its label (None if window is not contained entirely within label)
        """
        start_pos = index * self.hop_size
        end_pos = start_pos + self.window_size
        return self.audio[start_pos:end_pos], self.labels[index]


# %%
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# %%
fs = 16000
win_secs = 2  # Pretty matrix: 2 seconds windows
hop_secs = 1
window_size = round(win_secs * fs)
hop_size = round(hop_secs * fs)
audio_ds = AudioDataset("Datasets/samples", glob_audios="clase_2_1*", window_size=window_size, hop_size=hop_size, fs=fs)
display(Audio(audio_ds.audio, rate=audio_ds.fs))

# %%
%%time
# Single batch for all dataset
batch_size = len(audio_ds)
audio_dl = DataLoader(audio_ds, batch_size, shuffle=False)
for batch, labels in audio_dl:
    encoded_batch = classifier.encode_batch(batch, normalize=False).squeeze()
    # Normalize to norm=1
    encoded_batch /= torch.norm(encoded_batch, dim=1, keepdim=True)
    print("Batch encoded")

# %%
similarity_matrix = encoded_batch @ encoded_batch.T
# %%
fig = plt.figure(figsize=(10, 10))
f = plt.imshow(
    similarity_matrix,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    cmap="coolwarm",
)
fig.colorbar(f)
display(Audio(audio_ds.audio, rate=audio_ds.fs))

# %%
