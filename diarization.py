# %%
import sys
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt

from pathlib import Path
from umap import UMAP
from sys import stderr
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from time import sleep, perf_counter as timer

sys.path.append("Resemblyzer")
from resemblyzer import preprocess_wav, VoiceEncoder, sampling_rate, audio
from demo_utils import play_wav

# %% [markdown]
"""
### Convert audio file to WAV @ 16kHz
"""
wav_fpath = Path("./", "guitar_annoying.wav")
wav = preprocess_wav(wav_fpath)
wav.shape

# %%
play_wav(wav)

# %% [markdown]
"""
### Show raw samples from signal
"""
n_samples = wav.shape[0]
total_time_s = n_samples / sampling_rate
t = np.arange(0, total_time_s, 1 / sampling_rate)

plt.figure()
plt.title(f"Raw wav ({wav.shape[0]} samples @ {sampling_rate} Hz)")
plt.plot(t, wav, 'b-')


# %% [markdown]
"""
### Fourier Spectrogram
"""
window_length = int(0.1 * sampling_rate)
window_step = int(0.02 * sampling_rate)

fourier = np.abs(librosa.stft(wav, n_fft=window_length, hop_length=window_length))
fourier_db = librosa.amplitude_to_db(fourier, ref=np.max)

_, ax = plt.subplots()
mappable = ax.imshow(fourier_db, cmap=cm.get_cmap(), aspect='auto')
cbar = plt.colorbar(mappable, ax=ax)
ax.set_title("Fourier Spectrogram")

# %% [markdown]
"""
### Mel Spectrogram
"""
mel_n_channels = 800
mel = librosa.feature.melspectrogram(
    S=fourier,
    # wav,  # TODO: why antialiased?
    # sampling_rate,
    n_fft=int(sampling_rate * window_length),
    hop_length=int(sampling_rate * window_step),
    n_mels=mel_n_channels
)
mel_db = librosa.amplitude_to_db(mel, ref=np.max)

_, ax = plt.subplots()
mappable = ax.imshow(mel_db, cmap=cm.get_cmap(), aspect='auto')
cbar = plt.colorbar(mappable, ax=ax)
ax.set_title("Mel Spectrogram")

# %% [markdown]
"""
### Calculate embeddings: Mel energies + LSTM + Linear layer
"""

wav_both_speakers = preprocess_wav("easy.wav")  # WAV with voices

encoder = VoiceEncoder("cpu")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav_both_speakers, return_partials=True, rate=16)


# %%
def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    if ax is None:
        _, ax = plt.subplots()

    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)

    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)


plot_embedding_as_heatmap(cont_embeds.T, shape=cont_embeds.T.shape)

# # %% [markdown]
# """
# ### Isolate some utterances for each speaker
# """
# speaker1_1_idx = 0
# speaker2_1_idx = 15
# speaker1_2_idx = 40
# speaker2_2_idx = 80
# speaker1_3_idx = 105
# speaker2_3_idx = 125
# speaker1_4_idx = 150
# speaker1_5_idx = 170

# speaker1_1 = wav_both_speakers[wav_splits[speaker1_1_idx]]
# speaker2_1 = wav_both_speakers[wav_splits[speaker2_1_idx]]
# speaker1_2 = wav_both_speakers[wav_splits[speaker1_2_idx]]
# speaker2_2 = wav_both_speakers[wav_splits[speaker2_2_idx]]
# speaker1_3 = wav_both_speakers[wav_splits[speaker1_3_idx]]
# speaker2_3 = wav_both_speakers[wav_splits[speaker2_3_idx]]
# speaker1_4 = wav_both_speakers[wav_splits[speaker1_4_idx]]
# speaker1_5 = wav_both_speakers[wav_splits[speaker1_5_idx]]

# # %%
# for label, speaker_wav in [
#     ("Speaker 1 / utterance 1", speaker1_1),
#     ("Speaker 2 / utterance 1", speaker2_1),
#     ("Speaker 1 / utterance 2", speaker1_2),
#     ("Speaker 2 / utterance 2", speaker2_2),
#     ("Speaker 1 / utterance 3", speaker1_3),
#     ("Speaker 2 / utterance 3", speaker2_3),
#     ("Speaker 1 / utterance 4", speaker1_4),
#     ("Speaker 1 / utterance 5", speaker1_5),
# ]:
#     print(f"Playing: {label}...")
#     play_wav(speaker_wav)

# # %% [markdown]
# """
# ### Build similarity matrix
# """
# all_speakers = [speaker1_1_idx, speaker2_1_idx,
#                 speaker1_2_idx, speaker2_2_idx,
#                 speaker1_3_idx, speaker2_3_idx,
#                 speaker1_4_idx, speaker1_5_idx]
# similarity_m = np.zeros((len(all_speakers), 2))
# row = 0
# for utterance_idx in all_speakers:
#     similarity_m[row, 0] = cont_embeds[utterance_idx, :] @ cont_embeds[speaker1_1_idx, :]
#     similarity_m[row, 1] = cont_embeds[utterance_idx, :] @ cont_embeds[speaker2_1_idx, :]
#     row += 1

# _, ax = plt.subplots()
# mappable = ax.imshow(similarity_m, cmap=cm.get_cmap())
# cbar = plt.colorbar(mappable, ax=ax)
# ax.set_title("Similarities")
# # %%

# %%
