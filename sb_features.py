# %%
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio
from speechbrain.processing.features import STFT


# %%
audio_file = "PC_1107003_5A_7112019_REC-2019-11-07T14_15_00Z.wav"
signal = read_audio(audio_file).squeeze()
Audio(signal, rate=16000)

# %%
compute_STFT = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
signal_STFT = compute_STFT(signal)  # [batch, time, channel1, channel2]

# %%
