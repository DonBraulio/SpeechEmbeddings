# %%
import os
import glob
import sys
import cv2

# import pysptk
import torch
import numpy as np
import pandas as pd

# import librosa
import torchaudio
import matplotlib.pyplot as plt
import webrtcvad as wrtcvad
import speechbrain as sb

from pathlib import Path
from matplotlib import cm
from datetime import time, timedelta
from scipy.interpolate import interp1d
from rich import print
from rich.progress import track
from speechbrain.pretrained import EncoderClassifier
from speechbrain.lobes import features
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import OrdinalEncoder

# %%
files_recs = glob.glob("Datasets/Clases/*.mp4")
for idx, f in enumerate(files_recs):
    f_path = Path(f)
    files_recs[idx] = f_path
    print(f"{idx}: {f_path.name}")

f_input = files_recs[int(input("File index to use: "))]
print(f"[green]Selected:[/green] {f_input.name}")
input_audiofile = f_input.with_suffix(".wav")

# %%
# Extract WAV audio fragment from any input file type
start_time = timedelta(minutes=2, seconds=15)
max_duration = timedelta(seconds=60)
sample_rate = 16000
ffmpeg_cmd = (
    f"ffmpeg -y -i {f_input} -ss {start_time} -t {max_duration.seconds}"
    f" -c:a pcm_s16le -ar {sample_rate} {input_audiofile}"
)

if input("Run ffmpeg? [y/n]") == "y":
    print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)

# %%
print(f"Loading: {input_audiofile.name}")
signal, fs = torchaudio.load(input_audiofile)
assert fs == 16000  # Required by encode_batch

# Cut the signal to an integer number of seconds
signal = signal.squeeze()
duration_secs = len(signal) // fs
print(f"Loaded signal duration: {duration_secs}")
signal = signal[: duration_secs * fs]

# %%
# Numpy signal
y = signal.squeeze().numpy()
n_samples = len(y)
duration = n_samples / fs
t_start = start_time.seconds
t_end = t_start + duration
t = np.linspace(t_start, t_end, num=n_samples)

# %%
# Plot waveform (static)
def get_figure(n_axes):
    fig, axes = plt.subplots(nrows=n_axes, sharex=True, figsize=(30, 20))
    fig.tight_layout()
    return fig, axes


# %%
file_labels = f_input.with_suffix(".txt")

print(f"Loading labels from: {file_labels}")
df_labels = pd.read_csv(file_labels, sep="\t", names=["start", "end", "label"])

LABEL_NONE = "none"
LABEL_NONE_ID = 0
label_id_map = {
    label: id
    for id, label in enumerate([LABEL_NONE] + list(df_labels["label"].unique()))
}

# Fill gaps in labels with "none"
MAX_LABEL_GAP = 1
prev_end = t_start
gaps = []
for idx, row in df_labels.iterrows():
    gap = row["start"] - prev_end
    if gap > MAX_LABEL_GAP:
        gaps.append({"start": prev_end, "end": row["start"], "label": LABEL_NONE})
    prev_end = row["end"]
print(f"Gaps detected: {len(gaps)}")
df_labels = df_labels.append(gaps)

labels = np.array([label_id_map[label] for label in df_labels["label"]])

# Function to get nearest label at any time
t_labels = np.concatenate([df_labels["start"].values, df_labels["end"].values - 1e-3])
x_labels = np.concatenate([labels, labels])
# %%
eval_labels = interp1d(
    t_labels, x_labels, kind="nearest", bounds_error=False, fill_value=LABEL_NONE_ID
)

# %%
# SpeechBrain embeddings ecapa-tdnn in voxceleb
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
# batch = signal.reshape([-1, fs // 20])  # turn signal into batch of 100 msec wavs
# embeddings = classifier.encode_batch(batch).squeeze().T

# cmap = cm.get_cmap()
# mappable = axes[1].imshow(
#     embeddings, cmap=cmap, extent=[0, duration, 0, embeddings.shape[0]], aspect="auto"
# )
# %%
# VAD (Voice Activity Detector)
def float_to_pcm16(audio):
    ints = (audio * 32767).astype(np.int16)
    little_endian = ints.astype("<u2")
    buf = little_endian.tostring()
    return buf


buffer_size = int(20e-3 * sample_rate)  # 20 msec (webrtc accepts 10, 20 or 30ms)
vad = wrtcvad.Vad(3)  # 3 is most aggressive filtering
is_voice = np.zeros_like(y)
for start_idx in range(0, n_samples, buffer_size):
    end_idx = min(start_idx + buffer_size, n_samples)
    buffer_samples = y[start_idx:end_idx]
    vad_result = vad.is_speech(float_to_pcm16(buffer_samples), sample_rate)
    is_voice[start_idx:end_idx] = vad_result


# %%
# MFCC coeficients: STFT + Filterbank + DCT

# context=False is equivalent to setting left_frames and right_frames=0
mfcc_maker = features.MFCC(
    deltas=False,  # default: True
    context=False,  # default: True
    sample_rate=16000,
    f_min=0,
    f_max=None,
    n_fft=400,
    n_mels=23,  # default: 23,
    n_mfcc=20,  # default: 20,
    filter_shape="gaussian",  # default: "triangular",
    left_frames=0,  # default: 5
    right_frames=0,  # default: 5,
    win_length=25,
    hop_length=15,  # default: 10,
)
mfcc_signal = mfcc_maker(signal.unsqueeze(0))[0]
print(f"MFCC shape: {mfcc_signal.shape}")


# %%
# Time scale for mfcc coefs
t_mfcc = np.linspace(t_start, t_end, num=mfcc_signal.shape[0])

# Resample VAD to same size as mfcc
eval_vad = interp1d(t, is_voice, kind="nearest")
vad_mask = torch.Tensor(eval_vad(t_mfcc).astype(int))

# Set mfcc=0 in the parts where VAD=0
mfcc_vad = mfcc_signal * vad_mask.broadcast_to(mfcc_signal.shape[1], -1).T

# Mean of MFCC vector
mean_mfcc = mfcc_vad.mean(axis=0)

# Cosine similarity to mean
mfcc_cos = mfcc_vad.matmul(mean_mfcc) / (mfcc_vad.norm(dim=1) * mean_mfcc.norm() + 1e-6)


# %%
# Plot waveform, features and labels
fig, axes = get_figure(n_axes=5)
(wv_points,) = axes[0].plot(t, y)
axes[0].set_ylabel("Waveform")
axes[0].set_xlabel("time (s)")
axes[1].plot(t, is_voice, "r")
axes[1].set_ylabel("VAD")
axes[2].imshow(
    mfcc_signal.T,
    cmap=cm.get_cmap(),
    extent=[t_start, t_end, 0, mfcc_signal.shape[0]],
    aspect="auto",
    interpolation="none",
)
axes[3].plot(t_mfcc, mfcc_cos, "b")
axes[3].set_ylabel("cos(mfcc, mean_mfcc)")
axes[4].plot(t_mfcc, eval_labels(t_mfcc), "g")
axes[4].set_ylabel("label")
axes[4].set_yticks(list(label_id_map.values()))
axes[4].set_yticklabels(list(label_id_map.keys()))

# %%
# TODO
# predictor = TabularPredictor(label=COLUMN_NAME).fit(train_data=TRAIN_DATA.csv)
# predictions = predictor.predict(TEST_DATA.csv)


# %%
# Generate animated video from plots above
fps = 10
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

fig.canvas.draw()
base_frame = cv2.cvtColor(
    np.asarray(fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR
)
frame_shape = base_frame.shape[1::-1]

# %%
# Get matplotlib coordinates to draw progress line  (bottom-left is 0,0)
# xlim_start, xlim_end = axes[0].get_xlim()
xlim_start, xlim_end = wv_points.get_data()[0][[0, -1]]
ylim_top = axes[0].get_ylim()[1]
ylim_bottom = axes[-1].get_ylim()[0]  # Last plot, min Y value
(px_start, px_top) = axes[0].transData.transform((xlim_start, ylim_top))
(px_end, px_bottom) = axes[-1].transData.transform((xlim_end, ylim_bottom))
px_fig_width, px_fig_height = fig.canvas.get_width_height()

# Convert coordinates to refer them to top-left (matplotlib uses bottom-left as 0,0)
px_top = int(px_fig_height - px_top)
px_bottom = int(px_fig_height - px_bottom)

px_length = abs(px_end - px_start)
px_height = abs(px_bottom - px_top)
progress_color_bgr = (0, 0, 255)

# %%
f_features = f"features_{f_input.with_suffix('.mp4').name}"
try:
    os.remove(f_features)
except FileNotFoundError:
    pass

print(f"Frame shape: {frame_shape}")
n_frames = int(duration * fps + 0.5)
video = cv2.VideoWriter(f_features, fourcc, fps, frameSize=frame_shape)
for i in track(range(n_frames), description="Generating video..."):
    progress = i / n_frames

    frame = base_frame.copy()
    current_x = int(px_start + progress * px_length)
    cv2.line(frame, (current_x, px_top), (current_x, px_bottom), progress_color_bgr, 1)

    if frame.shape[1::-1] != frame_shape:
        print(f"New frame shape: {frame.shape[::-1]} | Init frame shape: {frame_shape}")
    video.write(frame)
video.release()

# %%
# Combine features (video) and audio into single video file
out_path = Path("out_videos/")
out_path.mkdir(exists_ok=True, parents=True)
out_filename = out_path / f_input.name

os.system(
    f"ffmpeg -y -i {f_features} -i {input_audiofile}"
    f" -c:v copy -c:a aac {out_filename}"
)

print(f"Output audio/features video: {out_filename}")
# %%
print(f"Removing auxiliary file: {f_features}")
os.remove(f_features)
