# %%
import os
import glob
import sys
import cv2

# import pysptk
import numpy as np

# import librosa
import torchaudio
import matplotlib.pyplot as plt
import webrtcvad as wrtcvad
import speechbrain as sb

from pathlib import Path
from matplotlib import cm
from rich import print
from rich.progress import track
from speechbrain.pretrained import EncoderClassifier
from speechbrain.lobes import features

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
start_time = 10
max_duration = 20
sample_rate = 16000
ffmpeg_cmd = (
    f"ffmpeg -y -i {f_input} -ss {start_time} -t {max_duration}"
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
# Load wav file as signal
# y, fs = librosa.load(input_audiofile, sr=sample_rate)
# assert fs == sample_rate
y = signal.squeeze().numpy()
n_samples = len(y)
duration = n_samples / fs
t = np.linspace(0, duration, num=n_samples)

# %%
# Plot waveform (static)
fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(30, 20))
(wv_points,) = axes[0].plot(t, y)
axes[0].set_ylabel("Waveform")
axes[0].set_xlabel("time (s)")
# progress_lines.append(ax1.axvline(x=0, color="r"))

# Plot fundamental frequency
# f0 = pysptk.swipe(y.astype(np.float64), fs=fs, hopsize=80, min=30, max=200, otype="f0")
# f0_t = np.linspace(0, duration, num=len(f0))
# axes[1].plot(f0_t, f0, "g")
# axes[1].set_ylabel("Fundamental freq (f0)")

# progress_lines.append(ax2.axvline(x=0, color="r", linestyle="--"))

# SpeechBrain embeddings ecapa-tdnn in voxceleb
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
batch = signal.reshape([-1, fs // 20])  # turn signal into batch of 100 msec wavs

embeddings = classifier.encode_batch(batch).squeeze().T

cmap = cm.get_cmap()
mappable = axes[1].imshow(
    embeddings, cmap=cmap, extent=[0, duration, 0, embeddings.shape[0]], aspect="auto"
)
# plt.colorbar(mappable, ax=ax)  # , fraction=0.046, pad=0.04)
# ax.set_xticks([]), ax.set_yticks([])

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
    # print(f"{start_idx}:{end_idx} buffer.shape: {buffer_samples.shape}")
    vad_result = vad.is_speech(float_to_pcm16(buffer_samples), sample_rate)
    is_voice[start_idx:end_idx] = vad_result
    # print(f"{start_idx}:{end_idx} -> {vad_result}")

axes[2].plot(t, is_voice, "r")
axes[2].set_ylabel("VAD")

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
    # current_pos = duration * progress

    frame = base_frame.copy()
    current_x = int(px_start + progress * px_length)
    cv2.line(frame, (current_x, px_top), (current_x, px_bottom), progress_color_bgr, 1)

    # for lp in progress_lines:
    #     lp.set_xdata(current_pos)
    # fig.canvas.draw()
    # frame = cv2.cvtColor(
    #     np.asarray(fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR
    # )
    if frame.shape[1::-1] != frame_shape:
        print(f"New frame shape: {frame.shape[::-1]} | Init frame shape: {frame_shape}")
    video.write(frame)
video.release()

# %%
# Combine features (video) and audio into single video file
out_filename = f_input.parent / f"{f_input.stem}_out.mp4"
os.system(
    f"ffmpeg -y -i {f_features} -i {input_audiofile}"
    f" -c:v copy -c:a aac {out_filename}"
)

print(f"Output audio/features video: {out_filename}")
# %%
print(f"Removing auxiliary file: {f_features}")
os.remove(f_features)
