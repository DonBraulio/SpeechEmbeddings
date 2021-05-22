# %%
import os
import sys
import cv2
import pysptk
import numpy as np
import librosa
import matplotlib.pyplot as plt
import webrtcvad as wrtcvad

# %%
# Extract WAV audio fragment from any input file type
input_filename = "PC_1107003_5A_7112019_REC-2019-11-07T14_15_00Z.mp4"
input_basename, input_extension = input_filename.split(".")
input_audiofile = f"{input_basename}.wav"
start_time = 10
max_duration = 20
ffmpeg_cmd = (
    f"ffmpeg -y -i {input_filename} -ss {start_time} -t {max_duration}"
    f" -c:a pcm_s16le -ar 16000 {input_audiofile}"
)

print(ffmpeg_cmd)
os.system(ffmpeg_cmd)

# %%
# Plot waveform
progress_lines = []
y, fs = librosa.load(input_audiofile)
n_samples = len(y)
duration = n_samples / fs
t = np.linspace(0, duration, num=n_samples)
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(15, 10))
ax1.plot(t, y)
ax1.set_ylabel("Waveform")
progress_lines.append(ax1.axvline(x=0, color="r"))

# Plot fundamental frequency
f0 = pysptk.swipe(y.astype(np.float64), fs=fs, hopsize=80, min=30, max=200, otype="f0")
f0_t = np.linspace(0, duration, num=len(f0))
ax2.plot(f0_t, f0, "g")
ax2.set_ylabel("Fundamental freq (f0)")
ax2.set_xlabel("time (s)")
progress_lines.append(ax2.axvline(x=0, color="r", linestyle="--"))

# vad = wrtcvad.Vad(3)  # 3 is most aggressive filtering
# is_voice = []

# %%
# Generate animated video from plots above
out_features = f"features_{input_basename}.mp4"
try:
    os.remove(out_features)
except FileNotFoundError:
    pass
fps = 10
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

fig.canvas.draw()
frame_shape = fig.canvas.renderer.buffer_rgba().shape[1::-1]
print(f"Frame shape: {frame_shape}")
n_frames = int(duration * fps + 0.5)
video = cv2.VideoWriter(out_features, fourcc, fps, frameSize=frame_shape)
for i in range(n_frames):
    current_pos = duration * i / n_frames
    for lp in progress_lines:
        lp.set_xdata(current_pos)
    fig.canvas.draw()
    frame = cv2.cvtColor(
        np.asarray(fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR
    )
    if frame.shape[1::-1] != frame_shape:
        print(f"New frame shape: {frame.shape[::-1]} | Init frame shape: {frame_shape}")
    video.write(frame)
video.release()

# %%
# Combine features (video) and audio into single video file
out_filename = f"out_{input_basename}.mp4"
os.system(
    f"ffmpeg -y -i {out_features} -i {input_audiofile}"
    f" -c:v copy -c:a aac {out_filename}"
)

# %%
