# %%
# Example 1: short-term feature extraction
import matplotlib.pyplot as plt
import plotly.graph_objs as go 
import numpy as np 
import plotly
import IPython

from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO 

# %%
# read audio data from file 
# (returns sampling freq and signal as a numpy array)
audio_file = "pyAudioAnalysis/pyAudioAnalysis/data/speech_music_sample.wav"
fs, s = aIO.read_audio_file(audio_file)

# play the initial and the generated files in notebook:
IPython.display.display(IPython.display.Audio(audio_file))

# print duration in seconds:
duration = len(s) / float(fs)
print(f'duration = {duration} seconds')
win, step = 0.050, 0.050
time = np.arange(0, duration - step, win)
# %%
plt.figure()
plt.plot(s, '-')
# %%
# get the feature whose name is 'energy'
energy = fn[fn.index('energy'), :]
mylayout = go.Layout(yaxis=dict(title="frame energy value"),
                     xaxis=dict(title="time (sec)"))
plotly.offline.iplot(go.Figure(data=[go.Scatter(x=time, 
                                                y=energy)], 
                               layout=mylayout))
# %%

# extract short-term features using a 50msec non-overlapping windows
[f, fn] = aF.feature_extraction(s, fs, int(fs * win), 
                                int(fs * step))
print(f'{f.shape[1]} frames, {f.shape[0]} short-term features')
print('Feature names:')
for i, nam in enumerate(fn):
    print(f'{i}:{nam}')


# %%
# Plot features over the audio signal
for feat_name in ['zcr', 'energy', 'spectral_centroid']:
    # Normalize feature for visualization
    y_feat = f[fn.index(feat_name), :]
    y = y_feat / np.max(y_feat)

    # Plot
    plt.figure()
    plt.title(f"Feature: {feat_name} | shape: {y_feat.shape}")
    plt.plot(s / np.max(s), 'b-')
    plt.plot(np.linspace(0, len(s), len(y)), y, 'r-')
# %%
