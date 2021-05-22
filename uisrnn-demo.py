# %%
import sys
import uisrnn
import numpy as np

# from matplotlib import animation, rc
# from IPython.display import HTML
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.animation import FuncAnimation
# from matplotlib import cm
# from time import sleep, perf_counter as timer
# from umap import UMAP
# import matplotlib.pyplot as plt

sys.path.append("Resemblyzer")
from resemblyzer import preprocess_wav, VoiceEncoder, sampling_rate  # noqa

# %%
# Load file
wav = preprocess_wav("Resemblyzer/audio_data/X2zqiX6yL3I.mp3")

# %%
# Audio features
encoder = VoiceEncoder("cpu")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=5)

# %%
# Load UIS-RNN model
sys.argv = ['dummy']
model_args, training_args, inference_args = uisrnn.parse_arguments()
model = uisrnn.UISRNN(model_args)
model.load('uis-rnn/saved_model.uisrnn')

# %%
# Testing
test_sequence = cont_embeds.astype(float)
predictions = model.predict(test_sequence, inference_args)

# %%
