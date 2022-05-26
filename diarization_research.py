# %%[markdown]
"""
# Speaker embeddings

Notebook to test Embedding similarity, KNN, speaker change detection

Speaker embeddings based on:
https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
"""
# %%
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from pyannote.audio import Inference

# %%
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
# %%
audio_files = [
    "Datasets/samples/clase_1_1_20s.wav",
    "Datasets/samples/clase_2_1_30s.wav",
    "Datasets/samples/clase_3_1_20s.wav",
]

signals = []
for audio_file in audio_files:
    signal, fs = torchaudio.load(audio_file)
    assert fs == 16000, f"{audio_file=} , {fs=}"
    signals.append(signal)

# %%
# Create a single vector 1xT
signals = torch.concat(signals, dim=1)

# TODO: create batches & audio normalizer
# ECAPA-TDNN: 25ms windows, 10ms frame shift
# %%
from pyannote.audio import Inference
inference = Inference("pyannote/embedding", 
                      window="sliding",
                      duration=3.0, step=1.0)
# %%
%time embeddings = classifier.encode_batch(signals, normalize=True)

# %%
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
