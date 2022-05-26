# %%[markdown]
"""
# Speaker Diarization using SpeechBrain (via HuggingFace)
The aim of this notebook is to perform diarization using speaker
embeddings based on:
https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
"""
# %%
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.processing import diarization
from speechbrain.processing.PLDA_LDA import StatObject_SB

# %%
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs = torchaudio.load(
    "repo_tesis/speechbrain/samples/audio_samples/example1.wav"
)
embeddings = classifier.encode_batch(signal)

# %%
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)
# Result means: do these wavs belong to the same speaker?
score, prediction = verification.verify_files(
    "speechbrain/spkrec-ecapa-voxceleb/example1.wav",
    "speechbrain/spkrec-ecapa-voxceleb/example2.flac",
)

# %%
# TODO: create diary_obj
diarization.do_spec_clustering(
    diary_obj,
    "output.rttm",
    rec_id="rec0",
    k=None,
    pval=0.3,
    affinity_type="cos",
    n_neighbors=0,
)
