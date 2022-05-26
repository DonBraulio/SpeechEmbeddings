# %%
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import torchaudio
from pyannote.audio import Pipeline, Audio
from IPython.display import Audio as IPythonAudio


# %%

# %%
file_path = "Datasets/samples/clase_1_1_20s.wav"
# file_path = "Datasets/samples/clase_2_1_30s.wav"
# file_path = "Datasets/samples/clase_3_1_20s.wav"


# %%
# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
pipeline = SpeakerDiarization(
    segmentation="pyannote/segmentation",
    embedding="speechbrain/spkrec-ecapa-voxceleb",
    clustering="AgglomerativeClustering",
)

pipeline.instantiate(
    dict(
        clustering={"method": "average", "threshold": 0.582398766878762},
        min_activity=6.073193238899291,
        min_duration_off=0.09791355693027545,
        min_duration_on=0.05537587440407595,
        offset=0.4806866463041527,
        onset=0.8104268538848918,
        stitch_threshold=0.04033955907446252,
    )
)

# %%
diarization = pipeline(file_path)
diarization

# %%
# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# %%
# load audio waveform, crop excerpt, and play it
# waveform, sr = torchaudio.load(file_path)
# IPythonAudio(waveform.flatten().numpy(), rate=sr, autoplay=True)
