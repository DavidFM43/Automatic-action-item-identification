import numpy as np
import pandas as pd
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import \
    PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from utils import convert_time

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CONTEXT = 1000
OFFSET = 4
MAX_TEXT_LENGTH = 500
MAX_SPEAKERS = 15
EMBEDDING_SIZE = 192
whisper_model = "small.en"
model = WhisperModel(
    whisper_model, compute_type="int8", device=device
)  # compute_type is by default float16

# load speaker embedding model
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb", device=device
)


def transcribe_audio(audio_path):
    segments, info = model.transcribe(audio_path, language="en")
    # generate the transcript and parse the relevant data to a dicts
    print("Generating transcription segments.")
    segments = [
        {"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments
    ]
    return segments, info


def segment_embedding(segment, info, audio_path):
    """Generates the speaker embedding of the segment."""

    audio = Audio()
    start = segment["start"]
    end = min(segment["end"], info.duration)
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(audio_path, clip)

    return embedding_model(waveform.unsqueeze(0))


def embed_segments(segments, info, audio_path):
    # TODO: replace this code with torch tensors
    embeddings = np.zeros(shape=(len(segments), EMBEDDING_SIZE))
    for idx, segment in enumerate(segments):
        embeddings[idx] = segment_embedding(segment, info, audio_path)
    embeddings = np.nan_to_num(embeddings)

    return embeddings


def diarize_speakers(segments, info, audio_path):
    print("Diarizing speakers.")
    embeddings = embed_segments(segments, info, audio_path)
    num_speakers = find_number_speakers(embeddings)
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)
    return segments


def find_number_speakers(embeddings):
    num_speakers = 0

    if num_speakers == 0:
        # find the best number of speakers
        score_num_speakers = {}

        for num_speakers in range(2, MAX_SPEAKERS + 1):
            clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
            score = silhouette_score(embeddings, clustering.labels_, metric="euclidean")
            score_num_speakers[num_speakers] = score
        best_num_speakers = max(score_num_speakers, key=lambda x: score_num_speakers[x])
    else:
        best_num_speakers = num_speakers

    return best_num_speakers


def preprocess_segments(segments):
    # parse the time segments into speaker segments
    transcription_dict = {"start_time": [], "end_time": [], "speaker": [], "text": []}
    text = ""
    for i, segment in enumerate(segments):
        if (
            i == 0
            or segments[i - 1]["speaker"] != segment["speaker"]
            or len(text) >= MAX_TEXT_LENGTH
        ):
            transcription_dict["start_time"].append(str(convert_time(segment["start"])))
            transcription_dict["speaker"].append(segment["speaker"])
            if i != 0:
                transcription_dict["end_time"].append(
                    str(convert_time(segments[i - 1]["end"]))
                )
                transcription_dict["text"].append(text)
                text = ""
        text += segment["text"] + " "
    transcription_dict["end_time"].append(str(convert_time(segments[i - 1]["end"])))
    transcription_dict["text"].append(text)

    return pd.DataFrame(transcription_dict)


def generate_transcription(audio_path):
    segments, info = transcribe_audio(audio_path)
    segments = diarize_speakers(segments, info, audio_path)

    transcription_df = preprocess_segments(segments)

    return transcription_df
