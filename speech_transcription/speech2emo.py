"""
The following code is taken from https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3
and slightly adapted to suit my needs.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import librosa
import torch
import numpy as np
from lortools.sort.sort_tools import SortTools


class Speech2Emo:
    @classmethod
    def preprocess_audio(cls, audio_path,
                         start_time,
                         duration,
                         feature_extractor, max_duration=30.0):
        audio_array, sampling_rate = librosa.load(audio_path,
                                                  offset=start_time,
                                                  duration=duration,
                                                  sr=None)

        max_length = int(feature_extractor.sampling_rate * max_duration)
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        else:
            audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

        inputs = feature_extractor(
            audio_array,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    @classmethod
    def predict_emotion(cls, audio_path,
                        start_time, duration,
                        model, feature_extractor, labels, max_duration=30.0):
        inputs = cls.preprocess_audio(audio_path=audio_path,
                                      start_time=start_time, duration=duration,
                                      feature_extractor=feature_extractor,
                                      max_duration=max_duration)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits[0], dim=0).cpu().numpy()
        scores, emos = SortTools.sort_together(probs, labels, b_desc=True)

        return emos, scores
