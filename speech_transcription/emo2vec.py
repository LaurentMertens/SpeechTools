"""
A wrapper to use Emotion2Vec for Speech Emotion Prediction.
See https://github.com/ddlBoJack/emotion2vec?tab=readme-ov-file#inference-with-checkpoints

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import librosa
import numpy as np


class Emo2Vec:
    """
    Following is taken from Emo2Vec github repo:
    Using the finetuned emotion recognization model

    rec_result contains {'feats', 'labels', 'scores'}
    	extract_embedding=False: 9-class emotions with scores
    	extract_embedding=True: 9-class emotions with scores, along with features

    9-class emotions:
    iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
    iic/emotion2vec_base_finetuned (Jan. 2024 release)
        0: angry
        1: disgusted
        2: fearful
        3: happy
        4: neutral
        5: other
        6: sad
        7: surprised
        8: unknown
    """
    @classmethod
    def process_file(cls, audio_path, start_time, duration, model):
        audio_array, sampling_rate = librosa.load(audio_path,
                                                  offset=start_time,
                                                  duration=duration,
                                                  sr=16000)
        # model="iic/emotion2vec_base"
        # model="iic/emotion2vec_base_finetuned"
        # model="iic/emotion2vec_plus_seed"
        # model="iic/emotion2vec_plus_base"
        # model_id = "iic/emotion2vec_plus_large"

        # model = AutoModel(
        #     model=model_id,
        #     hub="hf",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
        #     disable_update=True
        # )

        rec_result = model.generate(audio_array, output_dir="./outputs", granularity="utterance", extract_embedding=False)
        max_emo = cls.extract_max_emo(rec_result=rec_result[0])

        return max_emo

    @classmethod
    def extract_max_emo(cls, rec_result):
        scores = rec_result['scores']
        idx_max = np.argmax(np.asarray(scores))
        max_emo = rec_result['labels'][idx_max]
        if idx_max < 8:
            max_emo = max_emo.split('/')[1]


        return max_emo

