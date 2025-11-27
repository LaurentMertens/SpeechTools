"""
Trying different things...

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os

from config import Config

if __name__ == '__main__':
    # Smallest file in the corpus
    demo_file = os.path.join(Config.DIR_DATA, 'DEMO_SAMPLE.wav')

    # Speaker separation
    if False:
        from speechbrain.inference.separation import SepformerSeparation as separator

        model = separator.from_hparams(source="speechbrain/sepformer-libri2mix",
                                       savedir='pretrained_models/sepformer-libri2mix',
                                       run_opts={"device": "cuda"}
                                       )

        # for custom file, change path
        est_sources = model.separate_file(path=demo_file)
        # print(est_sources)

        torchaudio.save(os.path.join(Config.DIR_DATA, "source1hat.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(Config.DIR_DATA, "source2hat.wav"), est_sources[:, :, 1].detach().cpu(), 8000)

    # Speech Voice Activity Detection
    # https://huggingface.co/speechbrain/vad-crdnn-libriparty
    if False:
        from speechbrain.inference.VAD import VAD

        VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty",
                               savedir="pretrained_models/vad-crdnn-libriparty",
                               run_opts={"device": "cuda"},
                               overrides={"sample_rate": 8000}
                               )
        boundaries = VAD.get_speech_segments(os.path.join(Config.DIR_DATA, "source2hat.wav"))

        # Print the output
        VAD.save_boundaries(boundaries)

    # Emo classification
    if False:
        from speechbrain.inference.interfaces import foreign_class

        classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                                   pymodule_file="custom_interface.py",
                                   classname="CustomEncoderWav2vec2Classifier")

        files = os.listdir(Config.DIR_DATA)

        out_prob, score, index, text_lab =\
            classifier.classify_file(demo_file)
        print(text_lab)

    # Emotion Diarization
    if False:
        from speechbrain.inference.diarization import Speech_Emotion_Diarization

        demo_file = os.path.join(Config.DIR_DATA, 'DEMO_SAMPLE_2_44100_MONO.wav')
        classifier = Speech_Emotion_Diarization.from_hparams(
            source="speechbrain/emotion-diarization-wavlm-large",
            run_opts={"device": "cuda"},
        )
        diary = classifier.diarize_file(demo_file)
        print(diary)
