"""
Use whisper-diarization to transcribe text, obtain timestamps and obtain speaker ID.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os.path
from datetime import datetime

from funasr import AutoModel
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from config import Config
from speech_transcription.emo2vec import Emo2Vec
from speech_transcription.speech2emo import Speech2Emo
from whisper_diarization.diarize import diarize_file


class EmoClassifier:
    EMO2VEC = 'Emo2Vec'
    SPEECH2EMO = 'Speech2Emo'


class Language:
    CN = 'zh'
    NL = 'nl'


class Processor:
    DUMMY_DATE = '01/01/2025'  # We use a dummy date to make computing the difference between the timestamps easier

    @classmethod
    def process_folder(cls, folder: str, file_ext='mp3', language=Language.NL):
        """

        :param folder: folder to be processed
        :param file_ext: audio file extension to look for
        :param language: language used to conduct the interviews
        :return:
        """
        if not os.path.isdir(folder):
            raise FileExistsError("The folder you specified does not appear to exist.")

        # Process files, one by one
        files = []
        # First, gather valid files, so we know how many there are
        for f in os.listdir(folder):
            if f.endswith(file_ext):
                files.append(os.path.join(folder, f))

        # Second, process files one by one
        nb_files = len(files)
        file_ok = []
        file_bad = []
        for idx_f, f in enumerate(sorted(files)):
            print(f"At file {idx_f+1}/{nb_files}: [{f}]")
            try:
                cls.process_file(f, language=language)
                file_ok.append(f)
            except Exception as e:
                print(f"Something went wrong in processing of file [{f}].\n{e}")
                file_bad.append(f)

        if file_bad:
            print("Could not process the following files:")
            for f in file_bad:
                print(f)

    @classmethod
    def process_file(cls, file: str, emo_classifier=EmoClassifier.SPEECH2EMO,
                     language=Language.NL):
        """

        :param file: path to audio file to be processed
        :return:
        """
        srt_file = os.path.splitext(file)[0] + '.srt'
        # First, diarize file, if this has not been done already.
        # This will generate two new files with the same base filename, but with extensions '.srt' and '.txt'.
        # The '.srt' file contains the transcriptions, sentence by sentence, with timestamps and speaker ID.
        if not os.path.isfile(srt_file):
            diarize_file(file=file, language=language)

        # Load corresponding '.srt' file.
        segments = cls.read_srt_file(srt_file=srt_file)

        # Process each segment with speech emo detection network
        print("Loading speech emotion recognition model...")
        if emo_classifier == EmoClassifier.SPEECH2EMO:
            model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
            model = AutoModelForAudioClassification.from_pretrained(model_id)

            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
            id2label = model.config.id2label
        elif emo_classifier == EmoClassifier.EMO2VEC:
            model_id = "iic/emotion2vec_plus_large"
            model = AutoModel(
                model=model_id,
                hub="hf",
                # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
                disable_update=True
            )
        else:
            raise ValueError(f"Invalid option for emo_classifier: {emo_classifier}")

        print("Processing segments...")
        out_text = ''
        for idx_s, s in enumerate(segments):
            start_time = datetime.strptime(f'{cls.DUMMY_DATE} {s['start']}', '%d/%m/%Y %H:%M:%S,%f')
            end_time = datetime.strptime(f'{cls.DUMMY_DATE} {s['end']}', '%d/%m/%Y %H:%M:%S,%f')
            duration = end_time - start_time
            duration_s = duration.seconds + float(f'0.{duration.microseconds}')
            if duration_s > 60:  # Cap maximum segment length at 60s
                duration_s = 60
            start_time_s = 60*start_time.minute + start_time.second + float(f'0.{start_time.microsecond}')

            if emo_classifier == EmoClassifier.SPEECH2EMO:
                seg_emo = Speech2Emo.predict_emotion(
                    audio_path=file,
                    start_time=start_time_s,
                    duration=duration_s,
                    model=model,
                    feature_extractor=feature_extractor,
                    id2label=id2label
                )
            elif emo_classifier == EmoClassifier.EMO2VEC:
                seg_emo = Emo2Vec.process_file(audio_path=file,
                                               start_time=start_time_s,
                                               duration=duration_s,
                                               model=model)
            else:
                raise ValueError(f"Invalid option for emo_classifier: {emo_classifier}")

            print(f'start_time: {start_time}, duration: {duration_s}, ins: {start_time_s}')
            print(f'"{s['text']}"')
            print(seg_emo)
            print()

            if idx_s > 0:
                out_text += '\n'
            out_text += f'Speaker {s['speaker']}\n'
            out_text += f'Timeframe: {s['start']} -- {s['end']}\n'
            out_text += f'Text: {s['text']}\n'
            out_text += f'Emotion: {seg_emo}\n'

        # Write output to file
        out_file = os.path.splitext(file)[0] + '_emo.txt'
        with open(out_file, 'w') as fout:
            fout.write(out_text)
        print("Done!")
        print(f"Output written to: {out_file}")

    @classmethod
    def read_srt_file(cls, srt_file):
        """
        Load an srt file into memory.
        Return format is a list containing a sequence of speech segments, each segment corresponding to a continuous
        segment spoken by a same speaker, and contains start and end times.

        :param srt_file:
        :return:
        """
        # Example of a segment in file
        # --------------------------------------------------
        # 1
        # 00:00:00,160 --> 00:00:01,080
        # Speaker 0: Mijn mentor.
        #
        # 2
        # ...
        # --------------------------------------------------
        segments = []
        prev_speaker, speaker, start_time, end_time, prev_end_time = 0, -1, '', '', ''
        seg_start, seg_end, seg_text = '', '', ''
        b_first = True
        b_new = False
        with open(srt_file, 'r') as fin:
            for l in fin:
                l = l.strip()
                nb_parts = len(l.split(' '))
                if nb_parts == 1:  # start of new sentence
                    b_new = True
                elif b_new:  # Next line contains timestamps
                    b_new = False
                    parts = l.split(' --> ')
                    start_time = parts[0].strip()
                    end_time = parts[1].strip()
                elif l.startswith('Speaker '):  # Extract speaker ID
                    parts = l.split(' ')
                    speaker = int(parts[1][:-1])  # Cut off ':' at the end --> [:-1]
                    text = l.split(':', maxsplit=1)[1].strip()

                    # Speaker is same speaker as previous line?
                    if speaker == prev_speaker:
                        seg_text += f' {text}'

                    # Nopes? Then start of new segment!
                    else:
                        # First, add segment to list
                        if b_first:
                            b_first = False
                        else:
                            seg_end = prev_end_time
                            segments.append({'speaker': 1 if speaker else 0, 'text': seg_text, 'start': seg_start, 'end': seg_end})

                        # Initialize new segment
                        seg_text = text
                        seg_start = start_time

                    prev_speaker = speaker
                    prev_end_time = end_time
            # Don't forget to add final segment!
            seg_end = prev_end_time
            segments.append({'speaker': 1 if speaker else 0, 'text': seg_text, 'start': seg_start, 'end': seg_end})

        return segments

if __name__ == '__main__':
    # _file = Config.FILE_DEMO
    _file_nl = os.path.join(Config.DIR_DATA, 'Dutch Recordings', '16_240619_part 1.mp3')
    _file_cn = os.path.join(Config.DIR_DATA, 'China Recordings', 'CN05_250330_part 1.mp3')

    # Language choices:
    # Language.CN for Chinese
    # Language.NL for Dutch

    # Process a single file
    if True:
        Processor.process_file(file=_file_nl,
                               emo_classifier=EmoClassifier.EMO2VEC,
                               language=Language.NL)

    # Process all mp3 files in a given folder
    if False:
        Processor.process_folder(folder=os.path.join(Config.DIR_DATA, 'Dutch Recordings'),
                                 file_ext='mp3',
                                 language=Language.NL)
