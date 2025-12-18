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
    def process_folder(cls, folder: str, file_ext='mp3',
                       b_print_all_emos=True,
                       b_print_emo_probs=True,
                       window_size=60,
                       window_stride=20,
                       emo_classifier=EmoClassifier.SPEECH2EMO,
                       language=Language.NL):
        """

        :param folder: folder to be processed
        :param file_ext: audio file extension to look for
        :param b_print_all_emos: print all emotions (True) or just top predicted emotion (False)? Emotions are printed in descending order of score; emotion with highest assigned probability first
        :param b_print_emo_probs: print probabilities for each emotion label?
        :param window_size: size in seconds of the sliding window, i.e., size of the segment patches to processed
        :param window_stride: size in seconds of the step by which to move the sliding window
        :param emo_classifier: which emotion classifier to use
        :param language: target language, i.e., language spoken in the interviews
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
                cls.process_file(file=f,
                                 b_print_emo_probs=b_print_emo_probs,
                                 b_print_all_emos=b_print_all_emos,
                                 window_size=window_size,
                                 window_stride=window_stride,
                                 emo_classifier=emo_classifier,
                                 language=language)
                file_ok.append(f)
            except Exception as e:
                print(f"Something went wrong in processing of file [{f}].\n{e}")
                file_bad.append(f)

        if file_bad:
            print("Could not process the following files:")
            for f in file_bad:
                print(f)

    @classmethod
    def process_file(cls, file: str,
                     b_print_all_emos=True,
                     b_print_emo_probs=True,
                     window_size=60,
                     window_stride=20,
                     emo_classifier=EmoClassifier.SPEECH2EMO,
                     language=Language.NL):
        """

        :param file: path to audio file to be processed
        :param b_print_all_emos: print all emotions (True) or just top predicted emotion (False)? Emotions are printed in descending order of score; emotion with highest assigned probability first
        :param b_print_emo_probs: print probabilities for each emotion label?
        :param window_size: size in seconds of the sliding window, i.e., size of the segment patches to processed
        :param window_stride: size in seconds of the step by which to move the sliding window
        :param emo_classifier: which emotion classifier to use
        :param language: target language, i.e., language spoken in the interviews
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
            labels = [model.config.id2label[x] for x in range(len(model.config.id2label))]
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
            print(f"Segment {idx_s}...")
            print(f'"{s['text']}"')
            start_time = datetime.strptime(f'{cls.DUMMY_DATE} {s['start']}', '%d/%m/%Y %H:%M:%S,%f')
            end_time = datetime.strptime(f'{cls.DUMMY_DATE} {s['end']}', '%d/%m/%Y %H:%M:%S,%f')
            # duration = end_time - start_time
            # duration_s = duration.seconds + float(f'0.{duration.microseconds}')
            # # if duration_s > 60:  # Cap maximum segment length at 60s
            # #     duration_s = 60

            start_time_s = 60 * start_time.minute + start_time.second + float(f'0.{start_time.microsecond:06d}')
            end_time_s = 60 * end_time.minute + end_time.second + float(f'0.{end_time.microsecond:06d}')

            if idx_s > 0:
                out_text += '\n'
            out_text += '='*90 + '\n'
            out_text += f'Segment  : {idx_s}\n'
            out_text += f'Speaker  : {s['speaker']}\n'
            out_text += f'Timeframe: {s['start']} -- {s['end']}\n'
            out_text += f'Text     : {s['text']}\n'

            at_patch = -1
            while True:
                at_patch += 1

                patch_start_time = start_time_s + (at_patch * window_stride)
                # Break out of loop if the start time of this patch is less than one fifth of the window size distance
                # from the end of the segment
                # Does not apply to the first patch
                if at_patch > 0 and patch_start_time - end_time_s > -window_size//5:
                    break

                patch_end_time = patch_start_time + window_size
                if patch_end_time > end_time_s:
                    patch_end_time = end_time_s
                duration_s = patch_end_time - patch_start_time

                print(f"Segment patch {at_patch}")
                print(f'start_time: {patch_start_time}, duration: {duration_s}, ins: {patch_start_time}')

                out_text += '-' * 90 + '\n'
                out_text += f'Segment patch : {at_patch}\n'
                out_text += f'Patch start   : {int(patch_start_time//60)}m {patch_start_time%60:.2f}s\n'
                out_text += f'Patch duration: {int(duration_s//60)}m {duration_s%60:.2f}s\n'
                if emo_classifier == EmoClassifier.SPEECH2EMO:
                    seg_emos, seg_scores = Speech2Emo.predict_emotion(
                        audio_path=file,
                        start_time=patch_start_time,
                        duration=duration_s,
                        model=model,
                        feature_extractor=feature_extractor,
                        labels=labels
                    )
                elif emo_classifier == EmoClassifier.EMO2VEC:
                    seg_emos, seg_scores = Emo2Vec.process_file(audio_path=file,
                                                                start_time=start_time_s,
                                                                duration=duration_s,
                                                                model=model)
                else:
                    raise ValueError(f"Invalid option for emo_classifier: {emo_classifier}")

                out_text += "Emotions     : "
                if b_print_all_emos:
                    for idx_emo, _emo in enumerate(seg_emos):
                        if idx_emo > 0:
                            print(f" -- ", end='')
                            out_text += f" -- "
                        print(f"{_emo}", end='')
                        out_text += f"{_emo}"
                        if b_print_emo_probs:
                            print(f" [{100*seg_scores[idx_emo]:.1f}%]", end='')
                            out_text += f" [{100*seg_scores[idx_emo]:.1f}%]"
                out_text += '\n'
                print('\n\n')

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
                               b_print_all_emos=True,
                               b_print_emo_probs=True,
                               window_size=60,
                               window_stride=20,
                               emo_classifier=EmoClassifier.EMO2VEC,
                               language=Language.NL)

    # Process all mp3 files in a given folder
    if False:
        Processor.process_folder(folder=os.path.join(Config.DIR_DATA, 'Dutch Recordings'),
                                 file_ext='mp3',
                                 b_print_all_emos=True,
                                 b_print_emo_probs=True,
                                 window_size=60,
                                 window_stride=20,
                                 emo_classifier=EmoClassifier.EMO2VEC,
                                 language=Language.NL)
