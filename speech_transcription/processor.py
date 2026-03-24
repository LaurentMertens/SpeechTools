"""
Use whisper-diarization to transcribe text, obtain timestamps and obtain speaker ID.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os.path
from collections import Counter
from datetime import datetime

import torch
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
            if not f.startswith('._') and f.endswith(file_ext):
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
        :param b_write_output: write output to disk
        :return:
        """
        # First, check if this file has already been processed with these specific window size/stride settings.
        out_file = os.path.splitext(file)[0] + f'_winSize={window_size}_winStride={window_stride}_proc={emo_classifier}.res'
        if os.path.isfile(out_file):
            print(f"File {file} has already been processed with these window size/stride settings.")
            print("Loading previously saved results...")
            emotions_per_segment = cls.read_res_file(out_file)
        else:
            emotions_per_segment = cls._process_file(file=file, b_print_all_emos=b_print_all_emos,
                                                     b_print_emo_probs=b_print_emo_probs, window_size=window_size,
                                                     window_stride=window_stride, emo_classifier=emo_classifier,
                                                     language=language, out_file=out_file)

        return emotions_per_segment

    @classmethod
    def _process_file(cls, file: str,
                      b_print_all_emos=True,
                      b_print_emo_probs=True,
                      window_size=60,
                      window_stride=20,
                      emo_classifier=EmoClassifier.SPEECH2EMO,
                      language=Language.NL,
                      out_file=None):
        """
        This method does the processing proper. It should only be called by process_file if the file to be processed
        has not already been processed before (and the output saved to disk).

        For parameter doc, check process_file().
        Only extra parameter is out_file, which is the path to the output file to be generated.
        """
        if out_file is None:
            out_file = os.path.splitext(file)[0] + f'_winSize={window_size}_winStride={window_stride}.res'

        # Define path to srt file that contains/will contain the automatic speech-to-text transcription
        srt_file = os.path.splitext(file)[0] + '.srt'
        # First, diarize audio file, if this has not been done already.
        # This will generate two new files with the same base filename, but with extensions '.srt' and '.txt'.
        # The '.txt' file contains only the text, separated into paragraphs with corresponding speaker ID.
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
        emotions_per_segment = []
        for idx_s, s in enumerate(segments):
            print(f"Segment {idx_s}...")
            print(f'"{s['text']}"')
            start_time = datetime.strptime(f'{cls.DUMMY_DATE} {s['start']}', '%d/%m/%Y %H:%M:%S,%f')
            end_time = datetime.strptime(f'{cls.DUMMY_DATE} {s['end']}', '%d/%m/%Y %H:%M:%S,%f')
            # duration = end_time - start_time
            # duration_s = duration.seconds + float(f'0.{duration.microseconds}')
            # # if duration_s > 60:  # Cap maximum segment length at 60s
            # #     duration_s = 60

            start_time_s = 3600 * start_time.hour + 60 * start_time.minute + start_time.second + float(f'0.{start_time.microsecond:06d}')
            end_time_s = 3600 * start_time.hour + 60 * end_time.minute + end_time.second + float(f'0.{end_time.microsecond:06d}')

            if idx_s > 0:
                out_text += '\n'
            out_text += '='*90 + '\n'
            out_text += f'Segment  : {idx_s}\n'
            out_text += f'Speaker  : {s['speaker']}\n'
            out_text += f'Timeframe: {s['start']} -- {s['end']}\n'
            out_text += f'Text     : {s['text']}\n'

            at_patch = -1
            emotion_per_patch = []
            while True:
                at_patch += 1

                patch_start_time = start_time_s + (at_patch * window_stride)
                # Break out of loop if the remaining audio size is less than window size
                # Does not apply to the first patch
                if at_patch > 0 and end_time_s - patch_start_time < window_size:
                    break

                patch_end_time = patch_start_time + window_size
                # The clause below shouldn't be needed anymore, but was needed when allowing processing patches with
                # remaining audio size < window_size
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

                out_text += "Emotions      : "
                for idx_emo, _emo in enumerate(seg_emos):
                    if idx_emo > 0:
                        if b_print_all_emos:
                            print(f" -- ", end='')
                        out_text += f" -- "
                    if b_print_all_emos:
                        print(f"{_emo}", end='')
                    out_text += f"{_emo}"
                    if b_print_emo_probs:
                        print(f" [{100*seg_scores[idx_emo]:.1f}%]", end='')
                    out_text += f" [{100*seg_scores[idx_emo]:.1f}%]"
                out_text += '\n'
                print('\n')

                # Fill list of emotion per patch
                emotion_per_patch.append((patch_start_time, patch_end_time, seg_emos, seg_scores))
            emotions_per_segment.append((start_time_s, end_time_s, emotion_per_patch))
        print("Done!")

        print("Clearing some VRAM...")
        del model
        torch.cuda.empty_cache()

        # Write output to file
        with open(out_file, 'w') as fout:
            fout.write(out_text)
        print(f"Output written to: {out_file}")

        return emotions_per_segment

    @classmethod
    def read_res_file(cls, res_file):
        """
        Load a res file into memory.
        Return format is the same as the output of _process_file().

        :param res_file:
        :return:
        """
        # Example of a segment in file
        # ==========================================================================================
        # Segment  : 28
        # Speaker  : 0
        # Timeframe: 00:05:00,800 -- 00:05:37,980
        # Text     : Jaren daarvoor was mij dat opgevallen. Want we zijn toen zelfs bij de oorarts geweest ook. En waarom? Als hij de telefoon opnam, dan wist hij nooit van de eerste keer met wie hij in gesprek moest gaan. En dan gebruikte hij zijn stopwoord. Ja, hoe is het met jou? Alles goed? En dan moesten de mensen drie, vier keren hun naam zeggen. Ik dacht dat dat met zijn oren, met zijn gehoor te maken had. Maar blijkt dat dat daar niet mee te maken had.
        # ------------------------------------------------------------------------------------------
        # Segment patch : 0
        # Patch start   : 5m 0.80s
        # Patch duration: 0m 37.18s
        # Emotions     : sad [53.4%] -- neutral [22.9%] -- disgusted [15.2%] -- fearful [2.4%] -- happy [2.1%] -- <unk> [2.0%] -- other [1.6%] -- surprised [0.3%] -- angry [0.1%]
        # ------------------------------------------------------------------------------------------
        # Segment patch : 1
        # Patch start   : 5m 20.80s
        # Patch duration: 0m 17.18s
        # Emotions     : neutral [98.2%] -- sad [0.9%] -- happy [0.3%] -- <unk> [0.2%] -- disgusted [0.2%] -- fearful [0.1%] -- other [0.1%] -- surprised [0.0%] -- angry [0.0%]
        # --empty line in file followed by 'equals sign line' to indicate start of next segment--

        #  Output format
        # emotions_per_segment[(start_time_seg, end_time_seg, emotion_per_patch)]
        # with
        # emotion_per_path[((patch_start_time, patch_end_time, seg_emos, seg_scores))]
        emotions_per_segment = []

        _res_seg = {}  # Container to contain results per segment
        _res_patch = {}  # Container to contain results per patch
        with open(res_file, 'r') as fin:
            for l in fin:
                l = l.strip()
                # SEGMENT START
                # Start of new segment
                if l == '='*90:
                    _res_seg = {'emotion_per_patch': []}
                # Segment  : 28
                elif l.startswith('Segment  :'):
                    _res_seg['seg_id'] = int(l.split(":")[1].strip())
                # Speaker  : 0
                elif l.startswith('Speaker  :'):
                    _res_seg['spkr_id'] = int(l.split(":")[1].strip())
                # Timeframe: 00:05:00,800 -- 00:05:37,980
                elif l.startswith('Timeframe:'):
                    times = l.split(":", maxsplit=1)[1].strip().split(' -- ')
                    _res_seg['start_time'] = times[0]
                    _res_seg['end_time'] = times[1]
                # Text     : Here be text.
                elif l.startswith('Text     :'):
                    _res_seg['text'] = l.split(":", maxsplit=1)[1].strip()

                # PATCH START
                # Start of new patch
                elif l == '-'*90:
                    _res_patch = {}
                # Segment patch : 0
                elif l.startswith('Segment patch :'):
                    _res_patch['patch_id'] = int(l.split(":")[1].strip())
                # Patch start   : 5m 0.80s
                elif l.startswith('Patch start   :'):
                    _res_patch['start_time'] = l.split(":")[1].strip()
                # Patch duration: 0m 37.18s
                elif l.startswith('Patch duration:'):
                    _res_patch['duration'] = l.split(":")[1].strip()
                # Emotions     : sad [53.4%] -- neutral [22.9%] -- disgusted [15.2%] -- fearful [2.4%] -- happy [2.1%] -- <unk> [2.0%] -- other [1.6%] -- surprised [0.3%] -- angry [0.1%]
                elif l.startswith('Emotions      :'):
                    _res_patch['emotions'] = cls._extract_emotions(l.split(":")[1].strip())
                    _res_seg['emotion_per_patch'].append(_res_patch)
                # line is empty; this is the end of a segment
                elif not l:
                    emotions_per_segment.append(_res_seg)
                else:
                    raise ValueError(f"Encountered a line that I don't know what to do with...\n[{l}]")

        # Don't forget to add the last segment!
        emotions_per_segment.append(_res_seg)

        return emotions_per_segment

    @classmethod
    def guess_interviewer(cls, emotions_per_segment: list):
        """
        Given a dictionary containing the results of an audio analysis of an interview file, guess which speaker is the
        interviewer and which is the interviewee.
        The approach is to assign the "interviewee" label to the speaker who talks the most.

        :param emotions_per_segment: result of either _process_file() or read_res_file()
        :return:
        """
        length_per_speaker = Counter()
        for seg in emotions_per_segment:
            t_start = datetime.strptime(seg['start_time'], '%H:%M:%S,%f')
            t_end = datetime.strptime(seg['end_time'], '%H:%M:%S,%f')
            t_duration = t_end - t_start
            length_per_speaker[seg['spkr_id']] += t_duration.total_seconds()

        # Person who speaks te least is assumed to be the interviewer
        if length_per_speaker[0] < length_per_speaker[1]:
            return 0
        else:
            return 1

    @classmethod
    def _extract_emotions(cls, line):
        # sad [53.4%] -- neutral [22.9%] -- disgusted [15.2%] -- fearful [2.4%] -- happy [2.1%] -- <unk> [2.0%] -- other [1.6%] -- surprised [0.3%] -- angry [0.1%]
        parts = line.split(' -- ')
        emotions = []
        for p in parts:
            _emo = p.split(' [')[0].strip()
            _prob = float(p.split('[')[1].split('%]')[0])
            emotions.append((_emo, _prob))

        return emotions

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
    # _file_nl = os.path.join(Config.DIR_DATA, 'Dutch Recordings', '16_240619_part 1.mp3')
    _file_nl = os.path.join(Config.DIR_DATA, 'Dutch Recordings', '13_240429_part 1.mp3')
    _file_cn = os.path.join(Config.DIR_DATA, 'China Recordings', 'CN05_250330_part 1.mp3')

    # Language choices:
    # Language.CN for Chinese
    # Language.NL for Dutch

    # Process a single file
    if False:
        Processor.process_file(file=_file_nl,
                               b_print_all_emos=True,
                               b_print_emo_probs=True,
                               window_size=30,
                               window_stride=15,
                               emo_classifier=EmoClassifier.SPEECH2EMO,
                               language=Language.NL)

    # Process all mp3 files in a given folder
    if True:
        Processor.process_folder(folder=os.path.join(Config.DIR_DATA, 'Dutch Recordings'),
                                 file_ext='mp3',
                                 b_print_all_emos=True,
                                 b_print_emo_probs=True,
                                 window_size=30,
                                 window_stride=15,
                                 emo_classifier=EmoClassifier.SPEECH2EMO,
                                 language=Language.NL)
