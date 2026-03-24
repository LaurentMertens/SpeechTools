import logging
import math
import os
import re

import dill
import faster_whisper
import torch

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from whisper_diarization.helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

def diarize_file(file: str=None, language='nl', max_length=60, overlap_mins=2):
    """

    :param file: path to file
    :param language: language spoken in the audio file
    :param max_length: maximum length in MINUTES of a continuous segment to be processed. If the file is longer than
    this, it will be chopped in pieces of max_length minutes, each segment will be processed individually, and all
    pieces will be stitched back together at the end
    :param overlap_mins: number of overlapping minutes for consecutive segments (it is over this range that the program
    will look for equal content)
    :return:
    """
    mtypes = {"cpu": "int8", "cuda": "float16"}

    pid = os.getpid()
    temp_outputs_dir = f"temp_outputs_{pid}"
    temp_path = os.path.join(os.getcwd(), "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)

    # Initialize arguments
    params = {
        'file': file,  # Name of the target audio file
        'no-stem': True,  # Disables source separation; this helps with long files that don't contain a lot of music.
        'suppress_numerals': False,  # Suppresses Numerical Digits. This helps the diarization accuracy but converts all digits into written text.
        'whisper-model': 'large-v3',  # Name of the Whisper model to use
        'batch_size': 8,  # Batch size for batched inference, reduce if you run out of memory, set to 0 for original whisper longform inference
        'language': language,  # Language spoken in the audio, specify None to perform language detection
        'device': 'cuda',  # If you have a GPU use 'cuda', otherwise 'cpu'
        'diarizer': 'sortformer',  # Choose the diarization model to use
    }

    if params['file'] is None:
        raise ValueError("You did not specify a file to process!")

    language = process_language_arg(params['language'], params['whisper-model'])

    if not params['no-stem']:
        print("Applying source separation to audio file...")
        # Isolate vocals from the rest of the audio

        return_code = os.system(
            f'python -m demucs.separate -n htdemucs --two-stems=vocals "{params['file']}" -o "{temp_outputs_dir}"'
            f' --device "{params['device']}"'
        )

        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. "
                "Use --no-stem argument to disable it."
            )
            vocal_target = params['file']
        else:
            vocal_target = os.path.join(
                temp_outputs_dir,
                "htdemucs",
                os.path.splitext(os.path.basename(params['file']))[0],
                "vocals.wav",
            )
    else:
        print("Skipping source separation step...")
        vocal_target = params['file']

    # Transcribe the audio file
    print("Transcribing audio file...")
    whisper_model = faster_whisper.WhisperModel(
        params['whisper-model'], device=params['device'], compute_type=mtypes[params['device']]
    )
    print("Creating Whisper pipeline...")
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    print("Loading audio file...")
    sampling_rate = 16000
    audio_waveform = faster_whisper.decode_audio(vocal_target, sampling_rate=sampling_rate)
    print("Creating token suppressor...")
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if params['suppress_numerals']
        else [-1]
    )

    # Default sampling rate for faster_whisper.decode_audio() is 16khz; file is loaded in mono
    # E.g., 30 minutes = 16000*30*60 samples = 28800000 samples
    max_samples = max_length*60*sampling_rate
    overlap_samples = overlap_mins*60*sampling_rate
    nb_samples = len(audio_waveform)

    nb_segs = math.ceil(nb_samples/max_samples)

    processed_segments = []

    for idx_seg in range(nb_segs):
        print(f"Processing segment {idx_seg+1}/{nb_segs}...")
        if idx_seg == 0:
            start_sample = 0
        else:
            start_sample = (idx_seg * max_samples) - overlap_samples
        end_sample = (idx_seg+1) * max_samples

        if end_sample < nb_samples:
            _audio_seg = audio_waveform[start_sample:end_sample]
        else:
            _audio_seg = audio_waveform[start_sample:]

        wsm, ssm = _diarize_segment(params=params, whisper_pipeline=whisper_pipeline, whisper_model=whisper_model,
                                    audio_waveform=_audio_seg, language=language, suppress_tokens=suppress_tokens)

        # For debugging purposes: store intermediate stages
        # with open(f"{os.path.splitext(params['file'])[0]}_seg={idx_seg}.txt", "w", encoding="utf-8-sig") as f:
        #     get_speaker_aware_transcript(ssm, f)
        #
        # with open(f"{os.path.splitext(params['file'])[0]}_seg={idx_seg}.srt", "w", encoding="utf-8-sig") as srt:
        #     write_srt(ssm, srt)

        processed_segments.append((wsm, ssm))

    print("Concatenating processed segments...")
    wsm, ssm = _concatenate_processed_segments(processed_segments, ms_seg=1000*60*max_length, ms_overlap=1000*60*overlap_mins)

    print("Writing results to disc!")
    with open(f"{os.path.splitext(params['file'])[0]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{os.path.splitext(params['file'])[0]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

def _diarize_segment(params, whisper_pipeline, whisper_model, audio_waveform, language, suppress_tokens):
    print("Transcribing segments...")
    if params['batch_size'] > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=params['batch_size'],
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)

    # clear gpu vram
    print("Clearing some VRAM...")
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    # Forced Alignment
    print("Forcing alignment...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        params['device'],
        dtype=torch.float16 if params['device'] == "cuda" else torch.float32,
    )

    print("Generating emissions...")
    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device),
        batch_size=params['batch_size'],
    )

    print("Clearing some VRAM...")
    del alignment_model
    torch.cuda.empty_cache()

    print("Preprocessing text...")
    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )

    # dill.dump((emissions, tokens_starred, alignment_tokenizer), open(f'{os.path.basename(file)}.dill', 'wb'))
    # emissions, tokens_starred, alignment_tokenizer = dill.load(open(f'{os.path.basename(file)}.dill', 'rb'))

    print("Getting alignments...")
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    print("Getting spans...")
    spans = get_spans(tokens_starred, segments, blank_token)

    print("Getting timestamps...")
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    print("Diarizing...")
    if params['diarizer'] == "msdd":
        from whisper_diarization.diarization import MSDDDiarizer

        diarizer_model = MSDDDiarizer(device=params['device'])
    elif params['diarizer'] == "sortformer":
        from whisper_diarization.diarization import SortformerDiarizer

        diarizer_model = SortformerDiarizer(device=params['device'])
    else:
        raise ValueError(f"Don't know what to do with specified diarizer: [{params['diarizer']}]")

    speaker_ts = diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
    del diarizer_model
    torch.cuda.empty_cache()

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    print("Restoring punctuation...")
    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language."
            " Using the original punctuation."
        )

    print("Doing some final stuff...")
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    return wsm, ssm

def _concatenate_processed_segments(processed_segments, ms_seg, ms_overlap):
    ps = processed_segments

    nb_segs = len(processed_segments)
    # Only one segment, no need to concatenate
    if nb_segs == 1:
        return processed_segments[0]

    # Concatenate wms objects
    wsm = ps[0][0]
    for idx_seg in range(1, nb_segs):
        ms_offset = (ms_seg*idx_seg)-ms_overlap
        seg_next = ps[idx_seg][0]
        # Find index of first element in "next" segment that matches element in "previous" segment
        _res = _get_matching_wsm_index(seg_next, wsm, ms_offset=ms_offset)
        if _res is None:
            raise RuntimeError(f"Couldn't concatenate wms segments; at idx_seg={idx_seg}.")
        idx_s_next, idx_s_prev, b_same_spkr_id = _res
        for s in seg_next:
            s['start_time'] += ms_offset
            s['end_time'] += ms_offset
            # Flip speaker id if necessary
            if not b_same_spkr_id and s['speaker'] < 2:
                s['speaker'] = (s['speaker'] + 1)%2

        # Perform sanity check during code development
        # for idx_prev in range(idx_s_prev, len(wms)):
        #     assert (wms[idx_prev]['start_time'] - seg_next[idx_s_next+(idx_prev-idx_s_prev)]['start_time']) < 250
        #     assert (wms[idx_prev]['end_time'] - seg_next[idx_s_next+(idx_prev-idx_s_prev)]['end_time']) < 250
        #     assert wms[idx_prev]['word'] == seg_next[idx_s_next+(idx_prev-idx_s_prev)]['word']
        #     if wms[idx_prev]['speaker'] < 2:
        #         assert wms[idx_prev]['speaker'] == seg_next[idx_s_next+(idx_prev-idx_s_prev)]['speaker']

        wsm = wsm[:idx_s_prev] + seg_next[idx_s_next:]

    # Concatenate ssm objects
    ssm = ps[0][1]
    for idx_seg in range(1, nb_segs):
        ms_offset = (ms_seg*idx_seg)-ms_overlap
        seg_next = ps[idx_seg][1]
        # Find index of first element in "next" segment that matches element in "previous" segment
        _res = _get_matching_ssm_index(seg_next, ssm, ms_offset=ms_offset)
        if _res is None:
            raise RuntimeError(f"Couldn't concatenate ssm segments; at idx_seg={idx_seg}.")
        idx_s_next, idx_s_prev, b_same_spkr_id = _res
        for s in seg_next:
            s['start_time'] += ms_offset
            s['end_time'] += ms_offset
            # Flip speaker id if necessary
            _speaker_id = int(s['speaker'].split(' ')[1])
            if not b_same_spkr_id and _speaker_id < 2:
                s['speaker'] = f'Speaker {(_speaker_id + 1)%2}'

        # Perform sanity check
        # for idx_prev in range(idx_s_prev, len(wms)):
        #     assert (wms[idx_prev]['start_time'] - seg_next[idx_s_next+(idx_prev-idx_s_prev)]['start_time']) < 250
        #     assert (wms[idx_prev]['end_time'] - seg_next[idx_s_next+(idx_prev-idx_s_prev)]['end_time']) < 250
        #     assert wms[idx_prev]['word'] == seg_next[idx_s_next+(idx_prev-idx_s_prev)]['word']
        #     if wms[idx_prev]['speaker'] < 2:
        #         assert wms[idx_prev]['speaker'] == seg_next[idx_s_next+(idx_prev-idx_s_prev)]['speaker']

        ssm = ssm[:idx_s_prev] + seg_next[idx_s_next:]

    return wsm, ssm

# wms matching
def _get_matching_wsm_index(wsm_next, wsm_prev, ms_offset):
    for idx_s_next, s_next in enumerate(wsm_next):
        for idx_s_prev in range(len(wsm_prev) - 5, -1, -1):
            s_prev = wsm_prev[idx_s_prev]
            # Make sure this is an utterance with speaker_id < 2, and not some spuriously assigned utterance
            if s_prev['speaker'] > 1:
                continue
            # Check that next 5 elements are equal
            b_match = True
            for i in range(5):
                if not _check_wsm_element_match(wsm_next[idx_s_next + i], wsm_prev[idx_s_prev + i], ms_offset):
                    b_match = False
            if b_match:
                return idx_s_next, idx_s_prev, (s_prev['speaker'] == s_next['speaker'])
    return None

def _check_wsm_element_match(e1, e2, ms_offset):
    return ((e1['start_time'] + ms_offset == e2['start_time']) and
            (e1['end_time'] + ms_offset == e2['end_time']) and (
                    e1['word'] == e2['word']))

# ssm matching
def _get_matching_ssm_index(ssm_next, ssm_prev, ms_offset):
    for idx_s_next, s_next in enumerate(ssm_next):
        for idx_s_prev in range(len(ssm_prev) - 1, -1, -1):
            s_prev = ssm_prev[idx_s_prev]
            # Make sure this is an utterance with speaker_id < 2, and not some spuriously assigned utterance
            if int(s_prev['speaker'].split(' ')[1]) > 1:
                continue
            if _check_ssm_element_match(ssm_next[idx_s_next], ssm_prev[idx_s_prev], ms_offset):
                return idx_s_next, idx_s_prev, (s_prev['speaker'] == s_next['speaker'])
    return None

def _check_ssm_element_match(e1, e2, ms_offset):
    return ((e1['start_time'] + ms_offset == e2['start_time']) and
            (e1['end_time'] + ms_offset == e2['end_time']) and (
                    e1['text'] == e2['text']))
