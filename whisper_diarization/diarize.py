import logging
import os
import re

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

def diarize_file(file: str=None, language='nl'):
    """

    :param file: path to file
    :param language: language spoken in the audio file
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
    audio_waveform = faster_whisper.decode_audio(vocal_target)
    print("Creating token suppressor...")
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if params['suppress_numerals']
        else [-1]
    )

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

    print("Writing results to disc!")
    with open(f"{os.path.splitext(params['file'])[0]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{os.path.splitext(params['file'])[0]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    cleanup(temp_path)
