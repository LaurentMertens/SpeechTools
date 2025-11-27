"""
Use whisper-timestamped to transcribe audio and get timestamp at word-level.

!!! ATTENTION !!!
No longer used. Switched to using whisper_diarization instead.
I keep the code to have an example of how this works, just in case I need it in the future.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os

import whisper_timestamped as whisper

from config import Config


class TimeStamp:
    @staticmethod
    def process_file(file: str):
        """

        :param file: path to audio file to be processed
        :return:
        """
        audio = whisper.load_audio(file)

        # Default tiny model
        #model = whisper.load_model("tiny", device="cuda")
        # Large OpenAI v3 model; 2.88G download;
        # For other options, check, e.g., https://github.com/openai/whisper.
        # That page contains a table with other options and their size.
        model = whisper.load_model("large-v3", device="cuda")

        result = whisper.transcribe(model,
                                    audio,
                                    language="nl",
                                    vad=True  # Detect voice activation first
                                    )
        print(result)

        #
        # import json
        # print(json.dumps(result, indent = 2, ensure_ascii = False))

if __name__ == '__main__':
    TimeStamp.process_file(file=Config.FILE_DEMO)

