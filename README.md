# Interviews To Emotions
A pipeline to convert recorded interviews into segments, and process each segment with an emotion prediction model.

## Usage
Have a look at the bottom part of ```speech_transcription/processor.py``` for examples.

Essentially, if you want to process a single file, use:
```python
Processor.process_file(file=_file_nl,
                       b_print_all_emos=True,
                       b_print_emo_probs=True,
                       window_size=60,
                       window_stride=20,
                       emo_classifier=EmoClassifier.EMO2VEC,
                       language=Language.NL)
```

To process an entire folder (in this case of mp3 files):
```python
Processor.process_folder(folder=path_to_folder,
                         file_ext='mp3',
                         b_print_all_emos=True,
                         b_print_emo_probs=True,
                         window_size=60,
                         window_stride=20,
                         emo_classifier=EmoClassifier.EMO2VEC,
                         language=Language.NL)
```

Each processed file will first be split into segments, with each segment representing a continuous passage spoken by a
same speaker, as determined by Whisper. To be more precise, a slightly customized version of the ```whisper-diarization```
repository is used. For the original, see [https://github.com/MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization).

Then, each segment will be processed by applying a sliding window approach. The parameter ```window_size``` represents
the size of the window in seconds, ```window_stride``` means the window will jump forward by this many seconds each time.
E.g., if the segment is 74s long, ```window_size```=60 and ```window_stride```=20 (the default values), then first the
timeframe from 0s to 60s will be processed, followed by the timeframe from 20s to 74s. Note that if the timeframe duration
is less than 1/5th of the ```window_size```, the timeframe will not be processed. E.g., assuming a 25s segment and the same
default ```window_size``` and ```window_stride``` values, the timeframe from 20s to 25s will not processed, as 5s is
less than 1/5th of the 60s ```window_size```.

For each processed file, 3 new files will be generate. Assuming file ```dummy.mp3``` is processed, the following
new files will be created:
* ```dummy.srt```: contains a line-by-line transcription of the audio, with timestamps and speaker id.
* ```dummy.txt```: contains a full text transcription, with text grouped per segment (i.e., continuous speech by same speaker), with speaker id.
* ```dummy_emo.txt```: contains a segment-by-segment transcription of the audio + predicted emotion(s) per segment.

### Languages
You can specify one of two languages:
* ```Language.NL```: Dutch
* ```Language.CN```: Chinese


### Emotion Detection methods
You can specify one of two Emotion Detection methods:
* ```EmoClassifier.EMO2VEC```   : uses <https://github.com/ddlBoJack/emotion2vec?tab=readme-ov-file#inference-with-checkpoints>
* ```EmoClassifier.SPEECH2EMO```: uses <https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3>

## Problem with missing libcudnn9-cuda-xx
If you encounter a problem with a ```libcudnn9-cuda-xx``` library that can not be found,
perform the following steps to resolve the issue:
1. Use ```torch.backends.cudnn.version()``` to verify which cuDNN version PyTorch is using.
2. Go to <https://developer.download.nvidia.com/compute/cuda/repos/distro/arch/>, where you should replace
```distro``` with your distribution and ```arch``` with your CPU architecture. E.g., if you are using Ubuntu 24.04
on a x64 CPU, go to <https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/>. You can start
from <https://developer.download.nvidia.com/compute/cuda/repos/>, and take it from there.
3. On that webpage, look for the ```libcudnn9-cuda-xx``` file that matches the cuDNN version PyTorch is using
   (see step 1.), and download it.
4. Install the library downloaded in step 3, and try running the code again. If all went well, the error should be
resolved.

## Licensing
This repository is made available under an MIT license (see [LICENSE.md](./LICENSE.md)).

## Contact
Author: Laurent Mertens \
Mail: [contact@laurentmertens.com](contact@laurentmertens.com)
