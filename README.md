# Interviews To Emotions
A pipeline to convert recorded interviews into segments, and process each segment with an emotion prediction model.

## Usage
Have a look at the bottom part of ```speech_transcription/processor.py``` for examples.

Essentially, if you want to process a single file, use:
```python
Processor.process_file(file=path_to_file,
                       emo_classifier=EmoClassifier.EMO2VEC,
                       language=Language.CN)
```

To process an entire folder (in this case of mp3 files):
```python
Processor.process_folder(folder=path_to_folder,
                         file_ext='mp3',
                         language=Language.NL)
```

For each processed file, a 3 new files will be generate. Assuming file ```dummy.mp3``` is processed, the following
new files will be created:
* ```dummy.srt```: contains a line-by-line transcription of the audio, with timestamps and speaker id.
* ```dummy.txt```: contains a full text transcription, with text grouped per segment (i.e., continuous speech by same speaker), with speaker id.
* ```dummy_emo.txt```: contains a segment-by-segment transcription of the audio + predicted emotion per segment.

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
