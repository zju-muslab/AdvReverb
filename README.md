# AdvReverb
This repo is the pytorch implementation of the paper "AdvReverb: Rethinking the Stealthiness of Audio Adversarial Examples to Human Perception"

## Installation
Clone this repo to your local path:
```
git clone git@github.com:AdvReverb/AdvReverb.git
```
Create a virtual python environment with `python>=3.8.0` and install the packages:
```
pip install -r requirements.txt
```

## Usage
### Prepare speech datasets
Download the following speech datasets:
- [Google Speech Command v0.02](https://tensorflow.google.cn/datasets/catalog/speech_commands)
- [VoxCeleb v1 (test)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
- [LibriSpeech (test)](http://www.openslr.org/12)

Extract the `path`, `label`, `interval` information from each dataset to a csv file, e.g.,:
- `data/kws_speechcommand_3200.csv`
- `data/asi_voxceleb_3200.csv`
- `data/asr_librispeech_3200.csv`

where `path` is the audio file path, `label` refers to the command id, speaker id, or utterance transcript, and `interval` is a list of tuples that represent each segment (`start`, `end`, `phoneme group`) for convolution in AdvReverb.

> Note that segments with fixed length (e.g., 3200 points) or dynamic lengths produced by forced aligment tools (e.g., [Motreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html)) are both ok.

### Prepare target audio systems
- Pretrained BC-ResNet for KWS is provided in `ckpt/kws`
- Pretrained Ecapa-TDNN for ASI is provided in `ckpt/asi`
- Pretrained Wav2Vec2 for ASR is too larget and can be downloaded at [SpeechBrain](https://huggingface.co/speechbrain)

### Run adversarial attacks
Run AdvReverb against a KWS system (BC-ResNet) with speficied parameters:
```
python run_adv.py hydra.job.chdir=True task=kws adv=rir adv.epoch=200 adv.alpha=5 target=1 wav_len=16000 carrier=speech gpu=true device=0
```
Run AdvReverb against an ASI system (Ecapa-TDNN) with speficied parameters:
```
python run_adv.py hydra.job.chdir=True task=asi adv=rir adv.epoch=1000 adv.alpha=5 target=1 wav_len=48000 carrier=speech gpu=true device=1
```
Run AdvReverb against an ASR system (Wav2Vec2) with speficied parameters:
```
python run_adv.py hydra.job.chdir=True task=asr adv=rir adv.epoch=3000 adv.alpha=0.1 target=1 wav_len=null carrier=speech gpu=true device=2
```
You can also run PGD, C&W, PsychoMask, KenKu by specifing the `adv` option and other attack parameters corresponding to the config directory `config/adv/[attack_name]`, e.g.,
```
python run_adv.py hydra.job.chdir=True task=asi adv=pgd adv.epoch=1000 adv.epsilon=0.008 target=1 wav_len=48000 carrier=speech gpu=true device=1
```
Each experiment has a output directory `outputs/exp_name`. You can add the `save_wav=true` option to save the original samples and adversarial examples at `outputs/exp_name/wav`.

### Count objective metrics
For each attack experiment, you can count the objective metrics (SNR, MCD, PESQ) by running:
```
python calc_metrics.py --data-dir /path/to/exp_name/wav --csv-path /path/to/exp_name/objective_result.csv
```

## Citation
```
@article{advreverb,
    author = {Meng Chen,  Li Lu, Jiadi Yu, Zhongjie Ba, Feng Lin and Kui Ren},
    title  = {{AdvReverb}: Rethinking the Stealthiness of Audio Adversarial Examples to Human Perception},
    year   = {2023}
}
```