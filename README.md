# Parallel WaveGAN (+ MelGAN & Multi-band MelGAN) implementation with Pytorch

# 한국어 수정중입니다.
> Source of the figure: https://arxiv.org/pdf/1910.11480.pdf
![](https://user-images.githubusercontent.com/22779813/68081503-4b8fcf00-fe52-11e9-8791-e02851220355.png)

## Requirements

This repository is tested on Ubuntu 16.04 with a GPU Titan V.

- Python 3.6+
- Cuda 10.0
- CuDNN 7+
- NCCL 2+ (for distributed multi-gpu training)
- libsndfile (you can install via `sudo apt install libsndfile-dev` in ubuntu)
- jq (you can install via `sudo apt install jq` in ubuntu)
- sox (you can install via `sudo apt install sox` in ubuntu)

Different cuda version should be working but not explicitly tested.  
All of the codes are tested on Pytorch 1.0.1, 1.1, 1.2, 1.3.1, 1.4, and 1.5.1.

Pytorch 1.6 works but there are some issues in cpu mode (See #198).

## Setup

You can select the installation method from two alternatives.

### A. Use pip

```bash
$ git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
$ cd ParallelWaveGAN
$ pip install -e .
# If you want to use distributed training, please install
# apex manually by following https://github.com/NVIDIA/apex
$ ...
```
Note that your cuda version must be exactly matched with the version used for the pytorch binary to install apex.  
To install pytorch compiled with different cuda version, see `tools/Makefile`.

### B. Make virtualenv

```bash
$ git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
$ cd ParallelWaveGAN/tools
$ make
# If you want to use distributed training, please run following
# command to install apex.
$ make apex
```

Note that we specify cuda version used to compile pytorch wheel.  
If you want to use different cuda version, please check `tools/Makefile` to change the pytorch wheel to be installed.

## Recipe

This repository provides [Kaldi](https://github.com/kaldi-asr/kaldi)-style recipes, as the same as [ESPnet](https://github.com/espnet/espnet).  
Currently, the following recipes are supported.

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): English female speaker
- [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut): Japanese female speaker
- [CSMSC](https://www.data-baker.com/open_source.html): Mandarin female speaker
- [CMU Arctic](http://www.festvox.org/cmu_arctic/): English speakers
- [JNAS](http://research.nii.ac.jp/src/en/JNAS.html): Japanese multi-speaker
- [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html): English multi-speaker
- [LibriTTS](https://arxiv.org/abs/1904.02882): English multi-speaker
- [YesNo](https://arxiv.org/abs/1904.02882): English speaker (For debugging)

To run the recipe, please follow the below instruction.

```bash
# Let us move on the recipe directory
$ cd egs/ljspeech/voc1

# Run the recipe from scratch
$ ./run.sh

# You can change config via command line
$ ./run.sh --conf <your_customized_yaml_config>

# You can select the stage to start and stop
$ ./run.sh --stage 2 --stop_stage 2

# If you want to specify the gpu
$ CUDA_VISIBLE_DEVICES=1 ./run.sh --stage 2

# If you want to resume training from 10000 steps checkpoint
$ ./run.sh --stage 2 --resume <path>/<to>/checkpoint-10000steps.pkl
```

See more info about the recipes in [this README](./egs/README.md).

## How-to-use pretrained models

### Analysis-synthesis

Here the minimal code is shown to perform analysis-synthesis using the pretrained model.

```bash
# Please make sure you installed `parallel_wavegan`
# If not, please install via pip
$ pip install parallel_wavegan

# You can download the pretrained model from terminal
$ python << EOF
from parallel_wavegan.utils import download_pretrained_model
download_pretrained_model("<pretrained_model_tag>", "pretrained_model")
EOF

# You can get all of available pretrained models as follows:
$ python << EOF
from parallel_wavegan.utils import PRETRAINED_MODEL_LIST
print(PRETRAINED_MODEL_LIST.keys())
EOF

# Now you can find downloaded pretrained model in `pretrained_model/<pretrain_model_tag>/`
$ ls pretrain_model/<pretrain_model_tag>
  checkpoint-400000steps.pkl    config.yml    stats.h5

# These files can also be downloaded manually from the above results

# Please put an audio file in `sample` directory to perform analysis-synthesis
$ ls sample/
  sample.wav

# Then perform feature extraction -> feature normalization -> sysnthesis
$ parallel-wavegan-preprocess \
    --config pretrain_model/<pretrain_model_tag>/config.yml \
    --rootdir sample \
    --dumpdir dump/sample/raw
100%|████████████████████████████████████████| 1/1 [00:00<00:00, 914.19it/s]
$ parallel-wavegan-normalize \
    --config pretrain_model/<pretrain_model_tag>/config.yml \
    --rootdir dump/sample/raw \
    --dumpdir dump/sample/norm \
    --stats pretrain_model/<pretrain_model_tag>/stats.h5
2019-11-13 13:44:29,574 (normalize:87) INFO: the number of files = 1.
100%|████████████████████████████████████████| 1/1 [00:00<00:00, 513.13it/s]
$ parallel-wavegan-decode \
    --checkpoint pretrain_model/<pretrain_model_tag>/checkpoint-400000steps.pkl \
    --dumpdir dump/sample/norm \
    --outdir sample
2019-11-13 13:44:31,229 (decode:91) INFO: the number of features to be decoded = 1.
[decode]: 100%|███████████████████| 1/1 [00:00<00:00, 18.33it/s, RTF=0.0146]
2019-11-13 13:44:37,132 (decode:129) INFO: finished generation of 1 utterances (RTF = 0.015).

# you can find the generated speech in `sample` directory
$ ls sample
  sample.wav    sample_gen.wav
```

## References

- [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- [LiyuanLucasLiu/RAdam](https://github.com/LiyuanLucasLiu/RAdam)
- [MelGAN](https://arxiv.org/abs/1910.06711)
- [descriptinc/melgan-neurips](https://github.com/descriptinc/melgan-neurips)
- [Multi-band MelGAN](https://arxiv.org/abs/2005.05106)

## Acknowledgement

The author would like to thank Ryuichi Yamamoto ([@r9y9](https://github.com/r9y9)) for his great repository, paper, and valuable discussions.

## Author

Tomoki Hayashi ([@kan-bayashi](https://github.com/kan-bayashi))  
E-mail: `hayashi.tomoki<at>g.sp.m.is.nagoya-u.ac.jp`
