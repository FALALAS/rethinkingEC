# Rethinking Exposure Correction for Spatially Non-uniform Degradation

## Introduction

This repository is the official implementation  of "Rethinking Exposure Correction for Spatially Non-uniform Degradation"

[Ao Li](https://liaosite.github.io/)<sup>1#</sup>,  Jiawei Sun<sup>1#</sup>, Le Dong<sup>1*</sup>, Zhenyu Wang<sup>2</sup>, [Weisheng Dong](https://see.xidian.edu.cn/faculty/wsdong/index_en.htm)<sup>1</sup>

<sup>1</sup>School of Artificial Intelligence, Xidian University

<sup>2</sup>Hangzhou Institute of technology, Xidian University

*: Corresponding Author.

#: Equal Contribution.

## Setup
* Install the conda environment
```bash
conda create -n rethinkingEC python=3.10
conda activate rethinkingEC
```
*Install Pytorch
```bash
# CUDA 11.8
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```
*Install other requirements
```bash
pip install -r requirements.txt
```

## Dataset
Please refer to the link below to download the dataset.Then remember to modify the dataset options in OPTIONS accordingly.
* [LCDP](https://www.whyy.site/paper/lcdp)
* [MSEC](https://github.com/mahmoudnafifi/Exposure_Correction)
* [SICE](https://github.com/KevinJ-Huang/ExposureNorm-Compensation)
* [REED](https://github.com/juvenoia/REED)

## Run
* Testing. The testing configuration is in options/test/. Please put all the pre-training weights in the ./experiments. After running, the results of the visualization will be saved in . /results.
```bash
python test.py -opt options/test/test_lcdp.yml
python test.py -opt options/test/test_msec.yml
python test.py -opt options/test/test_sice.yml
python test.py -opt options/test/test_reed.yml
```
* Training. The training configuration is in options/train/.
```bash
python train.py -opt options/train/train_lcdp.yml
python train.py -opt options/train/train_sice.yml
python train.py -opt options/train/train_msec.yml
```

## Pretrained Models
We provide the pretrained checkpoints in [Baidu](https://pan.baidu.com/s/1yj5FU7e3U27ucJ2XCezlsw?pwd=8563). Download and place them into ./experiments/ directory. The expected structure is:
```text
RSNet/experiments/
├── MSEC_weights.pth       
├── LCDP_weights.pth
├── SICE_weights.pth  
└── REED_weights.pth       
```

## Citation

Please cite the following paper if you feel our work useful to your research:

```bibtex
@misc{li2026rethinkingexposurecorrectionspatially,
      title={Rethinking Exposure Correction for Spatially Non-uniform Degradation}, 
      author={Ao Li and Jiawei Sun and Le Dong and Zhenyu Wang and Weisheng Dong},
      year={2026},
      eprint={2604.04136},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.04136}, 
}
```