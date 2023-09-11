# JPEG-Ano-ViT

A code for toy project for attaching  [RGB no more: Minimally Decoded JPEG Vision Transformers](https://openaccess.thecvf.com/content/CVPR2023/html/Park_RGB_No_More_Minimally-Decoded_JPEG_Vision_Transformers_CVPR_2023_paper.html)\ into  [AnoViT: Unsupervised Anomaly Detection and Localization with Vision Transformer-based Encoder-Decoder](https://ieeexplore.ieee.org/abstract/document/9765986)\

some of the code is implement from those paper's official code.




## Usage
### Install

* This install guide is from [RGB no more: Minimally Decoded JPEG Vision Transformers Official Code](https://github.com/JeongsooP/RGB-no-more )

*Note that modified `libjpeg` requires **Linux** to compile properly. Other OS are not supported.*

- Clone this repo:

```bash
git clone https://github.com/johnbuzz98/JPEG-Ano-ViT.git
cd JPEG-Ano-ViT
```

- Create a conda virtual environment and activate it:

```bash
conda create -n jpeganovit python=3.10
conda activate jpeganovit
```

- Install `PyTorch>=1.12.1` `CUDA>=11.3` (we used `PyTorch=1.12.1` and `CUDA=11.3` in our paper):
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 "llvm-openmp<16.0" -c pytorch
```

*We recommend using `llvm-openmp<16.0` until the [multiprocessing issue](https://github.com/pytorch/pytorch/issues/101850) is fixed.*

- Install `gcc_linux-64==12.2.0`, `gxx_linux-64==12.2.0` and other necessary libraries:
```bash
conda install -c conda-forge gcc_linux-64==12.2.0 gxx_linux-64==12.2.0 torchmetrics torchinfo tensorboard einops scipy yacs pandas timm imageio iopath psutil
```

*Make sure the versions for `gcc_linux-64` and `gxx_linux-64` are **exactly 12.2.0.***

- Install other requirements:

```
pip install -r requirements.txt
```

- Compile `dct_manip` -- a modified `libjpeg` handler:

  - Open `dct_manip/setup.py`
  - Modify `include_dirs` and `library_dirs` to include your include and library folder.
  - Modify `extra_objects` to the path containing `libjpeg.so`
  - Modify `headers` to the path containing `jpeglib.h`

  - Run `cd dct_manip`
  - Run `pip install .`


### Train
```bash
. Scripts/scripts.sh
```

# Result

| Target      | JPEG_AnoViT | JPEG_AnoViT | AnoViT      | AnoViT      |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| ----------- | Image AUROC | Pixel AUROC | Image AUROC | Pixel AUROC |
| Bottle      | 0.8608      | 0.442       |  |  |
| Cable       | 0.7375      | 0.5988      |  |  |
| Capsule     | 0.5121      | 0.5006      |  |  |
| Hazelnut    |             |             |  |  |
| Metal Nut   |             |             |  |  |
| Pill        |             |             |  |  |
| Screw       |             |             |  |  |
| Toothbrush  |             |             |  |  |
| Transistor  |             |             |  |  |
| Zipper      |             |             |  |  |
| Carpet      | 0.5867      | 0.5279      |  |  |
| Grid        |             |             |  |  |
| Leather     |             |             |  |  |
| Tile        |             |             |  |  |
| Wood        |             |             |  |  |
# Known Issue
  - *CV2 호환 안됨, 
  - *torch load 이후 dct_manip 불러와야함  
# Reference
```plain
@InProceedings{Park_2023_CVPR,
    author    = {Park, Jeongsoo and Johnson, Justin},
    title     = {RGB No More: Minimally-Decoded JPEG Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22334-22346}
}
```
```plain
@article{lee2022anovit,
  title={AnoViT: Unsupervised anomaly detection and localization with vision transformer-based encoder-decoder},
  author={Lee, Yunseung and Kang, Pilsung},
  journal={IEEE Access},
  volume={10},
  pages={46717--46724},
  year={2022},
  publisher={IEEE}
}
```
