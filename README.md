## [NeurIPS 2022] Learning Optical Flow from Continuous Spike Streams

<h4 align="center"> Rui Zhao<sup>1,2</sup>, Ruiqin Xiong<sup>1,2</sup>, Jing Zhao<sup>3</sup>, Zhaofei Yu<sup>1,2,4</sup>, Xiaopeng Fan<sup>5</sup>, Tiejun Huang<sup>1,2,4</sup> </h4>
<h4 align="center">1. National Engineering Research Center of Visual Technology (NERCVT), Peking University<br>
2. Institute of Digital Media, School of Computer Science, Peking University<br>
3. National Computer Network Emergency Response Technical Team<br>
4. Institute for Artificial Intelligence, Peking University<br>
5. School of Computer Science and Technology, Harbin Institute of Technology</h4><br> 
This repository contains the official source code for our paper:

Learninng Optical Flow from Continuous Spike Streams

NeurIPS 2022

[Paper](https://openreview.net/pdf?id=3vYkhJIty7E)

[Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202022/55189.png)

## Environment

You can choose cudatoolkit version to match your server. The code is tested on PyTorch 1.10.1+cu113.

```bash
conda create -n spike2flow python==3.9
conda activate spike2flow
# You can choose the PyTorch version you like, we recommand version >= 1.10.1
# For example
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirements.txt
```

## Prepare the Data

#### 1. Download and deploy the RSSF dataset

[Link of Real Scenes with Spike and Flow](https://github.com/ruizhao26/RSSF)

#### 2. Set the path of RSSF dataset in your serve

In the line2 of `configs/spike2flow.yml`

#### 3. Pre-processing for DSFT (Differential of Spike Firing Time)

It's not difficult to compute DSFT in real time, but in this version of the code, we choose to pre-processing the DSFT and save it in .h5 format to since the GPU memory resource is limited.

You can pre-processing the DSFT using the following command

```bash
cd datasets && 
python3 dat_to_DSFT_h5.py \
--rssf_root 'your root of RSSF dataset'\
--device 'cuda'
```

We will release the code of getting DSFT in real time in the future.

## Evaluate

```bash
python3 main.py --eval --pretrained ckpt/spike2flow.pth
```

## Train

```bash
python3 main.py \
--configs ./configs/spike2flow.yml \
--batch_size 6 \
--learning_rate 3e-4 \
--num_workers 12 \
--decay_interval 10 \
--decay_factor 0.7 
```

We recommended to redirect the output logs by adding
`>> spike2flow.txt 2>&1` 
to the last of the above command for management.

## Citations

If you find this code useful in your research, please consider citing our paper.

```
@inproceedings{zhao2022learning,
  title={Learninng optical flow from continuous spike streams},
  author={Zhao, Rui and Xiong, Ruiqin and Zhao, Jing and Yu, Zhaofei and Fan, Xiaopeng and Huang, Tiejun},
  booktitle={Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

If you have any questions, please contact:  
ruizhao@stu.pku.edu.cn