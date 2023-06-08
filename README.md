# [Model/Data/Code] INTELLIGENS: Providing Better Features for Whole Slide Image Based on Self-distillation 
<!-- select Model and/or Data and/or Code as needed>
**[Zhejiang Lab](https://www.zhejianglab.com/)**
[[`Paper`](https://)] [[`Blog`](https://)] [[`Demo`](https://)] 
### Welcome to OpenMEDLab! 👋

<!--
**Here are some ideas to get you started:**
🙋‍♀️ A short introduction - what is your organization all about?
🌈 Contribution guidelines - how can the community get involved?
👩‍💻 Useful resources - where can the community find your docs? Is there anything else the community should know?
🍿 Fun facts - what does your team eat for breakfast?
🧙 Remember, you can do mighty things with the power of [Markdown](https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
-->


---

## Key Features

This repository provides the official implementation and pretrained models for INTELLIGENS. For details, the related paper would be released soon.
INTELLIGENS models produce robust and high-quality feature representations for whole slide image (WSI). The features can be directly employed with classifiers on slide-level multi-class subtyping problems. These trained models also perform well on patch-level classification tasks with slight fine-tuning. The models were pretrained on a dataset containing more than 10,000 WSIs without using any labels or annotations.

## Links

- [Paper](https://)
- [Model](https://)
- [Code](https://) 
<!-- [Code] may link to your project at your institute>


<!-- give a introduction of your project -->
## Details

intro text here.

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/openmedlab/sampleProject/blob/main/diagram_sample.png"></a>
</div>

More intro text here.


## Dataset Links

- [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094#learn_the_details)
- [Kidney and Kidney Tumor Segmentation (KiTS)](https://kits21.kits-challenge.org/participate#download-block)

## Get Started

**Some Main Requirements**  
> Linux (Tested on Ubuntu 18.04)   
> Python==3.9.16
> Pytorch==2.0.0  
> torch==1.11.0  
> torchvision==0.15.0
> openslide-python==1.2.0  
> opencv-python==4.7.0.72 
The training is performed using Pytorch on a Linux environment. It requires the main packages metioned above as well as a number of other 3rd party packages. To setup all the required dependencies for training and evaluation, please follow the instructions below:  
**Installation**
*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)** - Clone the repository and then create and activate an `INTELLIGENS` conda environment using the provided environment definition `environment.yaml`:
```bash
conda env create -f environment.yaml
conda activate INTELLIGENS
```
*[pip](https://pip.pypa.io/en/stable/getting-started/)* - Clone the repository and then use the provided `requirements.txt` to install the dependencies:

```bash
pip install -r requirements.txt
```

**Data Preparation**
The models were trained with a large WSI dataset, which contains more than 10,000 slides from multiple datasets, including about 6,000 slides from The Cancer Genome Atlas Program (TCGA), 1,000 slides from Camelyon17 and more than 3,000 private slides. For each slide, we used [CLAM](https://github.com/mahmoodlab/CLAM) to segment the tissue and exluding the blank areas, then extracted the patches within the segmented regions, saved the coordinates of patches in a .npy file. The following example assumes that the whole slide image data in well known standard formats (.svs, .tiff etc.) and the coordinates files are stored under a folder named DATA_DIRECTORY
```bash
DATA_DIRECTORY/
    SUBDATASET1/
        ├── slide_1.svs
        ├── slide_1.npy
        ├── slide_2.svs
        ├── slide_2.npy
        └── ...
    SUBDATASET2/
	    ├── slide_1.tiff
        ├── slide_1.npy
        ├── slide_2.tiff
        ├── slide_2.npy
        └── ...
```

**Training**
This codebase was developed with Python version 3.9.16, PyTorch version 2.0.0, CUDA 11.7 and torchvision 0.15.0. The arguments used can be found in the `args` column of the [pretrained models section](https://github.com/facebookresearch/dino#pretrained-models). Following is a vanilla training implementation example on 1 nodes with 4 GPUs each (total 4 GPUs):
```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
--master_addr="xx.xxx.xxx.xxx" --master_port=xxxx  train.py --patch_size 16 \
--arch "vit_base" --batch_size_per_gpu xxx --use_fp16 0 --output_dir ./output_dir 
```


**Validation**
```bash
python DDD
```


**Testing**
```bash
python DDD
```

## 🙋‍♀️ Feedback and Contact

- Email
- Webpage 
- Social media


## 🛡️ License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

A lot of code is modified from [monai](https://github.com/Project-MONAI/MONAI).

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@article{John2023,
  title={paper},
  author={John},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```

