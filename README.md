# INTELLIGENS: Providing Better Features for Whole Slide Image Based on Self-distillation 

**[Zhejiang Lab](https://www.zhejianglab.com/)**

[[`Paper`](https://)] [[`Blog`](https://)] [[`Demo`](https://)] 

This repository provides the official implementation and pretrained models for INTELLIGENS. More details will be included in the related paper which would be released soon.

INTELLIGENS models produce robust and high-quality feature representations for whole slide image (WSI). The features can be directly employed with classifiers on slide-level multi-class subtyping problems. These trained models also perform well on patch-level classification tasks with slight fine-tuning. The models were pretrained on a dataset containing more than 10,000 WSIs without using any labels or annotations.

## Updates / TODOs
Please follow this repository for more updates.

* 06/09/2023: First upload of INTELLIGENS. The weights of vitb-based backbone is added. The scripts for reproducing the slide-level multi-class subtyping tasks are provided. 


- [ ] Add results for more downstream tasks.
- [ ] Add pretrained model weights with different scale of parameter numbers.
- [ ] Provide a jupyter notebook for the complete workflow.
- [ ] ...



## Pretrained models
<table style="margin: auto">
  <tr>
    <th>model</th>
    <th>params</th>
    <th colspan="4">download</th>
  </tr>
  <tr>
    <td>ViT-S/16 </td>
    <td align="right">43 M</td>
    <td><a>backbone only</a></td>
    <td><a>full ckpt</a></td>
    <td><a>args</a></td>
    <td><a>logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16 </td>
    <td align="right">0.11 B</td>
    <td><a>backbone only</a></td>
    <td><a>full ckpt</a></td>
    <td><a>args</a></td>
    <td><a>logs</a></td>
  </tr>
  <tr>
    <td>ViT-L/16 </td>
    <td align="right">0.33 B</td>
    <td><a>backbone only</a></td>
    <td><a>full ckpt</a></td>
    <td><a>args</a></td>
    <td><a>logs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td align="right">49 M</td>
    <td><a>backbone only</a></td>
    <td><a>full ckpt</a></td>
    <td><a>args</a></td>
    <td><a>logs</a></td>
  </tr>
</table>


## Installation

**Some Main Requirements**  
> Linux (Tested on Ubuntu 18.04)   
> Python==3.9.16  
> Pytorch==2.0.0  
> torch==1.11.0  
> torchvision==0.15.0    
> openslide-python==1.2.0  
> opencv-python==4.7.0.72  


The training is performed using Pytorch on a Linux environment. It requires the main packages metioned above as well as a number of other 3rd party packages. To setup all the required dependencies for training and evaluation, please follow the instructions below:  

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)** - Clone the repository and then create and activate an `INTELLIGENS` conda environment using the provided environment definition `environment.yaml`:
```bash
conda env create -f environment.yaml
conda activate INTELLIGENS
```
*[pip](https://pip.pypa.io/en/stable/getting-started/)* - Clone the repository and then use the provided `requirements.txt` to install the dependencies:

```bash
pip install -r requirements.txt
```
## Dataset

**Dataset Links**
- [The Cancer Genome Atlas Program (TCGA)](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)
- [CAMELYON17](https://camelyon17.grand-challenge.org/Home)

**Data Preparation**
The models were trained with a large WSI dataset, which contains more than 10,000 slides from multiple datasets, including about 6,000 slides from The Cancer Genome Atlas Program (TCGA), 1,000 slides from CAMELYON17 and more than 3,000 private slides. For each slide, we used [CLAM](https://github.com/mahmoodlab/CLAM) to segment the tissue and exluding the blank areas, then extracted the patches within the segmented regions, saved the coordinates of patches in a .npy file. The following example assumes that the whole slide image data in well known standard formats (.svs, .tiff etc.) and the coordinates files are stored under a folder named DATA_DIRECTORY
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
## Training

This codebase was developed with Python version 3.9.16, PyTorch version 2.0.0, CUDA 11.7 and torchvision 0.15.0. The arguments used can be found in the `args` column of the [pretrained models section](https://github.com/facebookresearch/dino#pretrained-models). Following is a vanilla training implementation example on 1 nodes with 4 GPUs (total 4 GPUs):
```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
--master_addr="xx.xxx.xxx.xxx" --master_port=xxxx  train.py --patch_size 16 \
--arch "vit_base" --batch_size_per_gpu xxx --use_fp16 0 --output_dir ./output_dir 
```

## Evaluation

### Slide-level multi-class subtyping task
For this task, we adopted the multiple instance learning (MIL) framework and test models' performance on several dataset, including TCGA-BRCA, TCGA-RCC, TCGA-NSCLC, [CAMELYON16](https://camelyon16.grand-challenge.org), [PANDA](https://panda.grand-challenge.org), etc. The features for each slides are pre-extracted due to the large scale of WSI. Then the MIL classifier is trained on these features according to the common practices. The extracted feature embeddings, the trained models' weights and the test resluts are provided:
<table style="margin: auto">
  <tr>
    <th>Dataset</th>
    <th>Acc</th>
    <th>AUC</th>
    <th colspan="3">download</th>
  </tr>
  <tr>
    <td><a>TCGA-BRCA</a></td>
    <td><a>XX.XX</a></td>
    <td><a>XX.XX</a></td>
    <td><a>args</a></td>
    <td><a>weights</a></td>
    <td><a>embeddings</a></td>
  </tr>
  <tr>
    <td><a>TCGA-RCC</a></td>
    <td><a>XX.XX</a></td>
    <td><a>XX.XX</a></td>
    <td><a>args</a></td>
    <td><a>weights</a></td>
    <td><a>embeddings</a></td>
  </tr>
  <tr>
    <td><a>TCGA-NSCLC</a></td>
    <td><a>0.8878</a></td>
    <td><a>0.9539</a></td>
    <td><a>args</a></td>
    <td><a>weights</a></td>
    <td><a>embeddings</a></td>
  </tr>
  <tr>
    <td><a>CAMELYON16</a></td>
    <td><a>0.9535</a></td>
    <td><a>0.9756</a></td>
    <td><a>args</a></td>
    <td><a>weights</a></td>
    <td><a>embeddings</a></td>
  </tr>
  <tr>
    <td><a>PANDA</a></td>
    <td><a>0.9432</a></td>
    <td><a>0.9785</a></td>
    <td><a>args</a></td>
    <td><a>weights</a></td>
    <td><a>embeddings</a></td>
  </tr>
</table>
You can easily reproduce the test results by downloading the feature embeddings and running

```bash
python 
```



Here, we provide an example using [CLAM](https://github.com/mahmoodlab/CLAM) as classifier for training and testing.  

**Data Preparation**
Download or generate the feature embeddings using the pre-trained models provided at [pretrained models section](https://github.com/wustone1995/WSI_FoundationModel#pretrained-models). 
```bash
python feature_extract.py
```
The following example assumes the embedding files are stored under a folder named FEAT_DIRECTORY.

```bash
FEAT_DIRECTORY/
    pt_files/
        ├── slide_1.pt
        ├── slide_2.pt
        ├── slide_3.pt
        └── ...
```	
Then train and test the model by 
```bash
python train.py
python test.py
```



## 🛡️ License



## 🙏 Acknowledgement

- Code for weakly-supervised subtyping was largely adapted from [CLAM](https://github.com/mahmoodlab/CLAM) and [DTFD-MIL](https://github.com/hrzhang1123/DTFD-MIL).
- Code for self-supervised pretraining was largely adapted via making modifications to [DINO](https://github.com/facebookresearch/dino)


## 📝 Citation

The related paper will be released soon.

