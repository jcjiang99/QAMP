# FSMIS via QAMP
Codes for the following paper:
Learning Query-Assisted Multiple Prototypes for Few-Shot Medical Image Segmentation

### Abstract
Few-shot semantic segmentation, which is dedicated to the generalization of models to segment novel classes with scarce annotated samples, has achieved tremendous progress recently due to the significant advancement of deep CNNs. However, existing approaches in medical scenarios, namely Few-Shot Medical Image Segmentation (FSMIS), still encounter two primary obstacles. First, there exists huge appearance discrepancy between support and query images, which hinders the knowledge transferring and adversely affects segmentation performance. Second, almost all current prototype-based methods struggle to learn and optimize limited support prototypes, giving insufficient attention to query information, which makes it challenging to achieve high-quality query segmentation. Consequently, we propose a novel Query-Assisted Multiple Prototypes (QAMP) approach, where in addition to normal support prototypes, query prototypes are additionally mined leveraging high-confidence initial query predictions. Specifically, we design a Query Prior Generation (QPG) module to locate positions where query objects may belong to with high confidence. Subsequently, based on corresponding support mask and query prior, a Mask Guided Support Prototypes (MGSP) module and a Prior Guided Query Prototypes (PGQP) module are employed to generate support and query prototypes respectively, which can effectively capture underlying characteristics of the query targets. Extensive experiments and visualization on three publicly available medical image datasets demonstrate the superiority of our QAMP compared with current methods.

![image](FSMIS/QAMP.png)

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Datasets and pre-processing
Download:
1) **Abd-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Abd-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **Prostate-MRI**: [A male pelvic structure and prostate MRI dataset containing T2-weighted MR images]()

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.
Supervoxel segmentation is performed according to [Hansen et al.](https://github.com/sha168/ADNet.git) and we follow the procedure on their github repository.  

### Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./scripts/train_abd_mri.sh` 

### Inference
Run `./scripts/test_abd_mri.sh` 

### Acknowledgement
This code is based on [SSL-ALPNet](https://arxiv.org/abs/2007.09886v2) (ECCV'20) by [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git) and [ADNet](https://www.sciencedirect.com/science/article/pii/S1361841522000378) by [Hansen et al.](https://github.com/sha168/ADNet.git). 
