# Automated Lesion Segmentation in Whole-body FDG-PET/CT: Solution to autoPET challenge

## Introduction

The solution is based on the well-known [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). We make three modifications:

- using more data augmentations
- increasing the number of epochs to 1200
- DiceTopK loss function

The final model is the ensemble of 13 cross-validation models without testing-time augmentation. 

## Training

We train three groups cross-validation models

- Baseline model

```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 taskid fold # fold in [0,1,2,3,4]
```

- more data agumentation

```bash
nnUNet_train 3d_fullres nnUNetTrainerV2_DA5 taskid fold # fold in [0,1,2,3,4]
```

- DiceTopK loss

```bash
nnUNet_train 3d_fullres nnUNetTrainerV2_DA5_DiceTopK10 taskid fold # fold in [0,1,2,3,4]
```



## Inference

Donwload checkpoints: https://pan.baidu.com/s/1C3TaO0IVMXsBdSjAF-HMSg pw:4494 



Run

```bash
docker build -t autopet_fighttumor .
```





## Acknowledgements

- autoPET organizers: https://autopet.grand-challenge.org/
- nnUNet developers: https://github.com/MIC-DKFZ/nnUNet



