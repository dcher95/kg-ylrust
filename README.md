# kg-ylrust
Classification of crop disease (yellow rust) from hyperspectral images (HSI) using PyTorch

## Environment
Environment is created using kaggle-cpu image. Additional packages are available
-- jupyter-black

## Data
- Sentinel-2 images collected ~2021 across the US. 12 Bands are available (B10 is missing). More information regarding the data is available in `notes.txt`.

- Split into training, validation and testing. Since this is separate from the Kaggle competition, all data is sourced from the train repo from the competition dataset, and then randomly assigned to the different buckets with 20% kept separate as a holdout testing group. 
- Make sure to partition by geography. No mixing of same patch in different buckets due to possible cheating.

## Experimental Structure
Few experiments were run for educational reasons. Primarily to improve understanding of satellite imagery, neural net training, torchgeo and transformer architectures through PyTorch. Also useful to go through Karpath's tasklist when training neural nets.

Following questions were attempted to be answered:

- Does using false-color images specific for vegetation analysis outperform typical RGB based models?

- Does using a pre-trained model that contains all 13 bands of Sentinel-2 outperform ImageNet pretrained models with select bands used?

- Is a transformer more effective than ResNet's?


### Experiment 1: Does using false-color images specific for vegetation analysis outperform typical RGB based models?

Compare pre-trained ImageNet models on fine-tuned images using different bands.
 A) Typical RGB structure: [Band 4 (Red), Band 3 (Green), Band 2 (Blue)]
 B) Enhanced Vegetation Index: [Band 4 (Red), Band 2 (Blue), Band 8 (NIR)]
 <!-- C) False color Image for vegetation analysis: [Band 4 (Red), Band 3 (Green), Band 8 (NIR)] ?? -->

