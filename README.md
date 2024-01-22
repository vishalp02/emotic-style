# Emotion Recognition in Smell-Related Artworks

This project shows experiments in emotion recognition in smell-related artworks, creating a 
stylized version of the EMOTIC dataset to overcome the domain gap between photographic and artistic 
imagery and approximate the distribution of our artistic target domain. This work explores whether 
person-level emotion recognition in a set of smell-related artworks is technically feasible. 
Ultimately, the aim is to link the extracted emotions to olfactory references that have been 
extracted from the images previously.

<div align="center">
  <img src="docs/inference example.jpg" width="300"/>
</div>

## Setup

### Style Transfer
* Style transfer method from [IEContraAST](https://github.com/HalbertCH/IEContraAST/tree/main) is used for the Style transferring task on emotic dataset.
<div align="center">
  <img src="docs/Style transfer.png"/>
</div>

### Emotion Recognition
* Emotion Recognition pipeline from [Emotic](https://github.com/Tandon-A/emotic/tree/master) is used for the Emotion recognition task.


## Dataset
Emotic, WikiArt and Odeuropa datasets used for this project. All the datasets can be found [here](https://zenodo.org/records/10501312)

## Reluts
All the experiment results can be found in docs folder. 
