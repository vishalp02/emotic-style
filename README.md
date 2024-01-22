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
* Run the following command for each folder:
```
python Eval.py --content_dir input/content/mscoco/images/ --style_dir input/style/s_mscoco/ --output output/s_mscoco
```

<div align="center">
  <img src="docs/Style transfer.png"/>
</div>

### Emotion Recognition
* Emotion Recognition pipeline from [Emotic](https://github.com/Tandon-A/emotic/tree/master) is used for the Emotion recognition task. More details can be found in the original repository.

1. Preprocessing:
``` 
python mat2py.py --data_dir proj/data/emotic19 --generate_npy --body_image_size 128
```
* body_image_size: size of the body image to be extracted from the original image.

2. Training:
```
python main.py --mode train --data_path proj/data/emotic19/emotic_pre --experiment_path proj/debug_exp --weights places365
```
* weights: pre-trained weights for the backbone network.

3. Testing:
```
python main.py --mode test --data_path proj/ data/emotic19/emotic_pre --experiment_path proj/debug_exp
``` 
4. Inference:
``` 
python main.py --mode inference --inference_file final_inference_input.txt --experiment_path proj/debug_exp
```

## Dataset
Emotic, WikiArt and Odeuropa datasets used for this project. All the datasets can be found [here](https://zenodo.org/records/10501312).
Extract the files and put them in specified directory of the project.
* Emotic/Emotic-Style dataset should be in the following directory for the training and testing:
```
├── emotic19
│   ├── emotic
│   |    ├── ade20k
│   |    ├── emodb_small
│   |    ├── framesdb
│   |    ├── mscoco 
```
* Emotic as content and WikiArt as style dataset should be in the following directory for the style transfer:
```
├── input
│   ├── content
│   |    ├── ade20k
│   |    ├── emodb_small
│   |    ├── framesdb
│   |    ├── mscoco 
│   ├── style
│   |    ├── s_ade20k
│   |    ├── s_emodb_small
│   |    ├── s_framesdb
│   |    ├── s_mscoco 
```
## Results
All the experiment results can be found in docs folder. 
