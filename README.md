# Mutlimodality for skin lesions classification

Many people worldwide suffer from skin diseases. For diagnosis, physicians often combine multiple information sources. These include, for instance, clinical images, microscopic images and meta-data such as the age and gender of the patient. Deep learning algorithms can support the classification of skin lesions by fusing all the information together and evaluating it. Several such algorithms are already being developed. However, to apply these learning algorithms in the clinic, they need to be further improved to achieve higher diagnostic accuracy.

## Dataset

Download the [ISIC 2020 dataset](https://www.kaggle.com/nroman/melanoma-external-malignant-256).
In the directory you will find:
- metadata as `train.csv` and `test.csv`,
- images for train and test subsets.

## Training multimodal EfficientNet

In its most basic form, training new networks boils down to:

```.bash
python train.py --save-name efficientnetb2_256_20ep --data-dir ./melanoma_external_256/ --image-size 256 \
                --n-epochs 20 --enet-type efficientnet-b2 --CUDA_VISIBLE_DEVICES 0
python train.py --save-name efficientnetb2_256_20ep_meta --data-dir ./melanoma_external_256/ --image-size 256 \
                --n-epochs 20 --enet-type efficientnet-b2 --CUDA_VISIBLE_DEVICES 0 --use-meta
```

The first command is uses only images during training; for the second one additional addition of avalilable metadata is done.

## Training multilabel classifier

We created a model with multiple binary heads to distinguish between different type of biases, such as ruler and black frame.
To use the model check `multi_classification.py` script.

```.bash
python multi_classification.py --img_path ./melanoma_external_256/train/train --ann_path gans_biases.csv \
                               --mode train --model_path multiclasificator_efficientnet-b2_GAN.pth

python multi_classification.py --img_path ./melanoma_external_256/train/train --ann_path gans_biases.csv \
                               --mode val --model_path multiclasificator_efficientnet-b2_GAN.pth

python multi_classification.py --img_path ./melanoma_external_256/test/test --mode test \
                               --model_path multiclasificator_efficientnet-b2_GAN.pth --save_path annotations.csv
```

We can distinguish between 3 modes:
- train: we need here provided annotations of biases for each image,
- val: we need here provided annotations of biases for each image and trained model,
- test: we need trained model to create new annotations for unseen images.

## Creditentials

This project based on code produced by [1st place on liderboard for Kaggle ISIC 2020 competition](https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard).

More details can be found here:

https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution

https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412

http://arxiv.org/abs/2010.05351

## Acknowledgements

The project was developed during the first rotation of the [Eye for AI Program](https://www.ai.se/en/eyeforai) at the AI Competence Center of [Sahlgrenska University Hospital](https://www.sahlgrenska.se/en/). Eye for AI initiative is a global program focused on bringing more international talents into the Swedish AI landscape.
