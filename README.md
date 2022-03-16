# AI_DL_UPC_Project
UPC-Artificial Intelligence with Deep Learning (Ed. 2021-2022) - Project
## Objective
Create a GANs for image denoising. First of all, we tried to prepare the images with artificial noise, but after analyzing it, we have decided to search for a dataset with real noised images, to do a better real approach in the model. We want to apply the model in real cases and not in artificial noisy cases. 

## Init project
Use the following steps to start the project.
### Prepare dataset steps
- In the folder `/dataset/original` paste the original images
- In orde to rename original images, slice, move and rename sliced images run the command: `python ./src/utils/prepare_dataset.py`

### Using Miniconda
- Create conda enviroment, only one time, if it has not been created yet: `conda create --name aidl-project python=3.8`
- Activate conda enviroment: `conda activate aidl-project`
- Deactivate conda enviroment: `conda deactivate`
- Install dependencies: `pip install -r requirements.txt`

Use the following command to start the model training

`python ./src/main.py`

You can configure some training paramaters as args:
- `--net` You can select between `generator` or `gan`. By default is `gan`
- `--loss` You can select between `l1` or `mse`. By default is `l1`
- `--lr` Indicate the learning rate. By default is `0.001`
- `--batch_size` Indicate the batch size. By default is `4`
- `--epochs` Indicate the number of training epochs. By default is `10`

### Using Docker
- Be sure that you have docker and docker-compose installed
- Run the container with the command: `make service-up`
- Down the container with the command: `make service-down`


## Tensorboard Logs
In order to check the logs in tensorboard, run the following command:

`tensorboard --logdir=logs`


## Generator model - Unet
We are using Unet as a generator model in de GANs.
The U-Net architecture follows an encoder-decoder cascade structure, where the encoder gradually compresses information into a lower-dimensional representation. Then the decoder decodes this information back to the original image dimension. Owing to this, the architecture gets an overall U-shape, which leads to the name U-Net.

[![Unet](https://929687.smushcdn.com/2407837/wp-content/uploads/2021/11/u-net_training_image_segmentation_models_in_pytorch_header.png?lossy=1&strip=1&webp=1)](https://929687.smushcdn.com/2407837/wp-content/uploads/2021/11/u-net_training_image_segmentation_models_in_pytorch_header.png?lossy=1&strip=1&webp=1)

## Metricas para image denoising
- Peak Signal to Noise Ratio
- Structural Similarity Index
