## Start the project
Use the following steps to start the project.

### Prepare dataset steps

### Option 1 - Dowload the prepared images {#option1}
- Dowload the dataset from the following link: https://mega.nz/folder/Erg2EYiS#kVS2e-nAGL3etSqgeZ4lbw The images folder container the prepared dataset with the correct image name, also contain the validation, training and testing txt files.
- Copy the folder `images` in the `./dataset/`.
- Now, you have your dataset prepared.


### Option 2 - Dowload original images {#option1}
- Dowload the original images from the following link: https://www.eecs.yorku.ca/~kamel/sidd/dataset.php If you use this way, you need to download the images in different parts. We recomment to use the [option1](#my-anchor).
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

## Posible models
- model_1: Generator without Batchnorm and Sigmoid `python ./src/main.py --net generator --generator_norm False`
- model_2: Generator with Batchnorm and Sigmoid `python ./src/main.py --net generator --generator_norm batch --generator_last sigmoid`
- model_3: Generator with Batchnorm and Tanh `python ./src/main.py --net generator --generator_norm batch --generator_last tanh`
- model_4: Generator with Instance and Tanh `python ./src/main.py --net generator --generator_norm instance --generator_last tanh`
- model_5: Generator with Spectral and Tanh `python ./src/main.py --net generator --generator_norm spectral --generator_last tanh`
- model_6: Generator with Instance and Tanh + Discriminator with LeakyRelu and Sigmoid, D=2.5 `python ./src/main.py --net gan --generator_norm instance --generator_last tanh --discriminator_last sigmoid --discriminator_activation leakyRelu --d_weight 2.5` 
- model_7: Generator with Instance and Tanh + Discriminator with LeakyRelu and Tanh, D=40 `python ./src/main.py --net gan --generator_norm instance --generator_last tanh --discriminator_last tanh --discriminator_activation leakyRelu --d_weight 40` 