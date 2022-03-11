# AI_DL_UPC_Project
UPC-Artificial Intelligence with Deep Learning (Ed. 2021-2022) - Project
## Objective
Create a GANs for image denoising. First of all, we tried to prepare the images with artificial noise, but after analyzing it, we have decided to search for a dataset with real noised images, to do a better real approach in the model. We want to apply the model in real cases and not in artificial noisy cases. 

## Generator model - Unet
We are using Unet as a generator model in de GANs.
The U-Net architecture follows an encoder-decoder cascade structure, where the encoder gradually compresses information into a lower-dimensional representation. Then the decoder decodes this information back to the original image dimension. Owing to this, the architecture gets an overall U-shape, which leads to the name U-Net.

[![Unet](https://929687.smushcdn.com/2407837/wp-content/uploads/2021/11/u-net_training_image_segmentation_models_in_pytorch_header.png?lossy=1&strip=1&webp=1)](https://929687.smushcdn.com/2407837/wp-content/uploads/2021/11/u-net_training_image_segmentation_models_in_pytorch_header.png?lossy=1&strip=1&webp=1)

## Metricas para image denoising
- Peak Signal to Noise Ratio
- Structural Similarity Index

## Init project
Use the following steps to start the project.

### Set up project
Use the following steps to create the python enviroment

- Create conda enviroment:

    `conda create --name aidl-project python=3.8`

- Activate conda enviroment:

    `conda activate aidl-project`

- Deactivate conda enviroment:

    `conda deactivate`

- Install dependencies:

    `pip install -r requirements.txt`

### Start training
Use the following steps to start the model training

- Generator training:

    `cd src`
    `python generator-training.py`

You can configure some training paramaters as args:
- `--loss` You can select between `l1` or `mse`. By default is `l1`
- `--lr` Indicate the learning rate. By default is `0.001`
- `--batch_size` Indicate the batch size. By default is `4`
- `--epochs` Indicate the number of training epochs. By default is `10`

### Check logs in localhost 6006

`tensorboard --logdir=logs`

### Google Cloud
- Comand used to create ssh key: `ssh-keygen -t rsa -f ~/.ssh/aidl2022-project -C upcaigooglecloud -b 2048`
- Key phrase: `imagedenoising`
- Access: `ssh -i ~/.ssh/aidl2022-project upcaigooglecloud@34.68.201.52`
- Install cuda: `apt install nvidia-cuda-toolkit`
- Install cu: `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
- Check GPU: `hwinfo --gfxcard --short` or `sudo lshw -C display`
- Process to isntall GPU dependencies: https://www.mvps.net/docs/install-nvidia-drivers-ubuntu-18-04-lts-bionic-beaver-linux/

## Useful Data for the project
- Normalizar la GAN entre -1,1
- https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
### Artificial noise with Pytorch
```python
from torch import nn

class noiseLayer_normal(nn.Module):
    def __init__(self, noise_percentage, mean=0, std=0.2):
        super(noiseLayer_normal, self).__init__()
        self.n_scale = noise_percentage
        self.mean=mean
        self.std=std

    def forward(self, x):
        if self.training:
            device = x.get_device()
            if device>0:
                noise_tensor = torch.normal(self.mean, self.std, size=x.size()).to(x.get_device()) 
            else:
                noise_tensor = torch.normal(self.mean, self.std, size=x.size())
            x = x + noise_tensor * self.n_scale
        
            mask_high = (x > 1.0)
            mask_neg = (x < 0.0)
            x[mask_high] = 1
            x[mask_neg] = 0

        return x
```
### Links:
- https://github.com/jaxony/unet-pytorch/blob/master/model.py
- https://www.pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
- https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
- https://github.com/yinhaoz/denoising-fluorescence/blob/d83cad96b205793da0f62f6dc8094799d61929e6/denoising/train_n2n.py#L15
- https://github.com/chintan1995/Image-Denoising-using-Deep-Learning/blob/main/Models/(Baseline)_REDNet_256x256.ipynb
- https://olaleyeayoola.medium.com/removing-noise-from-images-using-a-cnn-model-in-pytorch-part-2-306be83afa46 (useful)
- https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f
- https://colab.research.google.com/github/ashishpatel26/Awesome-Pytorch-Tutorials/blob/main/13.%20Pytorch%20Image%20Data%20for%20Deep%20learning%20Data%20Augmentation.ipynb#scrollTo=3Wmr9mKSVufp
- https://www.pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/
- https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec
- https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2
- https://discuss.pytorch.org/t/shows-image-with-specific-index-from-mnist-dataset/29406
- https://medium.com/analytics-vidhya/image-denoising-using-deep-learning-dc2b19a3fd54
- https://towardsai.net/p/deep-learning/image-de-noising-using-deep-learning

### Posible dataset
- Autonomous driving: https://analyticsindiamag.com/top-10-popular-datasets-for-autonomous-driving-projects/
- Imágenes microscopicas: https://github.com/yinhaoz/denoising-fluorescence
- Imáneges satelitales: https://github.com/chrieke/awesome-satellite-imagery-datasets
- Imagenet: https://www.image-net.org/
- Noised/Nonoised images: https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset
- Noised/Nonoised images: https://www.eecs.yorku.ca/~kamel/sidd/dataset.php
- Noised/Nonoised images: https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset/tree/master/CroppedImages

### Coding resources
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- https://neuraspike.com/blog/split-and-manipulate-pixels-opencv/
- https://pypi.org/project/image-slicer/
- https://www.kaggle.com/leifuer/intro-to-pytorch-loading-image-data
- - https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict
### Metrics
- https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python
- https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html
- https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio

### Utilidades
- https://pytorch.org/docs/stable/generated/torch.transpose.html
- https://ipython.org/
- 
```python
from IPython import embed
embed()
```
- Notebook Eva: https://colab.research.google.com/drive/195wAqapxodmv-wvbyxCID_pAQhS8DIR6?usp=sharing
- https://colab.research.google.com/drive/1VdjXeeMz0EketpNFs7jkXQmZ1CWx1iMg?usp=sharing



# Reunion Eva
- TODO QUITAR ACCURACY
- IDEAS: Unet partir de un modelo ya entrenado, backbones. Añadir un backbone y ver si gana algo con ese añadido. Como vamos cortos de imagenes, va bien.
- Tener claro como se evalua.
- Comprar con otros papers.
- Metrica: Inspection score para las gans, como lo shumanos percibimos la calidad de las imagenes. https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/#:~:text=The%20Inception%20Score%2C%20or%20IS%20for%20short%2C%20is%20an%20objective,Improved%20Techniques%20for%20Training%20GANs.%E2%80%9D
- Ruido sintético vs ruido real