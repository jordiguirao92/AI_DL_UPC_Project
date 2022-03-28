# AI_DL_UPC_Project
UPC-Artificial Intelligence with Deep Learning (Ed. 2021-2022) - Project

## Objective
Information of resources that can be util for the project development.

## Google Cloud Instance
### Access
- Comand used to create ssh key: `ssh-keygen -t rsa -f ~/.ssh/aidl2022-project -C upcaigooglecloud -b 2048` Key phrase: `imagedenoising`
- Connect to the instance: `ssh -i ~/.ssh/aidl2022-project upcaigooglecloud@34.121.195.255`
- Keys for a Github repo in Google Cloud instance: https://cloud.redhat.com/blog/private-git-repositories-part-2a-repository-ssh-keys

### GPU-Cuda
- Install cuda: `apt install nvidia-cuda-toolkit`
- Install cu: ``pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html`
- Install Nvidia drivers guide: https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux
- Nvidia docker: `https://hub.docker.com/r/nvidia/cuda/tags`

### Upload Images with scp
- Be sure that we have the set up the keys in our computer to access to the instance, and we have installed scp
- We have to execute from our local computer `scp -i ~/.ssh/aidl2022-project -r ~/Downloads/Develop/AI_DL_UPC_Project/dataset/images upcaigooglecloud@34.121.195.255:/home/upcaigooglecloud` Check the ip address.
- The transfer files process will start. It will upload the full folder (-r). 
- We can connect to the instance to check the new files using:`ssh -i ~/.ssh/aidl2022-project upcaigooglecloud@34.121.195.255` Important to check de ip before to execute the command, the ip is dinamic and every time the instance is started the ip is different.

### Useful Comands
- Exportar comando miniconda: ` export PATH=~/miniconda3/bin:$PATH`
- Check GPU: `hwinfo --gfxcard --short` or `sudo lshw -C display`

## Artificial noise with Pytorch
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


## Links:
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


## Posible dataset
- Autonomous driving: https://analyticsindiamag.com/top-10-popular-datasets-for-autonomous-driving-projects/
- Imágenes microscopicas: https://github.com/yinhaoz/denoising-fluorescence
- Imáneges satelitales: https://github.com/chrieke/awesome-satellite-imagery-datasets
- Imagenet: https://www.image-net.org/
- Noised/Nonoised images: https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset
- Noised/Nonoised images: https://www.eecs.yorku.ca/~kamel/sidd/dataset.php
- Noised/Nonoised images: https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset/tree/master/CroppedImages


## Coding resources
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- https://neuraspike.com/blog/split-and-manipulate-pixels-opencv/
- https://pypi.org/project/image-slicer/
- https://www.kaggle.com/leifuer/intro-to-pytorch-loading-image-data
- https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict


## Metrics
- https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python
- https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html
- https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio

## Utilities
- https://pytorch.org/docs/stable/generated/torch.transpose.html
- https://ipython.org/
- 
```python
from IPython import embed
embed()
```
- Notebook Eva: https://colab.research.google.com/drive/195wAqapxodmv-wvbyxCID_pAQhS8DIR6?usp=sharing
- https://colab.research.google.com/drive/1VdjXeeMz0EketpNFs7jkXQmZ1CWx1iMg?usp=sharing


## Reunion Eva
- TODO QUITAR ACCURACY
- IDEAS: Unet partir de un modelo ya entrenado, backbones. Añadir un backbone y ver si gana algo con ese añadido. Como vamos cortos de imagenes, va bien.
- Tener claro como se evalua.
- Comprar con otros papers.
- Metrica: Inspection score para las gans, como lo shumanos percibimos la calidad de las imagenes. https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/#:~:text=The%20Inception%20Score%2C%20or%20IS%20for%20short%2C%20is%20an%20objective,Improved%20Techniques%20for%20Training%20GANs.%E2%80%9D
- Ruido sintético vs ruido real
- en vez de batchnorm que puede afectar a distribución del color, podemos probar capas de instance normalization (a nivel de imagen en vez de batch)
- Spectral normalization también puede ir bien (mirar normalizaciones para las GAN)
- En vez del blurring, jugar con el peso de la loss del discriminador
- probar de entrenar el discriminador cada x loops del generador
learning rates
- entrenar primero el generador, una vez tengo un resultado mas o menos. Guardamos. Cargamos y entrenamos con discriminador
- Otra cosa a probar es cambiar sigmoid por una tanh en la salida del generador (https://arxiv.org/pdf/1511.06434.pdf y https://neptune.ai/blog/pix2pix-key-model-architecture-decisions); quizás ayude con el tema de los colores