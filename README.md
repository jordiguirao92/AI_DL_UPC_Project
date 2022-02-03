# AI_DL_UPC_Project
UPC-Artificial Intelligence with Deep Learning (Ed. 2021-2022)-Project

## Init project

#### Create conda enviroment
`conda create --name aidl-project python=3.8`
#### Activate conda enviroment
`conda activate aidl-project`

To desactivate:
`conda deactivate`

#### Install dependencies
`pip install -r requirements.txt`




## Enlaces interés sore image denoising
- https://medium.com/analytics-vidhya/image-denoising-using-deep-learning-dc2b19a3fd54
- https://towardsai.net/p/deep-learning/image-de-noising-using-deep-learning

## Implementacion de noise con pytorch:
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

## Posible datasets
- Autonomous driving: https://analyticsindiamag.com/top-10-popular-datasets-for-autonomous-driving-projects/
- Imágenes microscopicas: https://github.com/yinhaoz/denoising-fluorescence
- Imáneges satelitales: https://github.com/chrieke/awesome-satellite-imagery-datasets
- Imagenet: https://www.image-net.org/

## Posible modelo:
- Normalmente una arquitectura encoder-decoder tipo Unet suele funcionar muy bien.

## Como entrenar:
- Optimizar L1 loss (o cualquier metrica que calcule la diferencia entre la predicción "limpia" y la imagen limpia de verdad)
- GANs son bastante populares: L1 loss para el generador + la loss del discriminador multiplicada por algun factor de escala


- Preparar justificacion de porque no usamor ruido artificial
- PixtoPix y UNET (Generador, quitar el ruido) Quitar ruido con UNET
- Tener datos descargados
- Implementar dataset y dataloader (Imagen y ground truth en los path)
- Dataset: https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset

1. Script para descargar las imagenes https://www.eecs.yorku.ca/~kamel/sidd/dataset.php
2. Tener imagenes


# Useful links:
- https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset/tree/master/CroppedImages
- https://github.com/jaxony/unet-pytorch/blob/master/model.py
- https://www.pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
- https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
- https://github.com/yinhaoz/denoising-fluorescence/blob/d83cad96b205793da0f62f6dc8094799d61929e6/denoising/train_n2n.py#L15
- https://github.com/chintan1995/Image-Denoising-using-Deep-Learning/blob/main/Models/(Baseline)_REDNet_256x256.ipynb
- https://olaleyeayoola.medium.com/removing-noise-from-imag es-using-a-cnn-model-in-pytorch-part-2-306be83afa46 (useful)

- https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f
- https://colab.research.google.com/github/ashishpatel26/Awesome-Pytorch-Tutorials/blob/main/13.%20Pytorch%20Image%20Data%20for%20Deep%20learning%20Data%20Augmentation.ipynb#scrollTo=3Wmr9mKSVufp
- https://www.pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/
- https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec
- https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2
- https://discuss.pytorch.org/t/shows-image-with-specific-index-from-mnist-dataset/29406
