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