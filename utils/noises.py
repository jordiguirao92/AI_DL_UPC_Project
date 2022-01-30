import torch
from skimage.util import random_noise

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# Posible to need the tensor as numpy array
# https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise
# Convert tensor to numpy array.
# x.detach().to("cpu").numpy()
class AddGaussianNoiseSkImage(object):
    def __init__(self, mean=0., var=0.05):
        self.mean = mean
        self.var = var
        
    def __call__(self, tensor):
        return torch.tensor(random_noise(tensor, mode='gaussian', mean=self.mean, var=self.var, clip=True))
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, var={1})'.format(self.mean, self.var)


class AddSaltNoise(object):
    def __init__(self, amount=0.05):
        self.amount = amount
        
    def __call__(self, tensor):
        return torch.tensor(random_noise(tensor, mode='salt', amount= self.amount))
    
    def __repr__(self):
        return self.__class__.__name__ + '(amount={0})'.format(self.amount)


class AddSpeckleNoise(object):
    def __init__(self, mean=0., var=0.05):
        self.mean = mean
        self.var = var
        
    def __call__(self, tensor):
        return torch.tensor(random_noise(tensor, mode='speckle', mean=self.mean, var=self.var, clip=True))
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, var={1})'.format(self.mean, self.var)


class AddSaltPeperNoise(object):
    def __init__(self, sp=0.5):
        self.sp = sp
        
    def __call__(self, tensor):
        return torch.tensor(random_noise(tensor, mode='s&p', salt_vs_pepper=self.sp, clip=True))
    
    def __repr__(self):
        return self.__class__.__name__ + '(sp={0})'.format(self.sp)


class AddPoissonNoise(object):
    def __init__(self, mode="poisson"):
        self.mode = mode
        
    def __call__(self, tensor):
        return torch.tensor(random_noise(tensor, mode='poisson', clip=True))
    
    def __repr__(self):
        return self.__class__.__name__ + '(mode={0})'.format(self.mode)



# https://discuss.pytorch.org/t/how-to-add-noise-to-inputs-as-a-function-of-input/54839/2
# https://colab.research.google.com/github/ashishpatel26/Awesome-Pytorch-Tutorials/blob/main/13.%20Pytorch%20Image%20Data%20for%20Deep%20learning%20Data%20Augmentation.ipynb#scrollTo=3Wmr9mKSVufp
# https://debuggercafe.com/adding-noise-to-image-data-for-deep-learning-data-augmentation/
# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
# ttps://discuss.pytorch.org/t/how-to-add-noise-to-inputs-as-a-function-of-input/54839/2
