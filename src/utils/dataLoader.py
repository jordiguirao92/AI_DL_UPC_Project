from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataset import NoiseDataset

def get_data_loaders(batch_size):
    #In colab: transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256), transforms.ToTensor()])
    transform_train = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(256), transforms.ToTensor()]) #old: transform_train = transforms.Compose([transforms.ToPILImage(),transforms.RandomCrop(256), transforms.Resize(256), transforms.ToTensor()])
    transform_eval = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(256), transforms.ToTensor()]) #old: transform_eval = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(256), transforms.Resize(256), transforms.ToTensor()])
    train_set = NoiseDataset(path_to_images='../../images', mode='training', transform=transform_train) #Linux base: `./dataset/images`
    eval_set = NoiseDataset(path_to_images='../../images', mode='validation', transform=transform_eval) #Linux base: `./dataset/images`
    test_set = NoiseDataset(path_to_images='../../images', mode='testing', transform=transform_eval)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader