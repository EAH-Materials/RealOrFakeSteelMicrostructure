import os
import csv
import pickle
import torch

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class ImageFileFolder(ImageFolder):
    def __init__(self, root, transform = None, target_transform = None, loader = ..., is_valid_file = None):
        super(ImageFileFolder, self).__init__(root=root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.imgs[index][0]

def discriminate_images_from_path(discriminator, base_path, save_path):
    os.makedirs(base_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    torch.manual_seed(0)
    folder = ImageFileFolder(save_path, transform)
    dataloader = DataLoader(folder, batch_size=32, shuffle=True, drop_last=True)

    csv_file_path = os.path.join(base_path, "discriminations.csv")

    file_exists = os.path.isfile(csv_file_path)
    if file_exists:
        os.remove(csv_file_path)
        file_exists = False
        
    results = torch.empty(0).cuda()

    with open(csv_file_path, mode="a", newline="") as csv_file:
        fieldnames = ["file", "discrimination value", "sigmoid", "classification"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()

        for img, file in dataloader:
            with torch.no_grad():
                result = discriminator(img.cuda(), c=None) # Minimize logits for generated images and Maximize logits for real images.
            results = torch.cat((results, torch.flatten(result)))

            writer.writerows([{
                "file" : file[i],
                "discrimination value" : f"{x.item():.2f}".replace(".", ","), 
                "sigmoid": f"{torch.sigmoid(x).item():.2f}".replace(".", ","), 
                "classification": "Fake" if x.detach().cpu().item() < 0. else "Real",
            } for i, x in enumerate(result)])

def main():
    model_name = "network-snapshot-005000.pkl"
    base_path = f"model"
    model_path = f"{base_path}/{model_name}"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    discriminator = model['D'].cuda()
    discriminator.eval()

    save_path = "evaluate/discriminator"
    
    real_imgs_path = f"data/real"
    real_save_path = os.path.join(save_path, "real")
    discriminate_images_from_path(discriminator, real_save_path, real_imgs_path)

    real_test_imgs_path = f"data/real_test"
    real_test_save_path = os.path.join(save_path, "real_test")
    discriminate_images_from_path(discriminator, real_test_save_path, real_test_imgs_path)

    fractal_path = f"data/fractalgen"
    fractalgen_save_path = os.path.join(save_path, "fractalgen")
    discriminate_images_from_path(discriminator, fractalgen_save_path, fractal_path)

    stylegan_path = f"data/stylegan"
    stylegan_save_path = os.path.join(save_path, "stylegan")
    discriminate_images_from_path(discriminator, stylegan_save_path, stylegan_path)

if __name__ == "__main__":
    main()