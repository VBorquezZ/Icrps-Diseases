import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset



# No Aug, dataset
class MaskDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# Classes (alphabetically ordered) that determine the amount of augmnentation to do.
NoAug = [33, 59, 46]
SoftAug = [34, 50, 53, 47, 56, 3, 57, 52, 10, 37, 58, 22]
HardAug = [21, 16, 17, 49, 51, 24, 12, 54, 44, 42, 36, 18, 11, 5, 2]
ExtremeAug = [15, 23, 48, 55, 45, 35, 4, 25, 41, 39, 26, 20, 40, 38, 31, 0, 6, 7, 1, 32, 30, 43, 9, 19, 28, 27, 13, 8, 14, 29]

# Pairs (group of class),(Number of duplications in augmentation)
default_classes_augs = [(NoAug,0),
                        (SoftAug,1),
                        (HardAug,2),
                        (ExtremeAug,3)]

# Dataset with augmentation
class MaskDatasetAugmented(Dataset):
    def __init__(self, csv_file, root_dir, classes_augs = default_classes_augs, transform=None, general_transforms=None, augmenting_transforms=None,):
        self.annotations = pd.read_csv(csv_file)
        self.classes_augs = classes_augs
        
        for class_type, n_augs in self.classes_augs:
            for n in range(n_augs):
                duplicated = self.annotations[self.annotations['labels'].isin(class_type)]
                self.annotations = pd.concat([self.annotations, duplicated], ignore_index=True)

        self.root_dir = root_dir
        self.general_transforms = general_transforms
        self.augmenting_transforms = augmenting_transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if y_label not in self.classes_augs[0][0] and self.augmenting_transforms:
            image = self.augmenting_transforms(image)

        elif self.general_transforms:
            image = self.general_transforms(image)

        return (image, y_label)
    

    def random_horizontal_flip(self, image):
        flip_h = torch.randint(2, (1,)).item()
        if flip_h == 0:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return image
        
    def random_vertical_flip(self, image):
        flip_v = torch.randint(2, (1,)).item()
        if flip_v == 0:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return image
    
    def random_rotation(self, image):
        rotation_angle = torch.randint(3, (1,)).item()
        if rotation_angle == 0:
            return image.rotate(90)
        elif rotation_angle == 1:
            return image.rotate(180)
        else:
            return image.rotate(270)
