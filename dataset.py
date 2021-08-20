import random
from torch.utils.data import Dataset
import Image
from PIL import ImageEnhance
import numpy as np
import torch
import pandas as pd


class HandDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False, IMG_SIZE=256):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.images = images
        self.labels = labels
        self.augment = augment
        self.IMG_SIZE = IMG_SIZE

    def __len__(self):

        return len(self.images)

    def img_augment(self, er, mask):

        hflip = random.randint(0, 1)
        if hflip:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            er = er.transpose(Image.FLIP_LEFT_RIGHT)

        x0 = 0 + random.randint(0, int(er.size[0] * 0.4))
        y0 = 0 + random.randint(0, int(er.size[1] * 0.4))
        x1 = er.size[0] - random.randint(0, int(er.size[0] * 0.4))
        y1 = er.size[1] - random.randint(0, int(er.size[1] * 0.4))

        brighten = random.randint(0, 1)

        if brighten:

            enhancer = ImageEnhance.Brightness(er)
            er = enhancer.enhance(random.uniform(0.5, 2.5))

        color = random.randint(0, 1)

        if color:

            converter = ImageEnhance.Color(er)
            er = converter.enhance(random.uniform(0.5, 2.5))

        sharpen = random.randint(0, 1)

        if sharpen:

            converter = ImageEnhance.Sharpness(er)
            er = converter.enhance(random.uniform(0.5, 2.5))

        contrast = random.randint(0, 1)

        if contrast:

            converter = ImageEnhance.Contrast(er)
            er = converter.enhance(random.uniform(0.5, 2.5))

        er = er.crop((x0, y0, x1, y1)).resize((self.IMG_SIZE, self.IMG_SIZE))
        mask = mask.crop(((x0, y0, x1, y1)))

        return er, mask

    def __getitem__(self, idx):

        img = (
            Image.open(self.images[idx])
            .resize((self.IMG_SIZE, self.IMG_SIZE))
            .convert("RGB")
        )
        mask = Image.open(self.labels[idx]).resize((self.IMG_SIZE, self.IMG_SIZE))

        if self.augment:
            img, mask = self.img_augment(img, mask)

        mask = Image.fromarray((np.asarray(mask) > 15).astype(np.uint8)).resize(
            (self.IMG_SIZE, self.IMG_SIZE)
        )

        return self.transform(img), torch.tensor(np.asarray(mask))


def get_label(img_name, annot):

    score_maps = []

    img_map = annot[(annot["img_name"] == img_name)]
    values = img_map["score_map"].unique()

    minimum = min(values)
    maximum = max(values)

    def vectorize(sub, axis):

        values = np.zeros((200, 200))

        sub_sub = sub[sub["score_map"] == axis]

        new_x = sub_sub["x"].item()
        new_y = sub_sub["y"].item()

        values[new_y : new_y + 30, new_x : new_x + 30] = 1

        return values

    if minimum == 0:
        score_maps.append(
            np.array([vectorize(img_map, index) for index in range(0, 21)])
        )

    if maximum > 21:
        score_maps.append(
            np.array([vectorize(img_map, index) for index in range(21, 42)])
        )

    return score_maps


class HandKeyPointDataset(Dataset):
    def __init__(self, data_path, transform=None, augment=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.data = pd.read_csv(data_path, names=["img_name", "score_map", "x", "y"])

    def __len__(self):
        return len(self.data["img_name"].unique())

    def __getitem__(self, idx):

        filepath = self.data.iloc[idx]["img_name"]
        image = Image.open(str(filepath)).convert("RGB").resize((200, 200))
        image = self.transform(image)
        score_maps = get_label(filepath, self.data)

        first = False
        second = False

        if len(score_maps) == 1:
            first = True
        if len(score_maps) == 2:
            second = True

        if first and second:
            score_maps = score_maps[random.randint(0, 1)]
        else:
            if first:
                score_maps = score_maps[0]
            else:
                score_maps = score_maps[1]
        return image, torch.tensor(score_maps)
