from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import transforms as T
import time

from models import KeyPointDetectionModel
from dataset import HandKeyPointDataset
from utils import shuffle_loader

Image.MAX_IMAGE_PIXELS = None


class Args:
    def __init__(self):
        self.random_seed = 1
        self.device = "cpu"
        self.IMG_SIZE = 256
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        self.transforms = T.Compose([T.ToTensor(), normalize])
        self.optimizer = optim.Adam
        self.loss_fn = nn.BCELoss()
        self.epochs = 10


def load_data(args):

    train_dataset = HandKeyPointDataset(
        "Pre-processed/training/data.csv", transform=args.transform, augment=False
    )
    train_loader = shuffle_loader(train_dataset)

    eval_dataset = HandKeyPointDataset(
        "Pre-processed/evaluation/data.csv", transform=args.transform, augment=False
    )
    eval_loader = shuffle_loader(eval_dataset)

    return train_loader, eval_loader


def evaluate(model, test_loader, criterion):

    model = model.eval()
    running_loss = 0.0

    for data, labels in test_loader:
        outputs = model(data)
        loss = criterion(outputs, labels.float())
        running_loss += loss.item()

    return running_loss / len(test_loader)


def train(args):

    train_loader, eval_loader = load_data(args)

    model = KeyPointDetectionModel().to(args.device).float()

    criterion = args.criterion
    optimizer = args.optimizer

    best_accuracy = 1000000

    for epoch in range(args.epochs):

        index = 0
        running_loss = 0.0

        start_time = time.time()

        for data, target in train_loader:
            preds = model(data)
            loss = criterion(preds, target.float())
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index += 1
            print("Index: ", index, "/", len(train_loader))

        end_time = time.time()

        print(f"Epoch: {epoch} and Running Loss:{running_loss/index}")
        test_loss = evaluate(model, eval_loader, criterion)
        print(f"Evaluation: {test_loss} and Time: {end_time-start_time}s")
        if test_loss <= best_accuracy:
            torch.save(model.state_dict(), "model_200_30_all_keypoint.pth")
            best_accuracy = test_loss
        print("\n")
