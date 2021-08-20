import os
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import transforms as T
from dataset import HandDataset
from utils import shuffle_loader
from models import HandSegmentationModel


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


def load_data(transform):

    train_images = []
    train_masks = []

    for i in os.listdir("dataset/training/color/"):
        train_images.append("dataset/training/color/" + i)
        train_masks.append("dataset/training/mask/" + i)

    eval_images = []
    eval_masks = []

    for i in os.listdir("dataset/evaluation/color/"):
        eval_images.append("dataset/evaluation/color/" + i)
        eval_masks.append("dataset/evaluation/mask/" + i)

    train_dataset = HandDataset(
        train_images, train_masks, transform=transform, augment=False
    )
    train_loader = shuffle_loader(train_dataset)

    eval_dataset = HandDataset(eval_images, eval_masks, transform=transform)
    eval_loader = shuffle_loader(eval_dataset)

    return train_loader, eval_loader


def evaluate(model, test_loader, args):

    model = model.eval()
    running_loss = 0.0

    for data, labels in test_loader:
        outputs = model(data)
        # print(labels.shape)
        # print(outputs.shape)
        loss = args.criterion(
            outputs, labels.view([-1, args.IMG_SIZE, args.IMG_SIZE]).float()
        )
        running_loss += loss.item()
        # print(loss)

    return running_loss / len(test_loader)


def train(args):

    # Define an optimizer and criterion
    model = HandSegmentationModel().to(args.device).float()
    criterion = nn.BCELoss()
    optimizer = args.optimizer(model.parameters(), lr=0.00001)

    train_loader, eval_loader = load_data(args.transform)

    best_accuracy = 1000000

    for epoch in range(args.epochs):

        index = 0
        running_loss = 0.0

        for data, target in train_loader:

            model.train()
            inputs = data.to(args.device)
            labels = target.to(args.device)

            # ============ Forward ============
            outputs = model(inputs)
            # print(outputs.shape)
            # print((labels==1).sum())
            # print(outputs.shape)
            loss = criterion(
                outputs, labels.view([-1, 1, args.IMG_SIZE, args.IMG_SIZE]).float()
            )

            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            index += 1
            running_loss += loss.item()
            print(
                "Epoch: ",
                epoch,
                " ",
                loss.item(),
                "Index (",
                index,
                "/",
                len(train_loader),
                ")",
                " ",
                (running_loss / index),
            )

        print("\n")
        test_loss = evaluate(model, eval_loader, args)
        print("Epoch: ", epoch, " ", (running_loss / index), " Test Loss: ", test_loss)
        print("\n")
        if test_loss <= best_accuracy:
            torch.save(model.state_dict(), "model_256_resnet.pth")
            best_accuracy = test_loss
