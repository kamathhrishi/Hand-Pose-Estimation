import os
from PIL import Image
import scipy.io


def segment_hands(inte, img, index1, index2, mat, mode, fp):

    sub_index = 1

    if mode == "evaluation":
        sub_index = 2

    hands = []
    score_maps = []

    x_minimum = float("inf")
    x_maximum = float("-inf")

    y_minimum = float("inf")
    y_maximum = float("-inf")

    for i in range(index1, index2):
        x = mat["frame" + str(int(inte[:-4]))][0][0][sub_index][i][0]
        y = mat["frame" + str(int(inte[:-4]))][0][0][sub_index][i][1]

        if x < x_minimum:
            x_minimum = x
        if x > x_maximum:
            x_maximum = x

        if y < y_minimum:
            y_minimum = y

        if y > y_maximum:
            y_maximum = y

    width, height = img.size

    if (x_minimum >= 0 and y_maximum >= 0) and (
        x_minimum <= width and y_maximum <= height
    ):
        y_minimum -= 20
        x_minimum -= 20
        x_maximum += 20
        y_maximum += 20
        new_img = img.crop((x_minimum, y_minimum, x_maximum, y_maximum))
        hands.append(new_img.resize((200, 200)).copy())

        hand_dir = None

        if index2 > 21:
            hand_dir = 2
        else:
            hand_dir = 1

        filepath = f"Pre-processed/{mode}/raw/{hand_dir}_{inte}"
        new_img.save(filepath)
        try:
            f = open(filepath)
            f.close()
        except FileNotFoundError:
            print(f"File {filepath} does not exist")

        for i in range(index1, index2):
            x = mat["frame" + str(int(inte[:-4]))][0][0][sub_index][i][0]
            y = mat["frame" + str(int(inte[:-4]))][0][0][sub_index][i][1]
            new_y = int((y - y_minimum) * (200 / new_img.size[1]))
            new_x = int((x - x_minimum) * (200 / new_img.size[0]))
            fp.write(f"{filepath}, {i} , {new_x} , {new_y} ")
            fp.write("\n")

    return hands, score_maps


def func(path, annot_path, mode):

    fp = open(f"Pre-processed/{mode}/data.csv", "w", newline="")
    mat = scipy.io.loadmat(annot_path)
    hand_imgs = []
    score_maps = []
    index = 0

    for inte in os.listdir(path):
        img = Image.open(path + str(inte)).convert("RGB")
        imgs, scores = segment_hands(inte, img, 0, 21, mat, mode, fp)
        img = Image.open(path + str(inte)).convert("RGB")
        imgs, scores = segment_hands(inte, img, 21, 42, mat, mode, fp)
        index += 1
        fp.close()

    return hand_imgs, score_maps


train_hand_imgs, train_score_maps = func(
    "dataset/training/color/", "dataset/training/anno_training.mat", "training"
)
test_hand_imgs, test_score_maps = func(
    "dataset/evaluation/color/", "dataset/evaluation/anno_evaluation.mat", "evaluation"
)
