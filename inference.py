import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models, transforms as T
import cv2
import math
from PIL import ImageDraw

from models import KeyPointDetectionModel, HandSegmentationModel

IMG_SIZE = 256


def determine_max(rect):

    x_min = math.inf
    x_max = 0

    y_min = math.inf
    y_max = 0

    for i in rect:

        if i[0] < x_min:

            x_min = i[0]

        if i[0] > x_max:

            x_max = i[0]

        if i[1] < y_min:

            y_min = i[1]

        if i[1] > y_max:

            y_max = i[1]

    # return ((x_min,y_min),(x_max,y_max))
    return ((x_min, y_min), (x_max, y_max))


def find_contour(contours):

    max_index = None
    max_area = None

    index = 0

    for cnt in contours:
        coordinates = determine_max(cnt.reshape([-1, 2]))
        area = (coordinates[0][0] - coordinates[0][1]) * (
            coordinates[1][0] - coordinates[1][1]
        )
        if max_area is None:
            max_area = area
            max_index = index
        elif area > max_area:
            max_area = area
            max_index = index
        index += 1

    return max_index


def detect_keypoints(predictions):
    ige = Image.new("RGB", (200, 200), "black")
    draw = ImageDraw.Draw(ige)
    for i in range(0, 21):
        print("\n")
        print("\n")
        contours1, hierarchy = cv2.findContours(
            predictions[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        cnt1 = None

        if len(contours1):

            cnt_index = find_contour(contours1)
            if cnt_index is not None:
                cnt1 = contours1[cnt_index]

            if cnt1 is not None:
                coordinates = determine_max(cnt1.reshape([-1, 2]))
                draw.ellipse(
                    (
                        coordinates[0][0],
                        coordinates[0][1],
                        coordinates[0][0] + 12,
                        coordinates[0][1] + 12,
                    ),
                    fill=(255, 255, 255),
                )

                contours2, hierarchy = cv2.findContours(
                    predictions[(i + 1) % 21], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                cnt2 = None

                if len(contours2):
                    cnt_index = find_contour(contours2)
                    if cnt_index is not None:
                        cnt2 = contours2[cnt_index]

                if (cnt1 is not None) and (cnt2 is not None):
                    coordinates1 = determine_max(cnt1.reshape([-1, 2]))
                    coordinates2 = determine_max(cnt2.reshape([-1, 2]))
                    draw.line(
                        (
                            coordinates1[0][0],
                            coordinates1[0][1],
                            coordinates2[0][0] + 5,
                            coordinates2[0][1] + 5,
                        ),
                        fill=(255, 255, 255),
                    )
    plt.imshow(ige)
    plt.show()


def perform_inference():

    from torchvision import transforms

    model = HandSegmentationModel().eval().float()
    model.load_state_dict(torch.load("model_256_resnet.pth"))

    key_model = KeyPointDetectionModel()
    key_model.load_state_dict(torch.load("model_200_keypoint_30.pth"))
    key_model.eval()

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    transform = T.Compose([T.ToTensor(), normalize])

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = np.asarray(Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE)))
        out = model(transform(frame).view([1, 3, IMG_SIZE, IMG_SIZE]).float())
        print(torch.max(out))
        out = (out.view([IMG_SIZE, IMG_SIZE]).detach().numpy() >= 0.5).astype(np.uint8)

        plt.imshow(out)
        plt.show()

        contours, hierarchy = cv2.findContours(
            out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        for cnt in contours:

            coordinates = determine_max(cnt.reshape([-1, 2]))

            area = (coordinates[1][0] - coordinates[0][0]) * (
                coordinates[1][1] - coordinates[0][1]
            )
            print(area)
            offset = 30

            if area >= 10:
                new_frame = Image.fromarray(frame)
                new_frame = new_frame.crop(
                    (
                        coordinates[0][0] - offset,
                        coordinates[0][1] - offset,
                        coordinates[1][0] + offset,
                        coordinates[1][1] + offset,
                    )
                )

                new_frame = new_frame.resize((200, 200))
                preds = (
                    key_model(transform(new_frame).view([-1, 3, 200, 200]))
                    .detach()
                    .numpy()
                    .reshape((21, 200, 200))
                )
                preds = np.asarray(preds.reshape((21, 200, 200)) > 0.5).astype(np.uint8)

                contours1, hierarchy = cv2.findContours(
                    preds[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                plt.imshow(new_frame)
                plt.show()

                detect_keypoints(preds)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


perform_inference()
