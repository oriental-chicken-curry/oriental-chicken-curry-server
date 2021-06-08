from albumentations import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from albumentations.core.transforms_interface import ImageOnlyTransform


class Preprocessing(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=1.0):
        super(Preprocessing, self).__init__(always_apply, p)

    def apply(self, img, **params):
        img = cv2.fastNlMeansDenoising(img, h=3)

        # # Thinning and Skeletonization
        # kernel = np.ones((5,5),np.uint8)
        # img = cv2.erode(img,kernel,iterations = 1)
        return img


class ParameterError(Exception):
    def __init__(self):
        super().__init__('Not valid Augmentation parameter')


def get_transforms(aug_type: str, input_height, input_width):
    """
    Data augmentation 객체 생성
    Args:
        aug_type(str) : augmentation타입 지정
        input_height(int) : 재조정할 이미지 높이
        input_width(int) : 재조정할 이미지 넓이
    Returns :
        list : train, validation, test데이터 셋에 대한 transform
    """

    if aug_type == 'baseline':
        train_transform = A.Compose([
            A.Resize(input_height, input_width, p=1.0),
            ToTensorV2()
            ])
        val_transform = A.Compose([
            A.Resize(input_height, input_width, p=1.0),
            ToTensorV2()
            ])
        test_transform = A.Compose([
            A.Resize(input_height, input_width, p=1.0),
            ToTensorV2()
            ])
    elif aug_type == 'aug1':
        train_transform = A.Compose([
            A.Resize(input_height, input_width, p=1.0),
            CLAHE(clip_limit=4.0, p=0.5),
            Normalize(
                mean=(0.6162933558268724),
                std=(0.16278683017346854),
                max_pixel_value=255.0,
                p=1.0),
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.Blur(p=1.0),
                A.GaussianBlur(p=1.0)
            ], p=0.5),
            ToTensorV2()
        ])
        val_transform = A.Compose([
            A.Resize(input_height, input_width, p=1.0),
            Normalize(
                mean=(0.6162933558268724),
                std=(0.16278683017346854),
                max_pixel_value=255.0,
                p=1.0),
            ToTensorV2()
        ])
        test_transform = A.Compose([
            A.Resize(input_height, input_width, p=1.0),
            Normalize(
                mean=(0.6162933558268724),
                std=(0.16278683017346854),
                max_pixel_value=255.0,
                p=1.0),
            ToTensorV2()
        ])
    elif aug_type == "aug2":
        train_transform = A.Compose([
            A.Resize(input_height, input_width, p=1.0),
            Preprocessing(p=1.0),
            Normalize(
                mean=(0.6162933558268724),
                std=(0.16278683017346854),
                p=1.0),
            ToTensorV2()
        ])
        val_transform = A.Compose([
            A.Resize(input_height, input_width, p=1.0),
            Preprocessing(p=1.0),
            Normalize(
                mean=(0.6162933558268724),
                std=(0.16278683017346854),
                p=1.0),
            ToTensorV2()
        ])
        test_transform = A.Compose([
            A.Resize(input_height, input_width, p=1.0),
            Preprocessing(p=1.0),
            Normalize(
                mean=(0.6162933558268724),
                std=(0.16278683017346854),
                p=1.0),
            ToTensorV2()
        ])
    else:
        raise ParameterError

    return train_transform, val_transform, test_transform
