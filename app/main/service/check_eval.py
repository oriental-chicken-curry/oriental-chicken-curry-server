import os
import sys
sys.path.append('D:/workspace2/oriental-chicken-curry-server')
import csv
import cv2
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from math import log
from tqdm import tqdm
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

from app.main.service.utils.flags import Flags
from app.main.service.data.dataset import START, END, PAD
from app.main.service.utils.checkpoint import load_checkpoint
from app.main.service.networks.SATRN import SATRN
# from metrics import word_error_rate, sentence_acc


class Denoising(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
    
    def apply(self, img, **params):
        return cv2.fastNlMeansDenoising(img, h=5)


def id_to_string(tokens, data_loader, do_eval=0):
    result = []
    if do_eval:
        eos_id = data_loader.dataset.token_to_id["<EOS>"]
        special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
                       data_loader.dataset.token_to_id["<EOS>"]]

    for example in tokens:
        string = ""
        # cou = 0 # 추가
        # st = 0 # 추가
        ### bracket post processing
        if do_eval:
            for token in example:
                token = token.item()
                # # { } 개수 맞추기
                # # ------ 추가 -------
                # if token == 224:
                #     cou += 1 
                #     st += 1 
                # elif token == 213:
                #     cou -= 1
                # elif st > 0:
                #     for i in range(st):
                #         string += "{" + " "

                #     st = 0

                # if cou == -1:
                #     cou = 0
                #     continue

                # if token == 213 and st > 0:
                #     st -= 1
                #     continue
                # # ------ 추가 -------
                if token not in special_ids:
                    # if token != -1 and token != 224:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
                elif token == eos_id:
                    break
                # if token not in special_ids:
                #     if token != -1:
                #         string += data_loader.dataset.id_to_token[token] + " "
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "

        result.append(string)
    return result


def encode_truth(truth, token_to_id):
    """ ground truth의 latex문구를 파싱하여 id로 변환

    Args:
        truth(str) : gt latex
        token_to_id(dict) : token의 아이디 정보가 담겨있는 딕셔너리

    Returns:
        list : 토큰들의 아이디 정보
    """
    truth_tokens = truth.split()
    for token in truth_tokens:
        if token not in token_to_id:
            raise Exception("Truth contains unknown token")
    truth_tokens = [token_to_id[x] for x in truth_tokens]
    if '' in truth_tokens: truth_tokens.remove('')
    return truth_tokens


def collate_eval_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "file_path":[d["file_path"] for d in data],
        "image": torch.stack([data[0]["image"], data[0]["image2"], data[0]["image3"]], dim=0),        
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

class LoadEvalDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        token_to_id,
        id_to_token,
        crop=False,
        transform=None,
        rgb=3,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadEvalDataset, self).__init__()
        self.crop = crop
        self.rgb = rgb
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.transform = transform
        self.data = [
            {
                "path": p,
                "file_path":p1,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, p1, truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        if self.rgb == 3:
            image = image.convert("RGB")
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image_np = np.array(image)
            image2 = cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image3 = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE) 
            image = self.transform(image=image_np)['image']
            image2 = self.transform(image=image2)['image']
            image3 = self.transform(image=image3)['image']
            # image = self.transform(image)

        return {"path": item["path"], "file_path":item["file_path"], "truth": item["truth"], "image": image, "image2": image2, "image3": image3}

def main(parser):
    is_cuda = torch.cuda.is_available()
    print(f"Cuda is available? : {is_cuda}")
    checkpoint = load_checkpoint(parser.checkpoint, cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        )
    print(options.input_size.height)

    transformed = A.Compose([
        A.Resize(options.input_size.height, options.input_size.width),
        Denoising(p=1),
        A.Normalize(mean=(0.6156), std=(0.1669)),
        ToTensorV2()
    ])

    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence
    # root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    # test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    test_data = [[x[0], x[0].split('/')[-1], dummy_gt] for x in data]
    test_dataset = LoadEvalDataset(
        test_data, checkpoint["token_to_id"], checkpoint["id_to_token"], crop=False, transform=transformed,
        rgb=options.data.rgb
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_eval_batch,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_eval_batch,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_eval_batch,
    )

    print(
        "[+] Data\n",
        "The number of test samples : {}\n".format(len(test_dataset)),
    )

    token_to_id = checkpoint["token_to_id"]
    id_to_token = checkpoint["id_to_token"]

    model = model = SATRN(options, id_to_token, token_to_id, model_checkpoint).to(device)

    model.eval()
    results = []
    for d in tqdm(test_data_loader):
        # print(d["image"].shape)
        images = d["image"].float().to(device)
        expected = d["truth"]["encoded"].to(device)

        output = model(images, expected, False, 0.0)
        
        # Original
        decoded_values = output.transpose(1, 2)

        ## cacluate probs
        probs = F.softmax(decoded_values, dim=1)    
        probs, arg_index = torch.max(probs, dim=-1)
        _, promising_index = probs.sum(dim=1).max(dim=0)

        _, sequence = torch.topk(decoded_values, 1, dim=1)
        ## get promising index sentence
        sequence = sequence.squeeze(1)[promising_index].unsqueeze(0)
        # sequence = sequence.squeeze(1)

        sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
        for path, predicted in zip(d["file_path"], sequence_str):
            results.append((path, predicted))

    os.makedirs(parser.output_dir, exist_ok=True)
    with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
        for path, predicted in results:
            w.write(path + "\t" + predicted + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default="C:/Users/KHJ/Desktop/checkpoints/0066.pth",
        type=str,
        help="Path of checkpoint file",
    )
    parser.add_argument(
        "--max_sequence",
        dest="max_sequence",
        default=230,
        type=int,
        help="maximun sequence when doing inference",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=1,
        type=int,
        help="batch size when doing inference",
    )

    eval_dir = os.environ.get('SM_CHANNEL_EVAL', 'D:/workspace2/input/data')
    file_path = os.path.join(eval_dir, 'train_dataset/test.txt')
    parser.add_argument(
        "--file_path",
        dest="file_path",
        default=file_path,
        type=str,
        help="file path when doing inference",
    )

    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', 'submit')
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default=output_dir,
        type=str,
        help="output directory",
    )

    parser = parser.parse_args()
    main(parser)
