import torch
import cv2
import numpy as np
import os
import torch.nn.functional as F

from pprint import pprint
from app.main.service.utils.checkpoint import load_checkpoint

from app.main.service.data.dataset import LoadEvalDataset, collate_eval_batch, START, PAD, END
from app.main.service.utils.flags import Flags
from app.main.service.networks.SATRN import SATRN
import random

from app.main.service.log import *

from PIL import Image, ImageOps


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


def id_to_string(tokens, token_to_id,id_to_token, do_eval=0):
    """token id 를 문자열로 변환하는 로직

    Args:
        tokens(list) : 토큰 아이디
        data_loader(Dataloaer) : 현재 사용하고 있는 데이터 로더
        do_eval(int): 0 - train, 이 외 - eval
    """
    result = []
    if do_eval:
        eos_id = token_to_id["<EOS>"]
        special_ids = [token_to_id["<PAD>"], token_to_id["<SOS>"],token_to_id["<EOS>"]]

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += id_to_token[token] + " "
                elif token == eos_id:
                    break
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += id_to_token[token] + " "

        result.append(string)
    return result


def main(image_info):
    checkpoint_file = "C:/Users/KHJ/Desktop/checkpoints/0070.pth"

    # eval_dir = os.environ.get('SM_CHANNEL_EVAL', '/Users/heesup/Applications/boostCamp_Pstage/stage4_OCR/')

    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', 'submit')

    is_cuda = torch.cuda.is_available()
    checkpoint = load_checkpoint(checkpoint_file, cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("image_info", image_info.shape)

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

    # Augmentation
    _, _, _, _, test_transformed = get_dataset(options)

    # token id dictionary
    token_to_id = checkpoint["token_to_id"]
    id_to_token = checkpoint["id_to_token"]

    model = SATRN(options, id_to_token, token_to_id, model_checkpoint).to(device)
    model.eval()
    results = []

    input_images = []

    dummy_sentence = "\sin " * 230  # set maximum inference sequence

    # 이미지 가져오기
    # image = Image.open("/Users/heesup/Applications/boostCamp_Pstage/stage4_OCR/eval_dataset/images/train_00000.jpg")
    image = Image.fromarray(image_info)
    image = image.convert("L")

    # crop
    bounding_box = ImageOps.invert(image).getbbox()
    image = image.crop(bounding_box)

    # numpy 변환
    image = np.array(image)
    image = image.astype(np.uint8)
    image_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)    

    transformed = test_transformed(image=image)
    image = transformed["image"]
    image = image.float()

    transformed_270 = test_transformed(image=image_270)
    image_270 = transformed_270["image"]
    image_270 = image_270.float()

    transformed_90 = test_transformed(image=image_90)
    image_90 = transformed_90["image"]
    image_90 = image_90.float()

    input_images.append(image.numpy())
    # input_images.append(image.numpy())
    input_images.append(image_270.numpy())
    input_images.append(image_90.numpy())

    input_images = np.array(input_images)
    input_images = torch.tensor(input_images)

    input_images = input_images.to(device)
    

    expected = [np.array([token_to_id[START]] + encode_truth(dummy_sentence, token_to_id) + [token_to_id[END]]),
                np.array([token_to_id[START]] + encode_truth(dummy_sentence, token_to_id) + [token_to_id[END]])]

    expected = torch.Tensor(expected)
    expected = expected.to(device)

    print("input image shape", input_images.shape)
    print("expected shape", expected.shape)

    output = model(input_images, expected, False, 0.0)

    file_paths = ["test1.jpg", "test1.jpg", "test1.jpg"]

    decoded_values = output.transpose(1, 2)#.mean(0).unsqueeze(0)

    x = F.softmax(decoded_values, dim=1)
    x, _ = torch.max(x, dim=-1)

    _, sequence = torch.topk(decoded_values, 1, dim=1)    
    sequence = sequence.squeeze(1)
    # pprint(sequence)
    row, col = torch.nonzero(sequence==1, as_tuple=True)    
   
    # print(col[row==0][0].item(), col[row==1][0].item(), col[row==2][0].item())
    index = np.argmax([x[0][:col[row==0][0].item()].mean().item(), \
                       x[1][:col[row==1][0].item()].mean().item(), \
                       x[2][:col[row==2][0].item()].mean().item()])

    sequence_str = id_to_string(sequence[index].unsqueeze(0), token_to_id, id_to_token, do_eval=1)
    for path, predicted in zip(file_paths, sequence_str):
        results.append((path, predicted))

    res = []
    os.makedirs(output_dir, exist_ok=True)
    for path, predicted in results:
        res.append(predicted)

    return res[0]


if __name__ == "__main__":
    main()
