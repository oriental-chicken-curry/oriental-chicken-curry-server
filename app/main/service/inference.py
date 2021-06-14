import random

import numpy as np

from app.main.service.utils.checkpoint import load_checkpoint
from app.main.service.data.special_tokens import START, END
from app.main.service.utils.flags import Flags
from app.main.service.networks.SATRN import SATRN
from app.main.service.data.augmentation import get_transforms
from app.main.service.utils.log import *

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


def id_to_string(tokens, token_to_id, id_to_token, do_eval=0):
    """token id 를 문자열로 변환하는 로직

    Args:
        tokens(list) : 토큰 아이디
        data_loader(Dataloaer) : 현재 사용하고 있는 데이터 로더
        do_eval(int): 0 - train, 이 외 - eval
    """
    result = []
    if do_eval:
        eos_id = token_to_id["<EOS>"]
        special_ids = [token_to_id["<PAD>"], token_to_id["<SOS>"], token_to_id["<EOS>"]]

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


def image_processing(image, test_transformed, device):
    """ inference를 위한 이미지 처리 작업

    Args:
        image(np.array) : 요청받은 이미지
        test_transformed : image augmentation
        device : 사용 디바이스

    Returns:
        torch.tensor : 처리 된 이미지
    """

    input_images = []

    image = np.array(image)
    image = image.astype(np.uint8)

    transformed = test_transformed(image=image)
    image = transformed["image"]
    image = image.float()

    input_images.append(image.numpy())
    input_images.append(image.numpy())

    input_images = np.array(input_images)
    input_images = torch.Tensor(input_images).to(device)

    return input_images


def inference(image_info):
    """ 요청받은 이미지 추론 작업

    Args:
        image_info(np.array) : 요청받은 이미지 정보

    Returns:
        str : 이미지에 대한 latex 문자열
    """
    checkpoint_file = "/Users/heesup/Applications/boostCamp_Pstage/stage4_OCR/checkpoints/0068.pth"

    is_cuda = torch.cuda.is_available()
    checkpoint = load_checkpoint(checkpoint_file, cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = get_device()

    model_checkpoint = checkpoint["model"]

    # Augmentation
    _, _, test_transformed = get_transforms(options.augmentation, options.input_size.height, options.input_size.width)

    # token id dictionary
    token_to_id = checkpoint["token_to_id"]
    id_to_token = checkpoint["id_to_token"]

    model = SATRN(options, id_to_token, token_to_id, model_checkpoint).to(device)
    model.eval()
    results = []

    dummy_sentence = "\sin " * 230  # set maximum inference sequence

    # 이미지 가져오기
    image = Image.fromarray(image_info)
    image = image.convert("L")

    # crop
    bounding_box = ImageOps.invert(image).getbbox()
    image = image.crop(bounding_box)

    input_images = image_processing(image,test_transformed,device)

    expected = [np.array([token_to_id[START]] + encode_truth(dummy_sentence, token_to_id) + [token_to_id[END]]),
                np.array([token_to_id[START]] + encode_truth(dummy_sentence, token_to_id) + [token_to_id[END]])]
    expected = torch.Tensor(expected).to(device)

    output = model(input_images, expected, False, 0.0)

    decoded_values = output.transpose(1, 2)
    _, sequence = torch.topk(decoded_values, 1, dim=1)
    sequence = sequence.squeeze(1)
    sequence_str = id_to_string(sequence, token_to_id, id_to_token, do_eval=1)
    for predicted in sequence_str:
        results.append(predicted)

    res = []
    for predicted in results:
        res.append(predicted)

    get_result(res[0])

    return res[0]
