import torch

use_cuda = torch.cuda.is_available()

default_checkpoint = {
    "epoch": 0,
    "train_losses": [],
    "train_symbol_accuracy": [],
    "train_sentence_accuracy": [],
    "train_wer": [],
    "validation_losses": [],
    "validation_symbol_accuracy": [],
    "validation_sentence_accuracy": [],
    "validation_wer": [],
    "lr": [],
    "grad_norm": [],
    "model": {},
    "configs":{},
    "token_to_id":{},
    "id_to_token":{},
}


def load_checkpoint(path, cuda=use_cuda):
    """ 저장되어 있는 모델을 반환

    Args:
        path(str) : 모델 저장 경로
        cuda(boolean) : cuda사용 여부
    """
    if cuda:
        return torch.load(path)
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)



