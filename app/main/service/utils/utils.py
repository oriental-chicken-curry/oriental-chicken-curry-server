import torch.optim as optim

from app.main.service.networks.SATRN import SATRN


def get_network(model_type, FLAGS, model_checkpoint, device, train_dataset):
    """model설정 함수
    """
    model = None

    if model_type == "SATRN":
        model = SATRN(FLAGS, train_dataset, model_checkpoint).to(device)
    else:
        raise NotImplementedError

    return model


def get_optimizer(optimizer, params, lr, weight_decay=None):
    """optimizer 설정 함수
    """
    if optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == "Adadelta":
        optimizer = optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer
