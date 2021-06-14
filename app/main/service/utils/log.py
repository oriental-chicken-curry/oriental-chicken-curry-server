import logging
import torch


def __get_logger():
    """ 로깅을 위한 인스턴스 반환

    Returns
        logging : 로깅을 위한 객채 반환
    """

    __logger = logging.getLogger('logger')

    # # 로그 포멧 정의
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # 스트림 핸들러 정의
    stream_handler = logging.StreamHandler()
    # 각 핸들러에 포멧 지정
    stream_handler.setFormatter(formatter)
    # 로거 인스턴스에 핸들러 삽입
    __logger.addHandler(stream_handler)
    # 로그 레벨 정의
    __logger.setLevel(logging.DEBUG)

    return __logger


# 로깅을 위한 객체
logger = __get_logger()


def get_device():
    """현재 사용가능한 device를 반환

    Retruns:
        torch.device : 현재 사용 가능한 디바이스
    """
    logger.info("--------------------------------------------------")
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        logger.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logger.info(f'We will use the GPU:{torch.cuda.get_device_name(0)}')
    else:
        _device = torch.device("cpu")
        logger.info('No GPU available, using the CPU instead.')
    logger.info("--------------------------------------------------\n")

    return _device

def get_result(result : str):
    """결과값 로그
    """

    logger.info("--------------------------------------------------")
    logger.info("result : " + result)
    logger.info("--------------------------------------------------\n")
