import torch

def get_sec(time_str):
    """Get Seconds from time. Used to find the corresponding frame
    for a given timestamp
    """
    h, m, s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def log_print(logger, text, log_mode='debug'):
    """Log and print text to console

    Args:
        logger (_type_): _description_
        text (_type_): _description_
    """
    if log_mode == 'debug':
        logger.debug(text)
    elif log_mode == 'info':
        logger.info(text)
    elif log_mode == 'warn':
        logger.warn(text)
    print(text)
    
def get_device():
    """Decide whether to run on CPU or CUDA

    Returns:
        (str): device to run on
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    return device


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
