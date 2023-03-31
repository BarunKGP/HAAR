import torch
import einops

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

def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out

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

class ActionMeter():
    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.noun_meter = AverageMeter(name, fmt)
        self.verb_meter = AverageMeter(name, fmt)
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val_noun = self.val_verb = 0
        self.avg_noun = self.avg_verb = 0
        self.sum_noun = self.sum_verb = 0
        self.count = 0

    def update(self, val_noun: float, val_verb: float, n: int = 1):
        self.val_noun = val_noun
        self.val_verb = val_verb
        self.sum_noun += val_noun * n
        self.sum_verb += val_verb * n
        self.count += n
        self.avg_noun = self.sum_noun / self.count
        self.avg_verb = self.sum_verb / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
