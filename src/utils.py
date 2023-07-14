from typing import Literal, Union, List, Tuple
import logging
from logging.handlers import RotatingFileHandler
import pickle
import sys

import einops
import torch


def get_sec(time_str) -> float:
    """Get Seconds from time. Used to find the corresponding frame
    for a given timestamp
    """
    h, m, s = time_str.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def get_loggers(
    name: str,
    fmt: logging.Formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    ),
    filename: Union[None, str] = None,
    filehandler: Union[None, logging.FileHandler] = None,
    filelevel: int = logging.DEBUG,
    handlers: List[Tuple[logging.Handler, int]] = [
        (logging.StreamHandler(stream=sys.stdout), logging.INFO)
    ],
) -> logging.Logger:
    """Utility function to provide a set of loggers. It can create simultaneous
    logging streams, but generates a StreamHandler as default. Useful to write logs to
    multiple places at once. Common use cases include logging to stdout as well as
    maintaining a log file.

    Args:
        name (str): name of the logger
        fmt (logging.Formatter, optional): log message format. Defaults to logging.
            Formatter( "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s" ).
        filename (Union[None, str], optional): name of the log file. If not provided, logs
            are only written to the default handler. Defaults to None.
        filehandler (Union[None, logging.FileHandler], optional): file handler for log files.
            Defaults to None.
        filelevel (int, optional): logging level for log files. Defaults to
            logging.DEBUG.
        handlers (List[Tuple[logging.Handler, int]], optional): all handlers for logs.
            You can overwrite the defaults to pass additional handlers. However,
            if you need just a file handler along with the default handler,
            use the specific options provided. Defaults to
            [(logging.StreamHandler(stream=sys.stdout), logging.INFO)].

    Returns:
        (logging.Logger): logger with all handlers configured
    """

    logger = logging.getLogger(name)
    if filename is not None:
        if filehandler is None:
            filehandler = RotatingFileHandler(
                filename=filename, maxBytes=50000, backupCount=3
            )
        handlers.append((filehandler, filelevel))

    for handler, level in handlers:
        handler.setLevel(level)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def log_print(logger, text, log_mode="debug"):
    """Log and print text to console

    Args:
        logger (_type_): _description_
        text (_type_): _description_
    """
    if log_mode == "debug":
        logger.debug(text)
    elif log_mode == "info":
        logger.info(text)
    elif log_mode == "warn":
        logger.warn(text)
    print(text)


def get_device():
    """Decide whether to run on CPU or CUDA

    Returns:
        (str): device to run on
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
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
    if vectors.device != indices.device:
        indices = indices.to(vectors.device)
    N, _, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, _ = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out


def write_pickle(o, pname):
    with open(pname, "xb") as handle:
        pickle.dump(o, handle)


def read_pickle(pname, single=True):
    with open(pname, "rb") as handle:
        if single:
            return pickle.load(handle)
        data = []
        while True:
            try:
                data.append(pickle.load(handle))
            except EOFError:
                return data


class ActionMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def get_average_values(self):
        return self.avg

    def __str__(self):
        fmtstr = "({name} ({avg" + self.fmt + "}), ({count" + self.fmt + "})"

        return fmtstr.format(**self.__dict__)
