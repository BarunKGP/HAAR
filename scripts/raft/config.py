from dataclasses import dataclass


@dataclass
class RAFTConfig:
    dropout: float = 0
    alternate_corr: bool = False
    small: bool = False
    mixed_precision: bool = False

    def __iter__(self):
        for k in self.__dict__:
            yield k
