import os
from frame_loader import FrameLoader
from scripts.constants import BATCH_SIZE, PICKLE_ROOT

from torch.utils.data import DataLoader


def get_dataloader():
    dataset = FrameLoader(
        loc = os.path.join(PICKLE_ROOT, 'samples/df_train100.pkl'),
        info_loc= os.path.join(PICKLE_ROOT, 'video_info.pkl')
        )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader