import os
import pickle
# from configparser import ConfigParser
from typing import Any, List, Tuple

from clip_features import get_features
from constants import DATA_ROOT, STRIDE
from torch.utils.data import Dataset
from utils import get_sec

# config = ConfigParser()
# config.read(r'config.ini')

"""
Custom PyTorch Dataset class that creates the dataset from 
the pickle files. It returns the rgb, flow and narration 
embeddings concatenated into a tensor. Stride is implemented 
so that we only consider 1 frame per stride. This helps us reduce
the dataset size and improve computation time.
"""

class FrameLoader(Dataset):
    def __init__(self, loc, info_loc, train: bool = True) -> None:
        super().__init__()
        self.dataset = []
        self.video_info_df = self.generate(info_loc)
        self.data_df = self.generate(loc)

        self.create_dataset()

    def generate(self, file_loc: str) -> List[Tuple]:
        """Generates the dataset from pickle file.

        Arguments:
            file_loc -- relative path to the pickle file

        Returns:
                the dataset as a list of tuple values
        """
        try:
            with open(file_loc, 'rb') as handle:
                df = pickle.load(handle)
        except:
            raise FileNotFoundError(f'Invalid pickle location: {file_loc}')

        return df

    def create_dataset(self):
        """Creates the dataset as a list of tuples. 
        Each entry in self.data_df is added as an item
        in self.dataset. Schema for self.dataset:
        self.dataset: [(
                participant_root_dir, 
                video_id, 
                frame_id, 
                narration_text
            )]

        Raises:
            Exception: Empty dataset exception
        """
        if self.video_info_df.empty or self.data_df.empty:
            raise Exception("Empty DataFrame")
        i = 0
        for index, row in self.data_df.iterrows():
            i += 1
            if i > 2:
                break
            video_id, participant_id, narr_timestamp, narr_text = (
                row['video_id'], row['participant_id'], row['narration_timestamp'], row['narration'])
            frame_rate = float(
                self.video_info_df.loc[self.video_info_df['video_id'] == video_id]['fps'].iat[0])
            start_frame = int(get_sec(narr_timestamp) * frame_rate)
            if index == len(self.data_df) - 1:
                end_frame = int(self.video_info_df.loc[self.video_info_df['video_id'] == video_id]
                                ['duration'].iat[0] * frame_rate) + 1
            else:
                end_frame = int(get_sec(self.data_df.at[index + 1, 'narration_timestamp']) * frame_rate)

            participant_root_dir = os.path.join(DATA_ROOT, participant_id)
            
            for frame in range(start_frame, end_frame, STRIDE):
                frame_id = 'frame_' + str(frame).zfill(10) + '.jpg'
                self.dataset.append((participant_root_dir, video_id, frame_id, narr_text))

        print('Created dataset')
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: str) -> Any:
        """Default Dataset function

        Args:
            idx (str): index

        Returns:
            video_id (str): video id of corresponding video
            frame_id (str): frame id of the clip
            feat (torch.Tensor): fused multimodal features of
                size (4096,)
        """
        root, video_id, frame_id, narr = self.dataset[idx]
        feats = get_features(root, video_id, frame_id, narr)
        return (video_id, frame_id, feats)