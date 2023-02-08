import os
import pickle
from configparser import ConfigParser
from typing import Any, List, Tuple

import pandas
from clip_features import get_feats
from torch.utils.data import Dataset, DataLoader
from utils import get_sec

config = ConfigParser()
config.read('config.ini')


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
            with open(file_loc, 'xb') as handle:
                df: pandas.DataFrame = pickle.load(handle)
        except:
            raise FileNotFoundError

        return df

    def create_dataset(self):
        """Creates the dataset as a list of tuples. 
        Each entry in self.data_df is added as an item
        in self.dataset. Schema for self.dataset:
        self.dataset: [(
                participant_root_dir, 
                video_id, 
                start_frame,
                end_frame, 
                narration_text
            )]

        Raises:
            Exception: Empty dataset exception
        """
        if self.video_info_df.empty or self.data_df.empty:
            raise Exception("Empty DataFrame")

        for index, row in self.data_df.iterrows():
            video_id, participant_id, narr_timestamp, narr_text = (
                row['video_id'], row['participant_id'], row['narration_timestamp'], row['narration'])
            FRAME_RATE = float(
                self.video_info_df.loc[self.video_info_df['video_id'] == video_id]['fps'].iat[0])
            start_frame = int(get_sec(narr_timestamp) * FRAME_RATE)
            if index == len(self.data_df) - 1:
                end_frame = int(self.video_info_df.loc[self.video_info_df['video_id'] == video_id]
                                ['duration'].iat[0] * FRAME_RATE) + 1  # Check this logic
            else:
                end_frame = int(
                    get_sec(self.data_df.at[index + 1, 'narration_timestamp']) * FRAME_RATE)

            participant_root_dir = os.path.join(
                config.get('default', 'data_root'), participant_id)
            self.dataset.append(
                (participant_root_dir, video_id, start_frame, end_frame, narr_text))
        print('Created dataset')
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: str) -> Any:
        """_summary_

        Args:
            idx (str): _description_

        Returns:
            video_id (str): Returns video id of corresponding video.
            feat (dict): Returns a dict containing multimodal features.
                 feats: {
                    'rgb_frames': torch.Tensor(size=(n_frames, 512)),
                    'flow_frames: torch.Tensor(size=(n_frames, 1024)),
                    'narration': torch.Tensor(size=(1, 512))
                    }
        """
        STRIDE = config.getint('feature_extraction', 'stride')
        root, video_id, start, end, narr = self.dataset[idx]
        feats = get_feats(root, video_id, start, end, narr, stride=STRIDE)
        return video_id, feats

if __name__ == '__main__':
    pickle_root = config.get('default', 'pickle_root')
    dataset = FrameLoader(
        loc = os.path.join(pickle_root, 'samples/df_train100_first10.pkl'), 
        info_loc = os.path.join(pickle_root, 'video_info.pkl')
        )
    loader = DataLoader(dataset, shuffle=True, batch_size=8)

    for video_id, feats in loader:
        print(f"Video {video_id} \t feature shape: rgb = {feats['rgb_frames'].shape}, flow = {feats['flow_frames'].shape}, narration = {feats['narration'].shape}")


