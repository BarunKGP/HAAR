import os
import pickle

# from configparser import ConfigParser
from typing import Any, List, Tuple

from clip_features import get_features
from constants import DATA_ROOT, STRIDE

import pandas as pd
from torch.utils.data import Dataset
from utils import get_sec


class FrameLoader(Dataset):

    """
    Custom PyTorch Dataset class that creates the dataset from
    the pickle files. It returns the rgb, flow and narration
    embeddings concatenated into a tensor. Stride is implemented
    so that we only consider 1 frame per stride. This helps us reduce
    the dataset size and improve computation time.
    """

    def __init__(self, loc: str, info_loc: str, train: bool = True) -> None:
        super().__init__()
        self.dataset = []
        self.video_info_df = self.generate(info_loc)
        self.data_df = self.generate(loc)
        if self.video_info_df.empty or self.data_df.empty:
            raise Exception("Empty DataFrame")
        self.data_df = self.data_df.reset_index()  # Special case for pilot
        self.create_dataset()

    def generate(self, file_loc: str) -> pd.DataFrame:
        """Generates the dataset from pickle file.

        Arguments:
            file_loc -- relative path to the pickle file

        Returns:
            df (DataFrame) -- the dataset as a pandas DataFrame
        """
        try:
            with open(file_loc, "rb") as handle:
                df = pickle.load(handle)
        except:
            raise FileNotFoundError(f"Invalid pickle location: {file_loc}")
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
        self.data_df = self.data_df.sort_values(by=["video_id", "narration_timestamp"])

        # Broadcasting
        # df = self.data_df[
        #     [
        #         "video_id",
        #         "participant_id",
        #         "narration_timestamp",
        #         "narration",
        #         "verb_class",
        #         "noun_class",
        #     ]
        # ]
        # df = pd.merge(df, self.video_info_df, how="left", on="video_id")
        # df["start_frame"] = df.apply(
        #     lambda row: int(get_sec(row["narration_timestamp"]) * row["fps"]) + 1,
        #     axis=1,
        # )
        # df["end_frame"] = df.groupby("video_id")["start_frame"].shift(-1, fill_value=0)
        # df["end_frame"] = df.apply(
        #     lambda row: int(row["duration"] * row["fps"]) + 1
        #     if row["end_frame"] == 0
        #     else row["end_frame"],
        #     axis=1,
        # )
        # df["root_dir"] = df.apply(
        #     lambda row: os.path.join(DATA_ROOT, row["participant_id"]), axis=1
        # )
        # df = df[
        #     [
        #         "video_id",
        #         "root_dir",
        #         "narration",
        #         "start_frame",
        #         "end_frame",
        #         "verb_class",
        #         "noun_class",
        #     ]
        # ]
        # self.dataset = df.to_numpy()

        # ? Takes a lot of time ig - need to check
        for index, row in self.data_df.iterrows():  # We need index to be an integer
            video_id = row["video_id"]
            participant_id = row["participant_id"]
            narr_timestamp = row["narration_timestamp"]
            narr_text = row["narration"]
            verb_class = row["verb_class"]
            noun_class = row["noun_class"]

            frame_rate = float(
                self.video_info_df.loc[self.video_info_df["video_id"] == video_id][
                    "fps"
                ].iat[0]
            )
            start_frame = int(get_sec(narr_timestamp) * frame_rate) + 1
            if (
                len(self.data_df) > index + 1  # type: ignore
                and self.data_df.loc[index + 1, "video_id"] == video_id  # type: ignore
            ):
                end_frame = int(
                    get_sec(self.data_df.at[index + 1, "narration_timestamp"])  # type: ignore # type: ignore
                    * frame_rate
                )
            else:
                end_frame = (
                    int(
                        self.video_info_df.loc[
                            self.video_info_df["video_id"] == video_id
                        ]["duration"].iat[0]
                        * frame_rate
                    )
                    + 1
                )
            # assert (
            #     end_frame/frame_rate<=self.video_info_df[["video_id"] == video_id]["duration"].iat[0]
            # ), f"end_frame = {end_frame} for video {video_id} is longer than duration"
            participant_root_dir = os.path.join(DATA_ROOT, participant_id)

            for frame in range(start_frame, end_frame, STRIDE):
                frame_id = "frame_" + str(frame).zfill(10) + ".jpg"
                self.dataset.append(
                    (
                        index,
                        participant_root_dir,
                        video_id,
                        frame_id,
                        narr_text,
                        verb_class,
                        noun_class,
                    )
                )
        # print("Created dataset...")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        """Default Dataset function

        Args:
            idx (int): index

        Returns:
            video_id (str): video id of corresponding video
            frame_id (str): frame id of the clip
            feat (torch.Tensor): fused multimodal features of
                size (4096,)
        """
        video_id, root, frame_id, narr, verb_class, noun_class = self.dataset[idx]
        feats = get_features(root, video_id, frame_id, narr)
        return (feats, verb_class, noun_class)
