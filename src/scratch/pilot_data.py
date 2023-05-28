import os
from pathlib import Path
import pickle
import re
from natsort import natsorted

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from constants import DATA_ROOT, PICKLE_ROOT
    from utils import get_sec, write_pickle
except ImportError or ModuleNotFoundError:
    import sys

    sys.path.append(sys.path[0] + "/..")
    from constants import DATA_ROOT, PICKLE_ROOT
    from utils import get_sec, write_pickle

print(PICKLE_ROOT)


def _format_ds_(data_df, video_info_df):
    df = data_df.sort_values(by=["video_id", "narration_timestamp"])
    df = pd.merge(df, video_info_df, how="left", on="video_id")
    final_frames = find_final_frames()
    df["start_frame"] = df.apply(
        lambda row: int(get_sec(row["narration_timestamp"]) * row["fps"]) + 1,
        axis=1,
    )
    df["end_frame"] = df.groupby("video_id")["start_frame"].shift(-1, fill_value=0)
    df["end_frame"] = df.apply(
        # lambda row: int(row["duration"] * row["fps"]) + 1
        lambda row: final_frames[row["video_id"]] + 1
        if row["end_frame"] == 0
        else row["end_frame"],
        axis=1,
    )
    df["start_frame"] = (df["start_frame"] - 100).clip(1)
    df["root_dir"] = df.apply(
        lambda row: os.path.join(DATA_ROOT, row["participant_id"]), axis=1
    )
    df = df.drop(labels=["participant_id"], axis=1)
    df = df[df.start_frame < df.end_frame]
    df = df[
        [
            "video_id",
            "root_dir",
            "narration_id",
            "narration",
            "narration_timestamp",
            "start_frame",
            "end_frame",
            "verb_class",
            "noun_class",
        ]
    ]
    return df


def main():
    with open("data/epic-kitchens-100-annotations/EPIC_100_train.pkl", "rb") as f:
        df_train = pickle.load(f)
    df_train100 = df_train[df_train.video_id.apply(lambda x: len(x) == 7)]
    df_train100 = df_train100.reset_index()

    video_df = pd.read_csv("data/epic-kitchens-100-annotations/EPIC_100_video_info.csv")

    # test-train split
    videos = df_train100.video_id.unique()
    p_train, p_test = train_test_split(videos, test_size=0.15)
    p_train, p_val = train_test_split(p_train, test_size=0.12)

    train = df_train100[df_train100["video_id"].isin(p_train)]
    val = df_train100[df_train100["video_id"].isin(p_val)]
    test = df_train100[df_train100["video_id"].isin(p_test)]

    print("Train videos: ", len(p_train))
    print("Validation videos:", len(p_val))
    print("Test videos: ", len(p_test))

    # format DataFrames
    train_modified = _format_ds_(train, video_df)
    val_modified = _format_ds_(val, video_df)
    test_modified = _format_ds_(test, video_df)
    print(train_modified.info())
    print(val_modified.info())
    print(test_modified.info())

    with open("data/train100_mod.pkl", "xb") as handle:
        pickle.dump(train_modified, handle)
    print("Wrote train pickle")

    with open("data/val100_mod.pkl", "xb") as handle:
        pickle.dump(val_modified, handle)
    print("Wrote val pickle")

    with open("data/test100_mod.pkl", "xb") as handle:
        pickle.dump(test_modified, handle)
    print("Wrote test pickle")


def find_final_frames(root=PICKLE_ROOT, save_loc="../../data", write_results=True):
    regex = re.compile(r"\d+")
    final_frames = {}
    root = Path(root)
    for subdir in root.iterdir():
        if subdir.is_dir():
            subroot = Path(subdir) / "flow_frames"
            for video_folder in subroot.iterdir():
                if video_folder.is_dir():
                    frames = (Path(video_folder) / "u").glob("*.jpg")
                    frames = natsorted(frames, key=str, reverse=True)
                    final_frame = [int(x) for x in regex.findall(frames[-1])]
                    final_frames[os.path.basename(video_folder)] = final_frame

    if write_results:
        write_pickle(final_frames, save_loc)
        print("Finished writing final_frames")

    return final_frames


if __name__ == "__main__":
    main()
