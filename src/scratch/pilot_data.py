import os
import pickle

import pandas as pd
from constants import DATA_ROOT
from sklearn.model_selection import train_test_split
from utils import get_sec


def _format_ds_(data_df, video_info_df):
    df = data_df[
        [
            "video_id",
            "participant_id",
            "narration_timestamp",
            "narration",
            "verb_class",
            "noun_class",
        ]
    ]
    df = pd.merge(df, video_info_df, how="left", on="video_id")
    df["start_frame"] = df.apply(
        lambda row: int(get_sec(row["narration_timestamp"]) * row["fps"]) + 1,
        axis=1,
    )
    df["end_frame"] = df.groupby("video_id")["start_frame"].shift(-1, fill_value=0)
    df["end_frame"] = df.apply(
        lambda row: int(row["duration"] * row["fps"]) + 1
        if row["end_frame"] == 0
        else row["end_frame"],
        axis=1,
    )
    df["root_dir"] = df.apply(
        lambda row: os.path.join(DATA_ROOT, row["participant_id"]), axis=1
    )
    df = df[
        [
            "video_id",
            "root_dir",
            "narration",
            "start_frame",
            "end_frame",
            "verb_class",
            "noun_class",
        ]
    ]
    return df


with open("data/epic-kitchens-100-annotations/EPIC_100_train.pkl", "rb") as f:
    df_train = pickle.load(f)
df_train100 = df_train[df_train.video_id.apply(lambda x: len(x) == 7)]

video_df = pd.read_csv("data/epic-kitchens-100-annotations/EPIC_100_video_info.csv")

# test-train split
participants = df_train100.participant_id.unique()
p_train, p_test = train_test_split(participants, test_size=0.15)

train = df_train100[df_train100["participant_id"].isin(p_train)]
test = df_train100[df_train100["participant_id"].isin(p_test)]

print("Train participants: ", p_train, "\n")
print("Test participants: ", p_test)

# format DataFrames
train_modified = _format_ds_(train, video_df)
test_modified = _format_ds_(test, video_df)
print(train_modified.info())
print(test.info())

with open("data/train100_mod.pkl", "xb") as handle:
    pickle.dump(train_modified, handle)
print("Wrote train pickle")

with open("data/test100_mod.pkl", "xb") as handle:
    pickle.dump(test, handle)
print("Wrote test pickle")
