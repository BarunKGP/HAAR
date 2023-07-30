import random
from torch.utils.data import Dataset

from datasets.tsn_dataset import TsnDataset


class HaarDataset(Dataset):
    def __init__(self, rgb_dataset: TsnDataset, flow_dataset: TsnDataset) -> None:
        self.rgb_dataset = rgb_dataset
        self.flow_dataset = flow_dataset
        super().__init__()

    def __getitem__(self, index):
        return self.rgb_dataset[index], self.flow_dataset[index]

    def __len__(self):
        return min(len(self.flow_dataset), len(self.rgb_dataset))

    def debug(self):
        print(f"rgb_dataset = {len(self.rgb_dataset)} items")
        print(f"flow_dataset = {len(self.flow_dataset)} items")
        random_idx = random.sample(range(1000), 10)
        print(
            "rgb: (video_id, narration_id, verb_class) | flow: (video_id, narration_id, verb_class)"
        )
        for idx in random_idx:
            print(
                f'{self.rgb_dataset[idx][1]["video_id"], self.rgb_dataset[idx][1]["narration_id"], self.rgb_dataset[idx][1]["verb_class"]}'
                + f'{self.flow_dataset[idx][1]["video_id"], self.flow_dataset[idx][1]["narration_id"], self.flow_dataset[idx][1]["verb_class"]}'
            )
