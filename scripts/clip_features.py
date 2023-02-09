import math
import os

from tqdm import tqdm

import clip
import torch
from PIL import Image
from constants import DATA_ROOT

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)



def get_features(root: str, video_id: str, start_frame: int, end_frame: int, narr: str, stride: int = 1) -> torch.Tensor:
    """_summary_

    Args:
        root (str): _description_
        video_id (str): _description_
        start_frame (int): _description_
        end_frame (int): _description_
        stride (int, optional): _description_. Defaults to 1.

    Raises:
        Exception: _description_

    Returns:
        torch.Tensor: _description_
    """
    if stride < 1:
        raise Exception("Stride should be at least 1")

    feats = {}
    rgb_tensor = torch.empty(size=(math.ceil((end_frame - start_frame)/stride), 1024))
    flow_tensor = torch.empty(size=(math.ceil((end_frame - start_frame)/stride), 2048))
    i = 0
    print(f'\n\n\n video: {video_id}, start = {start_frame}, end = {end_frame}')
    for frame in tqdm(range(start_frame, end_frame, stride), desc='Frame extraction progress: '):
        frame_str = 'frame_' + str(frame).zfill(10) + '.jpg'
        # print(f'frame_id = {frame_str}')
        rgb_loc = os.path.join(
            root, 'rgb_frames', video_id, frame_str)
        flow_locs = [os.path.join(root, 'flow_frames', video_id, 'u', frame_str),
                     os.path.join(root, 'flow_frames', video_id, 'v', frame_str)]
        rgb_tensor[i] = get_clip_features([rgb_loc])
        flow_tensor[i] = get_clip_features(flow_locs, modality="flow_frames")
        i += 1

    # print(torch.cuda.memory_summary(device=1, abbreviated=False))
    feats['rgb_frames'] = rgb_tensor
    feats['flow_frames'] = flow_tensor
    feats['narration'] = get_clip_features([narr], modality="narration")
    print(f'extracted frame features for {video_id} from frames {start_frame}-{end_frame}')
    return feats


def get_clip_features(data, modality: str = 'rgb_frames'):
    """Get multimodal clip features

    Args:
        data (list): a list of file locations. Must be a list of string
            if modality is "narration"
        modality (str, optional): the modality for feature extraction. 
            Defaults to 'rgb_frames'.

    Raises:
        Exception: raises exception for invalid modality or incorrect 
            data length

    Returns:
        _type_: _description_
    """
    assert type(data) == list, "data should be a list"
    feats = None
    if modality == "narration":
        assert len(data) == 1, "Entire narration text should be passed in a list"
        text = clip.tokenize(data).to(device)
        feats = model.encode_text(text)

    elif modality == "rgb_frames":
        assert len(data) == 1, "Pass only one frame for rgb"
        rgb_image = preprocess(Image.open(data[0])).unsqueeze(0).to(device)
        with torch.no_grad():
            rgb_features = model.encode_image(rgb_image)
        feats = rgb_features

    elif modality == "flow_frames":
        assert len(data) == 2, "Pass only 2 frame locations for flow_frames"
        flow_features = []
        for locs in data:
            flow_image = preprocess(Image.open(
                data[0])).unsqueeze(0).to(device)
            with torch.no_grad():
                flow_features.append(model.encode_image(flow_image))
        feats = torch.hstack(flow_features)
    else:
        raise Exception("Invalid modality")
    
    # print(f'{modality} feats: {feats.shape}')
    return feats.squeeze()


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pickle_loc', type=str, required=True)
#     parser.add_argument('--video_info_loc', type=str, required=True)
#     args = parser.parse_args()

#     return args


# if __name__ == '__main__':
#     args = parse_args()
#     dataset, video_info = get_dataset(args.pickle_loc, args.video_info_loc)
#     print('Initialized Dataframes')
#     feats = get_feats(dataset, video_info)
#     print('Finished extraction')
#     with open('clip_rgb_features.pickle', 'xb') as handle:
#         pickle.dump(feats, handle)
#     print('Wrote image features to pickle')
#     print('Exiting...')
