import os

import clip
import torch
from PIL import Image

def get_features(root: str, video_id: str, frame_id: str, narr: str) -> torch.Tensor:
    """_summary_

    Args:
        root (str): root directory
        video_id (str): video id of corresponding video
        frame_id (str): frame id of the clip
        narr (str): narration text for the frame

    Returns:
        feats (torch.Tensor): fused multimodal features for the frame
    """
    rgb_loc = os.path.join(
        root, 'rgb_frames', video_id, frame_id)
    flow_locs = [os.path.join(root, 'flow_frames', video_id, 'u', frame_id),
                    os.path.join(root, 'flow_frames', video_id, 'v', frame_id)]
    
    rgb_tensor = get_clip_features([rgb_loc])
    flow_tensor = get_clip_features(flow_locs, modality="flow_frames")
    narr_tensor = get_clip_features([narr], modality='narration')

    # print(f'Shapes: rgb = {rgb_tensor.shape}, flow = {flow_tensor.shape}, narr = {narr_tensor.shape}')
    
    feats = torch.hstack((rgb_tensor, flow_tensor, narr_tensor))
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
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device) # TODO: parameterize these

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
    
    return feats.squeeze()