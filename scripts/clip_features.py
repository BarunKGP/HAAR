import torch
import clip
from PIL import Image
from pandas import DataFrame

import pickle
import argparse
import os

device = "cuda:1" if torch.cuda.is_available() else "cpu"
# device = 'cuda[1]'
model, preprocess = clip.load("ViT-B/32", device=device)

DATA_ROOT= '../../2g1n6qdydwa9u22shpxqzp0t8m/'
# FRAME_RATE = 59.94006

def get_dataset(data_loc, video_info_loc):
    '''Dataset here is a pickle file which contains a
    Pandas dataframe
    
    Args:
        data_root: file location of the pickle
        
    Returns: the dataframe within the pickle
    '''
    with open(data_loc, 'rb') as handle:
        dataset = pickle.load(handle)

    with open(video_info_loc, 'rb') as handle:
        video_info = pickle.load(handle)

    return dataset, video_info

def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def get_feats(dataset: DataFrame, video_info: DataFrame):
    if dataset.empty:
        raise Exception('Empty DataFrame')
    feats = {}
    for index, row in dataset.iterrows():
        video_id, participant_id, narr_timestamp = (
            row['video_id'], 
            row['participant_id'],
            row['narration_timestamp']
        )
        FRAME_RATE = float(video_info.loc[video_info['video_id'] == video_id]['fps'].iat[0])
        start_frame = int(get_sec(narr_timestamp) * FRAME_RATE)
        if index == len(dataset) - 1:
            end_frame = int(video_info.loc[video_info['video_id'] == video_id]['duration'].iat[0] * FRAME_RATE) + 1 # Check this logic
        else:
            end_frame = int(get_sec(dataset.at[index + 1, 'narration_timestamp']) * FRAME_RATE)
        
        feats_tensor = torch.empty(size=(end_frame - start_frame, 512))
        image_loc_root = os.path.join(DATA_ROOT, participant_id)
        i = 0
        for frame in range(start_frame, end_frame):
            # only rgb frames for now
            frame_str = 'frame_' + str(frame).zfill(10) + '.jpg'
            image_loc = os.path.join(image_loc_root, 'rgb_frames', video_id, frame_str)
            image = preprocess(Image.open(image_loc)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image) # (1, 512) vector
            feats_tensor[i] = image_features
            i += 1
        print(f'Extracted frames for index {index}')
        print(f'Shape of extracted frames_tensor: {feats_tensor.shape}')
        print(torch.cuda.memory_summary(device=1, abbreviated=False))
        feats[index] = feats_tensor

    return feats

# image_loc = '../../2g1n6qdydwa9u22shpxqzp0t8m/P01/rgb_frames/P01_101/frame_0000009288.jpg'
# image = preprocess(Image.open(image_loc)).unsqueeze(0).to(device)
# image_features = model.encode_image(image)
# print("Image features: ", image_features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_loc', type=str, required=True)
    parser.add_argument('--video_info_loc', type=str, required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    dataset, video_info = get_dataset(args.pickle_loc, args.video_info_loc)
    print('Initialized Dataframes')
    feats = get_feats(dataset, video_info)
    print('Finished extraction')
    with open('clip_rgb_features.pickle', 'xb') as handle:
        pickle.dump(feats, handle)
    print('Wrote image features to pickle')
    print('Exiting...')