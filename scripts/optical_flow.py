import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from glob import glob
from PIL import Image
from tqdm import tqdm

from raft.core.raft import RAFT
from raft.core.utils import flow_viz
from raft.core.utils.utils import InputPadder
from raft.config import RAFTConfig

config = RAFTConfig(
    dropout=0,
    alternate_corr=False,
    small=False,
    mixed_precision=False
)

model = RAFT(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')

weights_path = 'raft/raft-sintel/raft-sintel.pth'

ckpt = torch.load(weights_path, map_location=device)
model.to(device)
model.load_state_dict(ckpt)

# TODO: Change to correct paths
# image_files = glob('/kaggle/input/raft-pytorch/raft/demo-frames/*.png')
# image_files = sorted(image_files)
def load_image(imfile, device):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def viz(img1, img2, flo):
    # TODO: change dimensions for grayscale
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    ax1.set_title('input image1')
    ax1.imshow(img1.astype(int))
    ax2.set_title('input image2')
    ax2.imshow(img2.astype(int))
    ax3.set_title('estimated optical flow')
    ax3.imshow(flo)
    plt.show()

def create_features(image_files1, image_files2):
    model.eval()

    for file1, file2 in tqdm(zip(image_files1, image_files2)):
        image1 = load_image(file1, device)
        image2 = load_image(file2, device)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        print(f'image1.shape = {image1.shape}, image2.shape = {image2.shape}')
        with torch.no_grad():
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            print(f'flow_low.shape = {flow_low.shape}, flow_up.shape = {flow_up.shape}')

    return flow_up

def main():
    img_root = '../../2g1n6qdydwa9u22shpxqzp0t8m/P01/rgb_frames/P01_101'
    image_files1, image_files2 = glob(os.path.join(img_root, 'u/frame_0000045937.jpg')), glob(os.path.join(img_root, 'v/frame_0000045940.jpg'))
    image_files1, image_files2 = sorted(image_files1), sorted(image_files2)

    print(f'Found {len(image_files1)}, {len(image_files2)} images')
    flow = create_features(image_files1, image_files2)
    # print(sorted(image_files))

if __name__ == '__main__':
    main()
