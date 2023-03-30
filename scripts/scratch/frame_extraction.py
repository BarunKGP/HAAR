from multiprocessing import get_logger, Pool
import logging
import os
import subprocess

from tqdm import tqdm

DATA_ROOT = "../2g1n6qdydwa9u22shpxqzp0t8m/"
participants = os.listdir(DATA_ROOT)
participants.remove('other')
participants.remove('P01')   # Do separately - some frames are already extracted

# Logger framework
logger = get_logger()
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
fileHandler = logging.FileHandler('frame_extraction_mp.log')
fileHandler.setFormatter(logFormatter)

logger.setLevel(logging.INFO)
logger.addHandler(fileHandler)

def get_tarfiles(root):
    onlyfiles = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    return onlyfiles


def extract(fpath):
    extracted_folder = fpath[:-4]
    os.mkdir(extracted_folder)
    command = f'tar -xf {fpath} -C {extracted_folder}'
    logger.info(command)
    subprocess.call(command, shell=True)
    logger.info('Extracted frames for ' + fpath)
    return fpath, len(os.listdir(extracted_folder))


def main():
    counter = {}
    for p in tqdm(sorted(participants)):
        rgb_root = os.path.join(DATA_ROOT, p, 'rgb_frames')
        flow_root = os.path.join(DATA_ROOT, p, 'flow_frames')
        rgb_tars = get_tarfiles(rgb_root)
        flow_tars = get_tarfiles(flow_root)

        # logger.info(rgb_tars)
        # logger.info(flow_tars)

        with Pool() as pool:
            for result in pool.imap_unordered(extract, rgb_tars):
                counter[result[0]] = counter[result[1]]

            for result in pool.imap_unordered(extract, flow_tars):
                counter[result[0]] = counter[result[1]]

            logger.info('Finished extracting frames for participant ' + p)

    logger.info('Main process completed')
    logger.info('Extraction statistics: ' + str(counter))

if __name__ == '__main__':
    main()





# for dirpath, dirnames, fnames in os.walk(DATA_ROOT):
#     print(dirpath)
#     print(dirnames)
#     print(fnames)
#     if dirnames == 'other':
#         continue
#     root = os.path.join(dirpath, dirnames)

