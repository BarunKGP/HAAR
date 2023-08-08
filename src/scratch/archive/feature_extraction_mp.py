import multiprocessing as mp
from multiprocessing import Queue, Process, set_start_method
import logging
import logging.config
import logging.handlers
from logging.handlers import QueueHandler
import pickle
import threading
import subprocess
import os
import traceback
import sys

from models.clip.extract_clip import ExtractCLIP
from utils.utils import build_cfg_path
# from torch.multiprocessing import Process, set_start_method

from omegaconf import OmegaConf

# DATA_ROOT = '../pickles/df_train100_'  # the original for multiprocessing
DATA_ROOT = '../pickles/samples/df_train100_first10.pkl' # for testing
video_info = None
try:
    with open('../pickles/video_info.pkl', 'rb') as handle:
        video_info = pickle.load(handle)
    print('Extracted video_info.pkl')
except Exception as err:
    print('Failed to extract video_info.pkl: ', err)


# def listener_configurer():
#     root = logging.getLogger('extract')
#     h = logging.handlers.RotatingFileHandler('mp-extract.log', 'a', 300, 10)
#     f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
#     h.setFormatter(f)
#     root.addHandler(h)

def logger_thread(queue):
    while True:
        try:
            # consume a log message, block until one arrives
            message = queue.get()
            # check for shutdown
            if message is None:
                break
            # log the message
            logger = logging.getLogger(message.name)
            logger.handle(message)
        except Exception:
            print('Error in logging: ', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def extract_CLIP_features(video_path):
    feature_type = 'clip'
    # model_name = 'ViT-B/32'

    args = OmegaConf.load(build_cfg_path(feature_type))
    args.feature_type = feature_type
    # args.device_type = 'cuda:1'
    args.batch_size = 32

    extractor = ExtractCLIP(args)
    feats = extractor.extract(video_path)
    return feats

def extract_features(ctr: list, queue):
    # Logging framework
    logger = logging.getLogger('extract')
    # logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    process = mp.current_process()
    logger.info(f'Child process {process.name} starting')
    # Extraction
    for c in ctr:
        path = DATA_ROOT    # for testing performance
        # path = DATA_ROOT + str(c) + '.pkl'
        # Read in the dataframe from pickle
        df = None
        try:
            with open(path, 'rb') as handle:
                df = pickle.load(handle)
                df['video_path'] = '../2g1n6qdydwa9u22shpxqzp0t8m/' + df.participant_id + '/videos/' + df.video_id + '.MP4'
        except Exception:
            logger.exception('Error in loading dataframe')
            traceback.print_exc()

        # Extract features
        feat_dict = {}
        try:
            if df.empty:
                raise Exception('Empty dataframe detected')
            for index, row in df.iterrows():
                video_id, video_path, narr_timestamp = row['video_id'], row['video_path'], row['narration_timestamp']
                duration = video_info.loc[video_info['video_id'] == video_id]['duration'].iat[0]
                start = get_sec(narr_timestamp)
                # start = max(start - 2, 0)
                end = get_sec(df.at[index + 1, 'narration_timestamp']) if index < len(df) - 1 else duration
                command = f'ffmpeg -ss {start} -i {video_path} -c copy -to {end} temp.mp4 -y'
                subprocess.call(command, shell=True)
                feat_dict[index] = extract_CLIP_features('temp.mp4')
                os.remove('temp.mp4')
                logger.info(f'Extracted CLIP features for {video_id}')
                print('Extracted CLIP features for index = ', index)
        except Exception:
            traceback.print_exc()
            logger.exception(f'Error in feature extraction')

        out_file = '../pickles/samples/CLIP_feats_' + str(c) + '.pkl' # name = CLIP_feats_{ctr}.pkl
        print(f'Dumping features to pickle')
        logger.info(f'Creating pickle for features extracted by child process {process.name}')
        with open(out_file, 'xb') as handle:
            pickle.dump(feat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Created pickle file at {out_file}')
        logger.info(f'Created pickle file at {out_file} for child process {process.name}')
    logger.info(f'Child process {process.name} completed')


if __name__ == "__main__":
    set_start_method('spawn')
    queue = Queue()
    d = {
        'version': 1,
        'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'mplog2.log',
                'mode': 'a',
                'formatter': 'detailed',
            },
        },
        'loggers': {
            'extract': {
                'handlers': ['file']
            }
        }
    }
    logging.config.dictConfig(d)
    listener = threading.Thread(target=logger_thread, args=(queue,))
    listener.start()

    logger = logging.getLogger('extract')
    logger.addHandler(QueueHandler(queue))
    # logger_p = mp.Process(target=logger_process, args=(queue, ))
    # logger_p.start()
    logger.info('Main process started')
    ctr1 = [1]
    ctr2 = [2, 4]
    p1 = Process(target=extract_features, args=(ctr1, queue))
    # p2 = Process(target=extract_features, args=(ctr2, queue))
    p1.start()
    # p2.start()

    p1.join()
    # p2.join()

    logger.info('Main process completed')
    queue.put(None)
    listener.join()

                
            

