import multiprocessing as mp
from multiprocessing import Queue, Process, set_start_method
import logging
import logging.config
import logging.handlers
from logging.handlers import QueueHandler
import threading
import traceback
import sys

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

def worker(ctr: int, k, queue):
    # Logging framework
    logger = logging.getLogger('test')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    process = mp.current_process()
    logger.info(f'Child process {process.name} starting')
    res = []
    logger.info(f'Starting processing for process {process.name}')
    for n in range(ctr):
        res.append(n ** k)
    print(f'Max = {max(res)}')
    logger.info(f'Max value calculated after process {process.name} = {max(res)}')
    logger.info(f'Exiting child process {process.name}')

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
                'level': 'INFO',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'mptest.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
        },
        'loggers': {
            'test': {
                'handlers': ['file']
            }
        }
    }
    logging.config.dictConfig(d)
    listener = threading.Thread(target=logger_thread, args=(queue,))
    listener.start()

    logger = logging.getLogger('test')
    logger.addHandler(QueueHandler(queue))

    logger.info('Main process started')
    ctr1 = 10003123
    ctr2 = 1000000
    p1 = Process(target=worker, args=(ctr1, 2, queue))
    p2 = Process(target=worker, args=(ctr2, 3.5, queue))
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    logger.info('Main process completed')
    queue.put(None)
    listener.join()
