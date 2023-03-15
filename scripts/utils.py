def get_sec(time_str):
    """Get Seconds from time. Used to find the corresponding frame
    for a given timestamp
    """
    h, m, s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def log_print(logger, text, log_mode='debug'):
    """Log and print text to console

    Args:
        logger (_type_): _description_
        text (_type_): _description_
    """
    if log_mode == 'debug':
        logger.debug(text)
    elif log_mode == 'info':
        logger.info(text)
    elif log_mode == 'warn':
        logger.warn(text)
    print(text)
