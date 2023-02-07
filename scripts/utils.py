def get_sec(time_str):
    """Get Seconds from time. Used to find the corresponding frame
    for a given timestamp
    """
    h, m, s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)