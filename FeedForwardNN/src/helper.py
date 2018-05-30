def make_sure_path_exists(path):
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
