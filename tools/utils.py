import os
import errno
import warnings
import os.path as osp


def makedir(fpath):
    """
    Creates directory of path if it is missing
    """
    if osp.isfile(fpath):
        dir_name = osp.dirname(fpath)
    elif osp.isdir(fpath):
        dir_name = fpath
    else:
        warnings.warn("{} is not a right path!".format(fpath))

    if not osp.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def isfile(fpath):
    """
    Checks if the given fpath is a file. True if it is.
    """
    indicator = osp.isfile(fpath)
    if not indicator:
        warnings.warn('No file is found at "{}"'.format(fpath))
    return indicator


def str_boolean(s='true'):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid bollean string')
    return s.upper() == 'TRUE'


