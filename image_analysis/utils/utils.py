import time
import os
import numpy as np
import skimage.io
from functools import wraps

default_ext = ('.jpg', '.png')


def timeit(thresh=None, classname=''):
    """
    DESCRIPTION:
        python decorator to time functinos

    PARAMS:
        :thresh: minimum speed allowed. Throws error if function is slower
            than thresh
        :classname: class where the function you timing is located

    """
    def decorator(method):
        @wraps(method)
        def timed(*args, **kw):
            start = time.time()
            result = method(*args, **kw)
            elapsed = time.time() - start

            print('\n {0}: {1}() timing: {2:.5f} sec'
                  .format(classname, method.__name__, elapsed))

            if thresh is not None and elapsed > thresh:
                print('{0}() ran slower than thresh={1}'
                      .format(method.__name__, thresh))
            return result
        return timed
    return decorator


def load_mult_images(dirname, exts=None, batchsize=1):
    """
    DESCRIPTION:
        Load multiple images at once into batches

    PARAMS:
        :dirname: directory of the images
        :exts: acceptable file extensions (must be a tuble)
        :batchsize: size of the batches

    """

    if dirname is None or os.path.exists(dirname) is False:
        raise ValueError('dirname: {0} is invalid'.format(dirname))

    if batchsize <= 0:
        raise ValueError('batchsize: {0} is invalid'.format(batchsize))

    if exts is None:
        exts = default_ext
    elif not isinstance(exts, tuple):
        raise ValueError('exts: {0} is invalid'.format(exts.__class__))

    batch_list = []
    batch = []
    for fname in sorted(os.listdir(dirname)):
        if fname.endswith(exts):
            absolute_fname = dirname + fname
            # load frame add it to batch
            batch.append(skimage.io.imread(absolute_fname))

        if len(batch) == batchsize:
            batch_list.append(batch)
            batch = []

    # padd last batch (if needed) to keep all batches the same size
    if len(batch) < batchsize:
        padsize = batchsize - len(batch)
        padlist = [np.zeros(shape=batch_list[0][0].shape)] * padsize
        batch += padlist
        batch_list.append(batch)

    return batch_list
