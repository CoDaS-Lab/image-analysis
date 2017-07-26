# Copyright 2017 Codas Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import skvideo.io
import numpy as np


def pad_batch(batch, batch_size, frame, pad=True):
    """
    DESCRIPTION:
        Take in a batch, pad it with 0s if necessary and return appended batch

    PARAMS:
        :frame: a numpy array extracted from an MPEG
                (length x width x channel)
        :batch: list of frames
        :batch_size: number of frames per batch (integer >= 1)

    RETURN:
        batch: list of a batch, batch has batch_size frames, each an ndarray:
        (L x W x C)
    """

    if len(batch) > batch_size:
        raise ValueError('len(batch) should be <= batch_size!')
    elif batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    elif len(batch) == batch_size:
        return batch
    elif len(batch) < batch_size:
        if pad is True:
            pad_list = [np.zeros(frame.shape)] * (batch_size - len(batch))
            batch += pad_list
        return batch
    else:
        raise ValueError('Something is wrong with the pad_batch function')


def decode_mpeg(v_path, batch_size=1, stride=1, start_idx=0, end_idx=-1,
                pad=True):
    """
    DESCRIPTION:
        Creates a list of batches of frames from an MPEG file.

    PARAMS:
        :v_path: Path to MPEG video (i.e. include the video's name &
                extension)
        :batch_size: Number of frames in each batch
        :stride: Stride indicates beginning of batches, i.e. every stride'th
                frame (integer > 1)
        :start_idx: Index of first frame for first batch (integer >= 0)
        :end_idx: Index of last frame in the range of interest (integer >= 0)
        :pad: Boolean value indicating whether the last batch should be
             padded if it is not full after decoding the mpeg

    RETURNS:
        LIST of NUMPY batches of frames (length x width x channels), and
        if the last batch is not full, it is padded with frames of:
        np.zeros((frame.shape))

    """

    if start_idx < 0 or end_idx < -1:
        raise ValueError('Cannot use start_idx < 0 or end_idx < -1')
    if end_idx > -1 and end_idx < start_idx:
        raise ValueError('Cannot use end_idx < start_idx')

    if batch_size < 1 or stride < 1:
        raise ValueError('Cannot use batch_size or stride < 1')

    batch_list = []
    count = 0
    temp = start_idx
    batch = []

    # grab frame count from video metadata
    if end_idx == -1:
        metadata = skvideo.io.ffprobe(v_path)
        vid_frame_count = int(metadata['video']['@nb_frames'])
        end_idx = vid_frame_count - 1

    # build batch_list
    for frame in skvideo.io.vreader(v_path):
        if count == end_idx:
            if len(batch) == batch_size:
                batch_list.append(np.array(batch))
                batch = []
            idx = count - ((count - start_idx) % stride)
            if count >= idx and count < (idx + batch_size):
                batch.append(frame)
                batch_list.append(np.array(
                    pad_batch(batch, batch_size, frame, pad=pad)))
            if stride > batch_size:
                return batch_list[1:]
            elif batch_size >= stride:
                return batch_list
        elif count >= start_idx and stride > batch_size:
            if (count - start_idx) % stride == 0:
                temp = count
                batch_list.append(np.array(batch))
                batch = []
                batch.append(frame)
            elif temp < count and count < (temp + batch_size):
                batch.append(frame)
        elif count >= start_idx and batch_size >= stride:
            if count - start_idx < batch_size:
                batch.append(frame)
            elif len(batch) == batch_size:
                batch_list.append(np.array(batch))
                batch = []
                for x in range(stride - batch_size, 0):
                    batch.append(batch_list[-1][x])
                batch.append(frame)
            elif len(batch) < batch_size:
                batch.append(frame)
        count += 1
