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


import unittest
import warnings
import skvideo.io
import os
import time
import numpy as np
from image_analysis.decode import video_decoder as vd

""" Do not delete these comments.
Data for test_video.mp4 is saved in test_video_data.npy.

You do not want to write tests using an skvideo function because if it breaks
or changes in functionality, you will waste time trying to figure out whether
the issue was video_decoder.py, skvideo, or the test(s).
"""


class TestVideoDecoder(unittest.TestCase):

    def setUp(self):
        data_dir = 'test/test_data/'

        self.vid_path = data_dir + 'test_video.mp4'
        self.correct_data = np.load(data_dir + 'test_video_data.npy')
        self.nframes = self.correct_data.shape[0]
        self.width = self.correct_data.shape[2]
        self.height = self.correct_data.shape[1]
        self.nchannels = self.correct_data.shape[3]

        self.timing_start = time.time()

    def tearDown(self):
        elapsed = time.time() - self.timing_start
        print('\n{} ({:.5f} sec)'.format(self.id(), elapsed))

    def test_decode_mpeg(self):
        # Ignore weird resource warnings for now.
        warnings.simplefilter('ignore')

        # Verifies dimensions: nframes x height x width, nchannels.
        def test_mpeg_dimensions(message='', *, nframes=self.nframes,
                                 vd_batch=None):
            if vd_batch is None:
                vd_nframes = self.nframes
                vd_height = self.height
                vd_width = self.width
                vd_nchannels = self.nchannels
            else:
                vd_nframes = vd_batch.shape[0]
                vd_height = vd_batch.shape[1]
                vd_width = vd_batch.shape[2]
                vd_nchannels = vd_batch.shape[3]

            self.assertEqual(nframes, vd_nframes, message + '\nnframes ' +
                             '= {0}, vd_nframes = {1}'.format(
                                 self.nframes, vd_nframes))
            self.assertEqual(self.height, vd_height, message +
                             '\nself.height = {0}, vd_height = {1}'.format(
                                 self.height, vd_height))
            self.assertEqual(self.width, vd_width, message +
                             '\nselfwidth = {0}, vd_width = {1}'.format(
                                 self.width, vd_width))
            self.assertEqual(self.nchannels, vd_nchannels, message +
                             '\nself.nchannels = {0}, vd_channels ' +
                             '= {1}'.format(self.nchannels, vd_nchannels))

        # Verifies that the frames are the same.
        def test_mpeg_frame(message='', *, frame=0, vd_frame=0):
            self.assertTrue(np.array_equal(frame, vd_frame), message +
                            '\nDecoded frame does not match.')

        # Checks for the number of correct batches.
        def test_mpeg_nbatches(message='', *,
                               nbatches=0, vd_nbatches=0):
            self.assertEqual(nbatches, vd_nbatches, message +
                             '\nnbatches = {0}, vd_nbatches = {1}'.format(
                                 nbatches, vd_nbatches))

        # Test default decode_mpeg settings.
        numframes = 1
        numbatches = self.nframes
        batch_list = vd.decode_mpeg(self.vid_path)
        test_mpeg_nbatches('Default settings test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))

        batch = batch_list[0]
        prompt = 'Default settings test: check first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[0])
        batch = batch_list[-1]
        prompt = 'Default settings test: check last frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[-1])

        # Test indices, with each frame as a separate batch: end > start.
        start = 13
        end = 88
        numbatches = end - start + 1
        numframes = 1
        batch_list = vd.decode_mpeg(self.vid_path, start_idx=start,
                                    end_idx=end)

        test_mpeg_nbatches('end > start test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))

        batch = batch_list[0]
        prompt = 'end > start test: check first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[start])
        batch = batch_list[-1]
        prompt = 'end > start test: check last frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[end])

        # Test each frame as a separate batch, using indices: start > end.
        # should raise valueerror

        # Test each frame as a separate batch, using indices start = end != 1.
        start = end = 14
        numbatches = 1
        numframes = 1
        batch_list = vd.decode_mpeg(self.vid_path, start_idx=start,
                                    end_idx=end)

        test_mpeg_nbatches('end == start test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))

        batch = batch_list[0]
        prompt = 'end == start test: check frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[start])

        # Test batch_size = stride, with padding on.
        start = 1
        end = 99
        numframes = b_stride = 10
        numbatches = (end - start + 1) // numframes + ((end - start + 1)
                                                       % numframes > 0)
        batch_list = vd.decode_mpeg(self.vid_path,
                                    start_idx=start, end_idx=end,
                                    batch_size=numframes, stride=b_stride)

        test_mpeg_nbatches('batch_size = stride test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))

        batch = batch_list[0]
        prompt = 'batch_size = stride test: check first batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[start])
        prompt = 'batch_size = stride test: check first batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=self.correct_data[start + numframes - 1])
        batch = batch_list[-1]
        prompt = 'batch_size = stride test: check last batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[start \
                        + numframes * ((end - start + 1) // numframes)])
        prompt = 'batch_size = stride test: check last batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=np.zeros(self.correct_data[0].shape))

        # Test batch_size > stride, with padding on.
        start = 0
        end = 100
        numframes = 15
        b_stride = 10
        numbatches = ((end - start + 1) // b_stride) + \
            (((end - start + 1) % b_stride) > (numframes - b_stride))
        batch_list = vd.decode_mpeg(self.vid_path,
                                    start_idx=start, end_idx=end,
                                    batch_size=numframes, stride=b_stride)

        test_mpeg_nbatches('batch_size > stride test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))

        batch = batch_list[0]
        prompt = 'batch_size > stride test: check first batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[start])
        prompt = 'batch_size > stride test: check first batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=self.correct_data[start + numframes - 1])
        batch = batch_list[1]
        prompt = 'batch_size > stride test: check second batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[start + b_stride])
        prompt = 'batch_size > stride test: check second batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=self.correct_data[start + b_stride + \
                                                numframes - 1])
        batch = batch_list[-1]
        prompt = 'batch_size>stride test: check last batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[start +
                        b_stride * (numbatches -
                                    (((end - start + 1) % b_stride) < (
                                        numframes - b_stride)))])

        prompt = 'batch_size > stride test: check last batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=np.zeros(self.correct_data[0].shape))


        # Test batch_size < stride.
        start = 0
        end = 100
        numframes = 10
        b_stride = 15
        numbatches = ((end - start + 1) // b_stride) + \
                     (((end - start + 1) % b_stride) > 0)
        batch_list = vd.decode_mpeg(self.vid_path,
                                    start_idx=start, end_idx=end,
                                    batch_size=numframes, stride=b_stride)

        test_mpeg_nbatches('batch_size > stride test:', nbatches=numbatches,
                           vd_nbatches=len(batch_list))

        batch = batch_list[0]
        prompt = 'batch_size < stride test: check first batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[start])
        prompt = 'batch_size < stride test: check first batch, last frame'
        test_mpeg_frame(prompt, vd_frame=batch[-1],
                        frame=self.correct_data[start + numframes - 1])
        batch = batch_list[-1]
        prompt = 'batch_size < stride test: check last batch, first frame'
        test_mpeg_dimensions(prompt, nframes=numframes, vd_batch=batch)
        test_mpeg_frame(prompt, vd_frame=batch[0],
                        frame=self.correct_data[
                            start + b_stride * ((end - start + 1) // b_stride)])
        prompt = 'batch_size < stride test: check last batch, ' + \
                 'last \"real frame\"'
        test_mpeg_frame(prompt, vd_frame=batch[-(((end - start + 1) % b_stride)\
                        > 0) * (((end - start + 1) % 15) - numframes)],
                        frame=self.correct_data[-1 - (((end - start + 1) %
                                                b_stride - numframes) > 0) *
                                                ((end - start + 1) % b_stride -
                                                numframes)])


if __name__ == '__main__':
    unittest.main()
