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
import time
import test.pipeline.test_features as features
from image_analysis.decode import video_decoder as vd
from image_analysis.pipeline.pipeline import Pipeline
from image_analysis.pipeline.svm import SVM
from sklearn import datasets


class TestPipeline(unittest.TestCase):

    def setUp(self):
        data_dir = 'test/test_data/'
        self.vid_path = data_dir + 'test_video.mp4'
        self.timing_start = time.time()

    def tearDown(self):
        elapsed = time.time() - self.timing_start
        print('\n{} ({:.5f} sec)'.format(self.id(), elapsed))

    def test_operations_save(self):
        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.ArgMaxPixel()

        testpipe = Pipeline(data=data,
                            ops=[rgb2gray, maxPixel],
                            save_all=True)
        # extract information or transform data by calling:
        pipeline_ouput = testpipe.extract()

        for frame in pipeline_ouput:
            op_keys = list(frame['frame_features'].keys())
            self.assertEqual(len(op_keys), 2)
            self.assertIsNotNone(frame['frame_features'][maxPixel.key_name])
            self.assertIsNotNone(frame['frame_features'][rgb2gray.key_name])
            self.assertIsNotNone(frame['input'])

            metadata = list(frame['meta_data'].keys())
            self.assertEqual(len(metadata), 2)
            self.assertIsNotNone(frame['meta_data']['frame_number'])
            self.assertIsNotNone(frame['meta_data']['batch_number'])

    def test_sequential_save(self):
        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.ArgMaxPixel()

        testpipe = Pipeline(data=data,
                            seq=[rgb2gray, maxPixel],
                            save_all=True)
        # extract information or transform data by calling:
        pipeline_ouput = testpipe.extract()

        for frame in pipeline_ouput:
            op_keys = list(frame['frame_features'].keys())
            self.assertEqual(len(op_keys), 0)
            self.assertIsNotNone(frame['seq_features'][maxPixel.key_name])
            self.assertIsNotNone(frame['seq_features'][rgb2gray.key_name])
            self.assertIsNotNone(frame['input'])

            metadata = list(frame['meta_data'].keys())
            self.assertEqual(len(metadata), 2)
            self.assertIsNotNone(frame['meta_data']['frame_number'])
            self.assertIsNotNone(frame['meta_data']['batch_number'])

    def test_sequential_nonsave(self):

        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.ArgMaxPixel()
        maxPixel.save = True

        testpipe = Pipeline(data=data,
                            seq=[rgb2gray, maxPixel],
                            save_all=False)
        # extract information or transform data by calling:
        pipeline_ouput = testpipe.extract()

        for frame in pipeline_ouput:
            op_keys = list(frame['frame_features'].keys())
            self.assertEqual(len(op_keys), 0)
            self.assertIsNotNone(frame['seq_features'][maxPixel.key_name])
            self.assertIsNotNone(frame['input'])

            metadata = list(frame['meta_data'].keys())
            self.assertEqual(len(metadata), 2)
            self.assertIsNotNone(frame['meta_data']['frame_number'])
            self.assertIsNotNone(frame['meta_data']['batch_number'])

    def test_as_ndarray(self):
        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        frame_width = data[0][0].shape[0]
        frame_height = data[0][0].shape[1]
        n = len(data) * len(data[0])  # n = numbatches * framesperbatch

        rgb2gray = features.RGBToGray()
        maxPixel = features.ArgMaxPixel()

        testpipe = Pipeline(data=data,
                            ops=[rgb2gray, maxPixel],
                            save_all=True)

        pipeline_ouput = testpipe.extract()
        gray_frames = testpipe.as_ndarray(frame_key=rgb2gray.key_name)
        self.assertTupleEqual(gray_frames.shape, (n, frame_width,
                                                  frame_height))

    def test_model_tranining(self):
        # test by running svm on digits
        digits = datasets.load_digits()
        images_and_labels = list(zip(digits.images, digits.target))
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))

        svm = SVM()
        pipe = Pipeline(models={'SVM': svm})
        pipe.train(data[:n_samples // 2], digits.target[:n_samples // 2])

        assert svm.classifier is not None

        expected = digits.target[n_samples // 2:]
        predicted = pipe.predict(data[n_samples // 2:])
        assert predicted['SVM'] is not None

if __name__ == '__main__':
    unittest.main()
