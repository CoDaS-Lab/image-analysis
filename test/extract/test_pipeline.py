import unittest
import os
import wget
import time
import sys
from decode import video_decoder as vd
from extract.pipeline import Pipeline
from test.extract import test_features as features

# TODO: write test for train_models and predict


class TestPipeline(unittest.TestCase):

    def setUp(self):
        data_dir = 'test/test_data/'
        self.vid_path = data_dir + 'test_video.mp4'
        vid_link = 'https://s3.amazonaws.com/codasimageanalysis/test_video.mp4'
        if not os.path.exists(self.vid_path):
            wget.download(vid_link, data_dir)

        self.timing_start = time.time()

    def tearDown(self):
        elapsed = time.time() - self.timing_start
        print('\n{} ({:.5f} sec)'.format(self.id(), elapsed))

    def test_transform_parallel_save(self):
        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.MaxPixel()
        batchNum = features.BatchOP()

        testpipe = Pipeline(data=data,
                            parallel=True,
                            save=True,
                            operations=[rgb2gray, maxPixel, batchNum],
                            models=None)

        # extract information or transform data by calling:
        pipeline_ouput = testpipe.transform()

        for batch in pipeline_ouput:
            for frame in batch:
                op_keys = list(frame['input'].keys())
                # op_keys should have only two items batchNum, rgb2gray,
                # maxPixel, and original
                self.assertEqual(len(op_keys), 4)
                self.assertIsNotNone(frame['input'][maxPixel.key_name])
                self.assertIsNotNone(frame['input'][rgb2gray.key_name])
                self.assertIsNotNone(frame['input'][batchNum.key_name])
                self.assertIsNotNone(frame['input']['original'])

                metadata = list(frame['metadata'].keys())
                self.assertEqual(len(metadata), 2)
                self.assertIsNotNone(frame['metadata']['frame_num'])
                self.assertIsNotNone(frame['metadata']['batch_num'])

    def test_transform_sequential(self):
        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.MaxPixel()
        batchNum = features.BatchOP()

        testpipe = Pipeline(data=data,
                            save=False,
                            parallel=False,
                            operations=[rgb2gray, maxPixel, batchNum],
                            models=None)
        # extract information or transform data by calling:
        pipeline_ouput = testpipe.transform()

        for batch in pipeline_ouput:
            for frame in batch:
                op_keys = list(frame['input'].keys())
                # op_keys should have only two items batchOP and rgb2gray
                # without the original
                self.assertEqual(len(op_keys), 2)
                self.assertIsNotNone(frame['input'][maxPixel.key_name])
                self.assertIsNotNone(frame['input'][batchNum.key_name])

                metadata = list(frame['metadata'].keys())
                self.assertEqual(len(metadata), 2)
                self.assertIsNotNone(frame['metadata']['frame_num'])
                self.assertIsNotNone(frame['metadata']['batch_num'])

    def test_transform_sequential_save(self):

        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.MaxPixel()
        batchNum = features.BatchOP()

        testpipe = Pipeline(data=data,
                            save=True,
                            parallel=False,
                            operations=[rgb2gray, maxPixel, batchNum],
                            models=None)
        # extract information or transform data by calling:
        pipeline_ouput = testpipe.transform()

        for batch in pipeline_ouput:
            for frame in batch:
                op_keys = list(frame['input'].keys())
                self.assertEqual(len(op_keys), 4)
                self.assertIsNotNone(frame['input'][maxPixel.key_name])
                self.assertIsNotNone(frame['input'][rgb2gray.key_name])
                self.assertIsNotNone(frame['input'][batchNum.key_name])
                self.assertIsNotNone(frame['input']['original'])

                metadata = list(frame['metadata'].keys())
                self.assertEqual(len(metadata), 2)
                self.assertIsNotNone(frame['metadata']['frame_num'])
                self.assertIsNotNone(frame['metadata']['batch_num'])

    def test_create_dict(self):
        fake_transformations = {'test1': 123456}
        fake_metadata = {'fake_metadata': 7890}
        testpipe = Pipeline()
        fake_dict = testpipe.create_dict(transforms=fake_transformations,
                                         metadata=fake_metadata)

        keys = list(fake_dict.keys())
        # only input and metadata in dict
        self.assertEqual(len(keys), 2)
        self.assertIsNotNone(fake_dict['input']['test1'])
        self.assertIsNotNone(fake_dict['metadata']['fake_metadata'])

    def test_data_as_nparray(self):
        data = vd.decode_mpeg(self.vid_path, batch_size=2, end_idx=9,
                              stride=2)

        rgb2gray = features.RGBToGray()
        maxPixel = features.MaxPixel()
        batchNum = features.BatchOP()

        testpipe = Pipeline(data=data,
                            save=True,
                            parallel=True,
                            operations=[rgb2gray, maxPixel, batchNum],
                            models=None)
        # extract information or transform data by calling:
        testpipe.transform()
        # get data as np arrays
        pipeline_ouput = testpipe.data_as_nparray()
        # shape should be (5, 2, 4)-> 5 batches with 2 frames each and 4
        # feature maps per frame (3 extracted features above + original frame)
        self.assertTupleEqual(pipeline_ouput.shape, (5, 2, 4))

if __name__ == '__name__':
    unittest.main()
