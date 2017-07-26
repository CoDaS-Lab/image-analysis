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
import numpy as np
import skimage.io
from image_analysis.utils import utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.img_dir = 'test/test_data/testimgs/'
        self.testimg1_path = self.img_dir + 'test1.jpg'
        self.testimg2_path = self.img_dir + 'test2.jpg'
        self.timing_start = time.time()

    def tearDown(self):
        elapsed = time.time() - self.timing_start
        print('\n{} ({:.5f} sec)'.format(self.id(), elapsed))

    def test_load_mult_imgs(self):
        testimg1 = skimage.io.imread(self.testimg1_path)
        testimg2 = skimage.io.imread(self.testimg2_path)
        imgs = utils.load_mult_images(dirname=self.img_dir)

        self.assertTrue(np.array_equal(testimg1, imgs[0][0]))
        self.assertTrue(np.array_equal(testimg2, imgs[1][0]))

    def test_load_mult_imgs_batches(self):
        batchsize = 2
        imgs = utils.load_mult_images(dirname=self.img_dir,
                                      batchsize=batchsize)

        # 2 batches each with two images inside. Last batch should be padded
        self.assertEqual(len(imgs), 2)
        self.assertEqual(len(imgs[0]), batchsize)
        self.assertEqual(len(imgs[1]), batchsize)


if __name__ == '__main__':
    unittest.main()
