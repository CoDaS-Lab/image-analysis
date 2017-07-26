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


import os
import sys
import skimage.io
import wget
from matplotlib import pyplot as plt
from image_analysis.decode import decode_mpeg
from image_analysis.pipeline import Feature
from image_analysis.pipeline import Pipeline
from demo_features import RGBToGray
from demo_features import BatchOP

vid_path = os.getcwd() + '/../test/test_data/'  # Path to test video.

# Decode video to build dataset (list of nd arrays)
batch_list = decode_mpeg(vid_path + 'test_video.mp4',
                            batch_size=2, end_idx=10)

motion_analysis = Pipeline(data=batch_list,
                           ops=[BatchOP(), RGBToGray()],
                           seq=[RGBToGray(), BatchOP()],
                           save_all=True)

pipeline_output = motion_analysis.extract()
# print(pipeline_output[-1]['seq_features'])

# print(motion_analysis.as_ndarray(seq_key='batch_length'))
# motion_analysis.display()
# Now, let's access  some of the features extracted from frames
imgs = [pipeline_output[1]['input'],
        pipeline_output[1]['frame_features']['grayscale']]
skimage.io.imshow_collection(imgs)
plt.show()

# You can access particular features and keys in the data structure, too!
# for key, thing in pipeline_output[0]['input'].items():
#    print(key)
