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


import skimage.color
from image_analysis.pipeline.feature import Feature


class RGBToGray(Feature):
    def __init__(self):
        Feature.__init__(self, 'grayscale', frame_op=True)

    def extract(self, RGB_frame):
        return skimage.color.rgb2gray(RGB_frame)


class BatchOP(Feature):
    def __init__(self):
        Feature.__init__(self, 'batch_length', batch_op=True)

    def extract(self, batch):
        return len(batch)


class ArgMaxPixel(Feature):
    def __init__(self):
        Feature.__init__(self, 'max_pixel', frame_op=True)

    def extract(self, frame):
        return np.max(frame)
