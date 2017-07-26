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
import numpy as np
import warnings
import cv2
import time
import skimage
from image_analysis.decode import video_decoder as vd
from image_analysis.pipeline OrientationFilter
from image_analysis.pipeline import Pipeline
from image_analysis.utils import FPS


cap = cv2.VideoCapture(0)
# image size
# we need to know the widths of the image in advance for the filter
vidwidth = 640
imgshape = (480, 640)
bowtie = OrientationFilter('bowtie', 90, 42, vidwidth // 2, .2,
                           vidwidth, 'triangle')
pipe = Pipeline(ops=[bowtie], save_all=True)

fps = FPS()
fps.start()
while True:
        warnings.simplefilter("ignore")

        # Capture frame-by-frame
        ret, frame = cap.read()
        pipe.data = [[frame]]
        altframe = pipe.extract()[0]['frame_features']['bowtie_filter']
        cv2.imshow('Real-Time', altframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        fps.update()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
