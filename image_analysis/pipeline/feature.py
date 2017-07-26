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


class Feature:
    """
    DESCRIPTION:
        Base class for features we want to extract or transformations we
        want to apply to data. If batch_op and frame_op is left false then it
        is a learning model (or anything else you want really)

    PARAMS:
        :batch_op: boolean to say the feature runs on batches of frames
        :frame_op: boolean to say the feature runs on each frame
        :save: boolean check to save feature in output dict
    """

    def __init__(self, key_name, batch_op=False, frame_op=False, save=False):
        self.batch_op = batch_op
        self.frame_op = frame_op
        self.key_name = key_name
        self.save = save

    def extract(self, **args):
        """
        DESCRIPTION:
            extract features from the data
        """
        raise NotImplementedError

    def train(self, **args):
        """
        DESCRIPTION:
            train models on images
        """
        raise NotImplementedError

    def predict(self, Y):
        """
        DESCRIPTION:
            predict new points
        """
        raise NotImplementedError
