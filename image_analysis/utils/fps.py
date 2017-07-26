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


from time import time


class FPS:
    """
    DESCRIPTION:
        calculates frames per seconds
    """
    def __init__(self):
        self._start = 0
        self._end = 0
        self._nframes = 0

    def start(self):
        """
        DESCRIPTION:
            start timer
        """
        self._start = time()

    def update(self):
        """
        DESCRIPTION:
            update timer and number of frames
        """
        if self.elapsed() >= 1:
            print('fps {}'.format(self.fps()))
            self._nframes = 0
            self._start = self._end
        else:
            self._nframes += 1

    def elapsed(self):
        """
        DESCRIPTION:
            calculate elapsed
        """
        self._end = time()
        return (self._end - self._start)

    def fps(self):
        """
        DESCRIPTION:
            returns the fps
        """
        return self._nframes / self.elapsed()


