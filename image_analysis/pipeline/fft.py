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


import pyfftw
from pyfftw.interfaces.numpy_fft import fftshift
from .feature import Feature


class FFT(Feature):
    """
    DESCRIPTION:
        Fast fourier transform feature

    PARAMS:
        :inputshape: we need to know in advance the shape of input for
                     performance reason
        :usegpu: #TODO use gpu for implementation or not
        :nthreads: number of threads to run fft in multhreaded mode
    """
    def __init__(self, inputshape, usegpu=False, nthreads=1):
        Feature.__init__(self, 'FFT', frame_op=True)
        self.usegpu = usegpu
        self.inputshape = inputshape
        self.nthreads = nthreads

        # memory objects for input and output since pyfftw requires it
        # upfront
        inputobj = pyfftw.empty_aligned(inputshape, dtype='complex128')
        outputbj = pyfftw.empty_aligned(inputshape, dtype='complex128')

        # fft, ifft functions
        self.fft2obj = pyfftw.builders.fft2(inputobj, threads=nthreads)
        self.ifft2obj = pyfftw.builders.ifft2(outputbj, threads=nthreads)

        # improve performance of fft by caching pyfftw fft objects
        # see https://hgomersall.github.io/pyFFTW/sphinx/tutorial.html#caveat
        pyfftw.interfaces.cache.enable()

    def fft2d(self, input):
        """
        DESCRIPTION:
            2d fast fourier transform

        PARAMS:
            :input: input frame to transform
        """
        assert input.shape == self.inputshape

        return self.fft2obj(input)

    def ifft2d(self, input):
        """
        DESCRIPTION:
            2d inverse fast fourier transform

        PARAMS:
            :input: input frame to transform
        """
        assert input.shape == self.inputshape

        return self.ifft2obj(input)

    def fftshift(self, input):
        """
        DESCRIPTION:
            2d fast fourier shift

        PARAMS:
            :input: input frame to shift
        """
        return fftshift(input)
