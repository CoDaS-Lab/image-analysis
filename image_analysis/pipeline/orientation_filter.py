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


import numpy as np
from skimage.color import rgb2gray
from image_analysis.pipeline.fft import FFT
from image_analysis.pipeline.feature import Feature


class OrientationFilter(Feature):
    """
    DESCRIPTION:
            Creates a filter that can be multiplied by the amplitude spectrum
            of an image to increase/decrease specific orientations/spatial
            frequencies.

    PARAMS:
        :inputshape: shape of then input for pyfftw builder
        :center_orientation: int for the center orientation (0-180)
        :orientation_width: int for the orientation width of the filter
        :high_cutoff: int high spatial frequency cutoff
        :low_cutoff: int low spatial frequency cutoff
        :target_size: int total size.
        :falloff: string 'triangle' or 'rectangle' shape of the filter
                falloff from the center.
        :nthreads: number of multithreads
    """

    def __init__(self, mask='bowtie', center_orientation=90,
                 orientation_width=20, high_cutoff=None, low_cutoff=.1,
                 target_size=None, falloff=''):

        Feature.__init__(self, mask + '_filter', frame_op=True,
                         batch_op=False)

        self.mask = mask
        available_mask = ['bowtie', 'noise']
        if self.mask not in available_mask:
            raise ValueError('mask: {0} does not exist'.format(mask))

        if orientation_width == 0:
            raise ValueError('Can\'t set orientation_width to 0 because ' +
                             'it will cause a division by zero in triangle ' +
                             'filter code.')
        # TODO: make this a parameter
        inputshape = (480, 640)
        self.center_orientation = center_orientation
        self.orientation_width = orientation_width
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff
        self.target_size = target_size
        self.fft = FFT(inputshape=inputshape, nthreads=4)

        self.falloff = falloff or 'triangle'
        available_falloff = ['rectangle', 'triangle']
        if self.falloff not in available_falloff:
            raise ValueError('falloff: {0} is invalid'.format(self.falloff))

        if self.mask == 'bowtie':
            self.filter = self.bowtie(center_orientation, orientation_width,
                                      high_cutoff, low_cutoff, target_size,
                                      falloff)
            self.filter = 1 - self.filter
            self.filter = self.fft.fftshift(self.filter)
        elif self.mask == 'noise':
            self.filter = self.noise_amp(target_size)
        else:
            raise ValueError('invalid mask: {0}'.format(self.mask))

        self.filter = self.filter[:inputshape[0], :inputshape[1]]

    def bowtie(self, center_orientation, orientation_width, high_cutoff,
               low_cutoff, target_size, falloff=''):
        """
        DESCRIPTION:
            Creates a filter that can be multiplied by the amplitude spectrum
            of an image to increase/decrease specific orientations/spatial
            frequencies.

        PARAMS:
            :center_orientation: int for the center orientation (0-180).
            :orientation_width: int for the orientation width of the filter.
            :high_cutoff: int high spatial frequency cutoff.
            :low_cutoff: int low spatial frequency cutoff.
            :target_size: int total size.
            :falloff: string 'triangle' or 'rectangle' shape of the filter
                    falloff from the center.

        RETURN:\n
            the bowtie shaped filter.
        """

        x = y = np.linspace(0, target_size // 2, target_size // 2 + 1)
        u, v = np.meshgrid(x, y)

        # derive polar coordinates: (theta, radius), where theta is in degrees
        theta = np.arctan2(v, u) * 180 / np.pi
        radii = (u**2 + v**2) ** 0.5

        # using radii for one quadrant, build the other 3 quadrants
        flipped_radii = np.fliplr(radii[:, 1:target_size // 2])
        radii = np.concatenate((radii, flipped_radii), axis=1)
        flipped_radii = np.flipud(radii[1:target_size // 2, :])
        radii = np.concatenate((radii, flipped_radii), axis=0)
        radii = self.fft.fftshift(radii)
        # note: the right-most column and bottom-most row were sliced off

        # using theta for one quadrant, build the other 3 quadrants
        flipped_theta = 90 + np.fliplr((theta[1:target_size // 2 + 1, :].T))
        # note: +1 is done for theta, but not for radii
        # note: transpose is done for theta, but not for radii
        theta = np.concatenate((flipped_theta, theta), axis=1)
        flipped_theta = 180 + np.flipud(np.fliplr(theta[1:, :]))
        # might be able to optimize by transposing and then flipping
        # instead of flip and then flip
        theta = np.concatenate((flipped_theta, theta), axis=0)

        center_orientation_2 = 180 + center_orientation
        # The 2D frequency spectrum is mirror symmetric, orientations must be
        # represented on both sides. All orientation functions below must be
        # repeated using both center_orientation's

        # clockwise orientation cutoff, from center_orientation
        cwb1 = center_orientation + orientation_width / 2
        # counterclockwise orientation cutoff, from center_orientation
        ccwb1 = center_orientation - orientation_width / 2
        # clockwise orientation cutoff, from center_orientation_2
        cwb2 = center_orientation_2 + orientation_width / 2
        # counterclockwise orientation cutoff, from center_orientation_2
        ccwb2 = center_orientation_2 - orientation_width / 2

        if ccwb1 < 0:
            theta = np.fliplr(theta).T
            center_orientation += 90
            center_orientation_2 += 90
            cwb1 += 90
            ccwb1 += 90
            cwb2 += 90
            ccwb2 += 90

        theta = theta[0:target_size, 0:target_size]

        # dim's
        anfilter = np.zeros(theta.shape)
        sffilter = (low_cutoff <= radii) & (radii <= high_cutoff)

        if falloff is 'rectangle':
            anfilter = ((ccwb1 <= theta) & (theta <= cwb1)) | (
                (ccwb2 <= theta) & (theta <= cwb2))
            # filt = sffiler*anfilter
        elif falloff is 'triangle':
            for idx, val in np.ndenumerate(theta):
                if ccwb1 <= val <= cwb1 and val <= center_orientation:
                    anfilter[idx] = (val - center_orientation +
                                     orientation_width / 2) * \
                        2 / orientation_width
                elif ccwb1 <= val <= cwb1 and val > center_orientation:
                    anfilter[idx] = (-val + center_orientation +
                                     orientation_width / 2) * \
                        2 / orientation_width
                elif ccwb2 <= val <= cwb2 and val <= center_orientation_2:
                    anfilter[idx] = (val - center_orientation_2 +
                                     orientation_width / 2) \
                        * 2 / orientation_width
                elif ccwb2 <= val <= cwb2 and val > center_orientation_2:
                    anfilter[idx] = (-val + center_orientation_2 +
                                     orientation_width / 2) \
                        * 2 / orientation_width
                else:
                    anfilter[idx] = 0
        else:
            angfilter1 = np.exp(-((theta - center_orientation) /
                                  (.5 * orientation_width)) ** 4)
            angfilter2 = np.exp(-((theta - center_orientation_2) /
                                  (.5 * orientation_width)) ** 4)
            anfilter = angfilter1 + angfilter2

        return sffilter * anfilter

    def noise_amp(self, size):
        """
        DESCRIPTION:
            Creates a size x size matrix of randomly generated noise with
            amplitude values with 1/f slope

        PARAMS:
            :size: size of matrix

        RETURN:
            :returns the amplitudes with noise added
        """

        slope = 1
        x = y = np.linspace(1, size, size)
        xgrid, ygrid = np.meshgrid(x, y)  # coordinates for a square grid
        xgrid = np.subtract(xgrid, size // 2)
        ygrid = np.subtract(ygrid, size // 2)

        amp = self.fft.fftshift(np.divide(np.sqrt(np.square(xgrid) +
                                          np.square(ygrid)),
                                          size * np.sqrt(2)))
        amp = np.rot90(amp, 2)
        amp[0, 0] = 1
        amp = 1 / amp**slope
        amp[0, 0] = 0
        return amp

    def extract(self, frame):
        """
        DESCRIPTION:
            Transforms a matrix using FFT, multiplies the result by a mask, and
            then transforms the matrix back using Inverse FFT.\n

        PARAMS:
            :input_frame: (m x n) numpy array
            :mask: int determining the type of filter to implement, where
                  1 = iso (noize amp) and 2 = horizontal decrement
                  (bowtie)

        RETURN:
            return the transformed and processed frame
        """
        if frame is None:
            return ValueError('Frame is none')

        # target size and frame width must be the same
        # https://github.com/CoDaS-Lab/IsoVideo/blob/master/isovideo/filter.py#L27
        # assert frame.shape[1] == self.target_size

        # fft spectrum
        altimg = None
        grayframe = rgb2gray(frame)
        dft_frame = self.fft.fft2d(grayframe)
        phase = np.arctan2(dft_frame.imag, dft_frame.real)

        # create filter
        if self.mask == 'noise':
            amp = np.copy(self.filter)
            phase = np.exp(phase * 1j)
            amp = np.multiply(phase, amp)

            # inverse fft and normalize
            altimg = self.fft.ifft2d(amp).real
            altimg -= altimg.min()
            altimg /= altimg.max()

        elif self.mask == 'bowtie':
            altimg = self.fft.ifft2d(dft_frame * self.filter).real

        return altimg
