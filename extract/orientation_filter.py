import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pyfftw
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from matplotlib import pyplot as plt
from decode.utils import timeit
from extract.feature import Feature


class OrientationFilter(Feature):
    """DESCRIPTION:\n
            Creates a filter that can be multiplied by the amplitude spectrum
            of an image to increase/decrease specific orientations/spatial
            frequencies.

        PARAMS:\n
            center_orientation: int for the center orientation (0-180).
            orientation_width: int for the orientation width of the filter.
            high_cutoff: int high spatial frequency cutoff.
            low_cutoff: int low spatial frequency cutoff.
            target_size: int total size.
            falloff: string 'triangle' or 'rectangle' shape of the filter
                    falloff from the center."""

    def __init__(self, mask='bowtie', center_orientation=90,
                 orientation_width=20, high_cutoff=None, low_cutoff=.1,
                 target_size=None, falloff='', ):

        Feature.__init__(self, mask + '_filter', frame_op=True, 
                         batch_op=False)

        self.mask = mask
        available_mask = ['bowtie', 'noise']
        if self.mask not in available_mask:
            raise ValueError('mask: {0} does not exist'.format(mask))

        self.center_orientation = center_orientation
        self.orientation_width = orientation_width
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff
        self.target_size = target_size

        self.falloff = falloff or 'triangle'
        available_falloff = ['rectangle', 'triangle']
        if self.falloff not in available_falloff:
            raise ValueError('falloff: {0} does not exist'.format(self.falloff))

    def bowtie(self, center_orientation, orientation_width, high_cutoff,
               low_cutoff, target_size, falloff=''):
        """
        DESCRIPTION:\n
            Creates a filter that can be multiplied by the amplitude spectrum
            of an image to increase/decrease specific orientations/spatial
            frequencies.

        PARAMS:\n
            center_orientation: int for the center orientation (0-180).
            orientation_width: int for the orientation width of the filter.
            high_cutoff: int high spatial frequency cutoff.
            low_cutoff: int low spatial frequency cutoff.
            target_size: int total size.
            falloff: string 'triangle' or 'rectangle' shape of the filter
                    falloff from the center.

        RETURN:\n
            filt: return the bowtie shaped filter.
        """
        if (target_size % 2) != 0:
            raise ValueError('Target_size should be even!')

        x = y = np.linspace(1, target_size // 2, target_size // 2 + 1)
        u, v = np.meshgrid(x, y)

        # derive polar coordinates: (theta, radius), where theta is in degrees
        theta = np.arctan2(v, u) * 180 / np.pi
        radii = (u**2 + v**2) ** 0.5

        # using radii for one quadrant, build the other 3 quadrants
        flipped_radii = np.fliplr(radii[:, 1:target_size // 2])
        radii = np.concatenate((radii, flipped_radii), axis=1)
        flipped_radii = np.flipud(radii[1:target_size // 2, :])
        radii = np.concatenate((radii, flipped_radii), axis=0)
        radii = np.fft.fftshift(radii)
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

        if falloff == 'rectangle':
            anfilter = ((ccwb1 <= theta) & (theta <= cwb1)) | (
                (ccwb2 <= theta) & (theta <= csb2))
            # filt = sffiler*anfilter
        elif falloff == 'triangle':
            for idx, val in np.ndenumerate(theta):
                if ccwb1 <= val <= cwb1 and val <= center_orientation:
                    anfilter[idx] = (val - center_orientation +
                                     orientation_width / 2) * \
                        2 / orientation_width
                elif ccwb1 <= val <= cwb1 and val > center_orientation:
                    anfilter[idx] = (-val + center_orientation_2 +
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
        DESCRIPTION:\n
            Creates a size x size matrix of randomly generated noise with
            amplitude values with 1/f slope

        PARAMS:\n
            size: size of matrix

        RETURN:\n
            returns the amplitudes with noise added
        """

        slope = 1
        x = y = np.linspace(1, size, size)
        xgrid, ygrid = np.meshgrid(x, y)  # coordinates for a square grid
        xgrid = np.subtract(xgrid, size // 2)
        ygrid = np.subtract(ygrid, size // 2)

        amp = np.fft.fftshift(np.divide(np.sqrt(np.square(xgrid) +
                                        np.square(ygrid)),
                                        size * np.sqrt(2)))
        amp = np.rot90(amp, 2)
        amp[0, 0] = 1
        amp = 1 / amp**slope
        amp[0, 0] = 0
        return amp

    def extract(self, frame):
        """
        DESCRIPTION:\n
            Transforms a matrix using FFT, multiplies the result by a mask, and
            then transforms the matrix back using Inverse FFT.\n

        PARAMS:\n
            input_frame: (m x n) numpy array
            mask: int determining the type of filter to implement, where
                  1 = iso (noize amp) and 2 = horizontal decrement
                  (bowtie)

        RETURN:\n
            return the transformed and processed frame
        """
        if frame is None:
            return ValueError('Frame is invalid: {0}'.format(grayframe))

        # fft spectrum
        grayframe = rgb2gray(frame)
        dft_frame = pyfftw.interfaces.numpy_fft.fft2(grayframe)
        phase = np.arctan2(dft_frame.imag, dft_frame.real)
        size = np.shape(dft_frame)[1]

        # create filter
        if self.mask == 'noise':
            amp = self.noise_amp(size)
        elif self.mask == 'bowtie':
            amp = np.abs(dft_frame)
            # remove this when we have n x n images, amp is not same shape
            rows = grayframe.shape[1] - grayframe.shape[0]
            padding = np.zeros((rows, size))
            amp = np.append(amp, padding, axis=0)
            if self.high_cutoff is None:
                self.high_cutoff = size // 2

            if self.target_size is None:
                self.target_size = size

            bowtie = self.bowtie(self.center_orientation,
                                 self.orientation_width, self.high_cutoff,
                                 self.low_cutoff, self.target_size,
                                 self.falloff)

            amp = amp * (1 - bowtie)

        # fft spectrum  * amp (filter)
        phase = np.exp(phase * 1j)
        rows = amp.shape[0] - phase.shape[0]
        padding = np.ones((rows, size))
        phase = np.append(phase, padding, axis=0)
        amp = np.multiply(phase, amp)

        # remove the padded values
        amp = amp[:240, :]
        # inverse fft and normalize
        altimg = pyfftw.interfaces.numpy_fft.ifft2(amp).real
        altimg -= altimg.min()
        altimg /= altimg.max()

        return altimg

    # def extract(self, frame):
    #     grayframe = rgb2gray(frame)
    #     filtered_img = self.fft_mask(grayframe, 1)
    #     # RMS = 9
    #     # filtered_img = np.multiply(RMS, filtered_img)
    #     # filtered_img = np.multiply(filtered_img, np.std(filtered_img))
    #     # filtered_img = np.add(filtered_img, 5)
    #     return filtered_img
