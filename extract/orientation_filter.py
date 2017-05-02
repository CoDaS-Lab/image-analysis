import os
import sys
import numpy as np
from pyfftw.interfaces.numpy_fft import fftshift
from pyfftw.interfaces.numpy_fft import fft2
from pyfftw.interfaces.numpy_fft import ifft2
from skimage.color import rgb2gray
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
            raise ValueError('falloff: {0} is invalid'.format(self.falloff))

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

        if (orientation_width == 0):
            raise ValueError('Can\'t set orientation_width to 0 because ' +
                             'it will cause a division by zero in triangle ' +
                             'filter code.')

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
        radii = fftshift(radii)
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
        dft_frame = fft2(grayframe)
        phase = np.arctan2(dft_frame.imag, dft_frame.real)
        size = np.shape(dft_frame)[1]

        # create filter
        if self.mask == 'noise':
            amp = self.noise_amp(size)
            # fft spectrum  * amp (filter)
            phase = np.exp(phase * 1j)
            amp = np.multiply(phase, amp)

            # inverse fft and normalize
            altimg = ifft2(amp).real
            altimg -= altimg.min()
            altimg /= altimg.max()
            return altimg

        elif self.mask == 'bowtie':

            if self.high_cutoff is None:
                self.high_cutoff = size

            if self.target_size is None:
                self.target_size = size

            bowtie = self.bowtie(self.center_orientation,
                                 self.orientation_width, self.high_cutoff,
                                 self.low_cutoff, self.target_size,
                                 self.falloff)

            bowtie = 1 - bowtie
            bowtie = fftshift(bowtie)
            altimg = ifft2(dft_frame * bowtie).real.astype(int)
            return altimg

        return None
