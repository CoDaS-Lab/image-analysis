import skimage.color
from extract.feature import Feature


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
