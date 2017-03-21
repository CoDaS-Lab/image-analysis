import os
import sys
sys.path.append(os.getcwd() + "/../")

import skimage.io
import wget
from matplotlib import pyplot as plt
from utils.perf import *
from decode import video_decoder as vd
from extract import feature_extractor as fe
from extract.fft import FFT


vid_path = os.getcwd() + '/../test/test_data/'

batch_list = vd.decode_mpeg(os.getcwd() + '/../test/test_data/test_video.mp4',
                            batch_size=2, end_idx=10)


transformed_data = fe.extract_features(batch_list,
                                       [FFT])

# Now, let's access  some of the features extracted from frames
img = img_fft = transformed_data[0]['input']['frame']
img_fft = transformed_data[0]['input']['fast_fourier_transform']
skimage.io.imshow_collection([img, img_fft])
plt.show()
