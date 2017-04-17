import os
import sys
sys.path.append(os.getcwd() + "/../")

import skimage.io
import wget
from matplotlib import pyplot as plt
from decode import video_decoder as vd
from extract import feature_extractor as fe
from demo_features import RGBToGray
from extract.orientation_filter import OrientationFilter
from demo_features import BatchOP
from extract.pipeline import Pipeline


vid_path = os.getcwd() + '/../test/test_data/'
data = vd.decode_mpeg(vid_path + 'test_video.mp4',
                      batch_size=2, end_idx=9, stride=2)

bowtie = OrientationFilter(mask='bowtie')

bpipe = Pipeline(data=data,
                 save=False,
                 parallel=False,
                 operations=[bowtie],
                 models=None)
# extract information or transform data by calling:
pipeline_ouput = bpipe.transform()
# the pipeline also saves the data internally for later access
# you can also access the data as numpy array by:
# bpipe.data_as_nparray()


# shape (batches, number of frames in batches, number of features per frame)
# grab first frame in first batch
frame = pipeline_ouput[0][0]

print(frame['input'].keys())
print(frame['metadata'].keys())


imgs = [frame['input']['original'],
        frame['input'][bowtie.key_name]]
skimage.io.imshow_collection(imgs)
plt.show()
