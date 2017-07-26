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
import skvideo.io
from skvideo.utils import rgb2gray
import os


def get_mpeg_dims(vid_fname):
    """
    ARGS
    vid_fname: file name for an mpeg video with one color channel

    RETURNS
    (tuple): containing number of rows, columns, and frames of data
    """
    meta_data = skvideo.io.ffprobe(vid_fname)
    height = meta_data['video']['@height']
    width = meta_data['video']['@width']
    n_frames = meta_data['video']['@nb_frames']

    return (int(height), int(width), int(n_frames))

'''
def checking_loop(checking, patch_num, idxs, frame_num, incomplete_patch_idxs,
                   incomplete_patches, frame, patch_dims):
        if patch_num >= idxs.shape[1]:
            checking = False
        elif frame_num != idxs[:, patch_num][2]:
            checking = False
        elif frame_num == idxs[:, patch_num][2]:
            row = idxs[:, patch_num][0]
            col = idxs[:, patch_num][1]
            incomplete_patch_idxs.append([row, col])
            incomplete_patches.append([frame[row: row + patch_dims[0],
                                                col: col + patch_dims[1]]])
            patch_num += 1
        else:
            raise ValueError('Checking loop is broken.')
'''

def gen_patch_idxs(vid_dims, patch_dims, n_patches, 
                   n_frames=None, save_dir=None):
    """ 
    ARGS
    vid_dims   (tuple):
    patch_dims (tuple):
    n_patches    (int): number of patches to sample from video
    n_frames     (int): number of consecutive frames to sample from video
    
    RETURNS
    indicies of "upper left" pixel of patches
    """
    max_idxs = (vid_dims[0] - patch_dims[0] + 1,
                vid_dims[1] - patch_dims[1] + 1,
                vid_dims[2] - patch_dims[2] + 1)
  
    if n_frames is None:
        row_idxs = np.random.randint(0, max_idxs[0], n_patches)
        col_idxs = np.random.randint(0, max_idxs[1], n_patches)
        frm_idxs = np.random.randint(0, max_idxs[2], n_patches)

    elif n_frames is not None:
        assert n_patches % n_frames == 0 and n_frames <= max_idxs[2]

        row_idxs = np.random.randint(0, max_idxs[0], n_patches)
        col_idxs = np.random.randint(0, max_idxs[1], n_patches)
        frm_idxs = np.array([x for x in range(n_frames)])
        n_repeats = n_patches // n_frames
        frm_idxs = np.repeat(frm_idxs, n_repeats)

    idxs = np.array([row_idxs, col_idxs, frm_idxs])
    idxs = idxs[:, np.argsort(idxs[2, :])]  # Sort by frame number.

    if save_dir is not None:
        np.save(save_dir + 'idxs_patches_from_video', idxs)
    return idxs


def update_patches(incomplete_patches=[], new_frame=None,
                   incomplete_patch_idxs=None, patch_dims=None):
    """
    ARGS
    incomplete_patches    (list): list of lists of ndarrays.
    new_frame          (ndarray): (1 x H x W x C).
    incomplete_patch_idxs (list): each entry is a list of ndarrays (h x w x c).
    patch_dims           (tuple): (l x h x w x c), where l >= 2.

    RETURNS
    complete_patches      (list): list of ndarrays (patches).
    incomplete_patches    (list): list of lists of ndarrays.
    incomplete_patch_idxs (list): list of lists of indices.
    """
    assert len(incomplete_patches) == len(incomplete_patch_idxs)
    assert new_frame is not None
    assert patch_dims is not None
 
    count = 0
    for idx in incomplete_patch_idxs:
        incomplete_patches[count].append(new_frame[idx[0]: idx[0] +
                                         patch_dims[0], idx[1]: idx[1] +
                                         patch_dims[1]])
        count += 1
 
    complete_patches = [patch for patch in incomplete_patches if \
                         len(patch) == patch_dims[2]]
    complete_patches = [np.squeeze(np.array(patch)) \
                        for patch in complete_patches]
 
    zipped = zip(incomplete_patch_idxs, incomplete_patches)
    old_idxs = incomplete_patch_idxs
    incomplete_patch_idxs = [idx for idx, patch in zipped if \
                              len(patch) < patch_dims[2]]
    complete_patch_idxs = [idx for idx in old_idxs if idx not in incomplete_patch_idxs]

    incomplete_patches = [patch for patch in incomplete_patches if \
                           len(patch) < patch_dims[2]]
 
    return complete_patches, incomplete_patches, incomplete_patch_idxs, complete_patch_idxs


def patch_list_to_ndarray(patch_list):
    patches = np.array(patch_list)
    n_patches = patches.shape[0]
    patches = patches.reshape(n_patches, 18)
    #patches += np.random.uniform(size=patches.shape)
    return patches

def grey_patch_list_to_ndarray(patch_list):
    print(len(patch_list), patch_list[0].shape)
    for x in range(len(patch_list)):
        patch_list[x] = rgb2gray(patch_list[x])
    print(len(patch_list), patch_list[0].shape)
    patches = np.array(patch_list)
    print(patches.shape)
    n_patches = patches.shape[0]
    patches = patches.reshape(n_patches, 18)
    #patches += np.random.uniform(size=patches.shape)
    patches = np.squeeze(patches)
    return patches
    

def no_transform(patch_list):
    return patch_list

# extract_patches needs to be broken up into about 4-5 functions. I will refactor later, but this will do for now since we are on a deadline. Tests are passing.
def extract_patches(mpeg_path, patch_dims, n_patches, *, n_frames=None, 
                    idxs=None, vid_dims=None,
                    batch_size=250000, patch_transform=no_transform, save_dir=None,
                    return_idxs=False):
    """ Extracts patches from an mpeg video.
    ARGS
    mpeg_path      (string): filename of mpeg video, from which patches will be
                              extracted. Video dimensions must be 
                              (F x H x W x C), where 
                               F = number of frames,
                               H = height,
                               W = width, and
                               C = number of color channels.
    patch_dims      (tuple): (f x h x w x C), C must match mpeg_path

    RETURNS
    complete_patches (list): list of ndarrays, with patch_dims dimension.
    """
    if vid_dims == None:
        vid_dims = get_mpeg_dims(mpeg_path)
    if idxs is None:
        idxs = gen_patch_idxs(vid_dims, patch_dims, n_patches,
                              n_frames=n_frames, save_dir=save_dir)

    curr_img = prev_img = []
    complete_patches = incomplete_patches = incomplete_patch_idxs = []
    frame_num = patch_num = 0
    vid_gen = skvideo.io.vreader(mpeg_path)

    batch_num = 1
    for frame in vid_gen:
        #if gray_scale is True:
           # frame = np.squeeze(rgb2gray(frame))

        # Get updated patch data.
        complete, \
        incomplete, \
        incomplete_idxs, \
        complete_idxs = update_patches(incomplete_patches, frame, incomplete_patch_idxs, patch_dims)

        # Update the patch lists.
        complete_patches += complete
        incomplete_patches = incomplete
        incomplete_patch_idxs = incomplete_idxs

        checking = True
        while checking:
            if patch_num >= idxs.shape[1]:
                checking = False
            elif frame_num != idxs[:, patch_num][2]:
                checking = False
            elif frame_num == idxs[:, patch_num][2]:
                row = idxs[:, patch_num][0]
                col = idxs[:, patch_num][1]
                incomplete_patch_idxs.append([row, col])
                incomplete_patches.append([frame[row: row + patch_dims[0],
                                                 col: col + patch_dims[1]]])
                patch_num += 1
            else:
                raise ValueError('Checking loop is broken.')
             
        frame_num += 1
        if frame_num % 100 == 0:
            print('Frame {}'.format(frame_num))
        if patch_transform is not None and save_dir is not None and complete_patches != []:
                complete_patches = patch_transform(complete_patches)
                np.save(save_dir + 'motion_patches_batch_{}'.format(frame_num - 1),
                    complete_patches)
                np.save(save_dir + 'motion_patch_vid_idxs_batch_' + 
                                    str(frame_num - patch_dims[2] + 1), 
                        np.array(complete_idxs).T)
                complete_patches = []

#        if len(complete_patches) >= batch_size:
#            print(len(complete_patches))
#            if patch_transform is not None and save_dir is not None:
#                complete_patches = patch_transform(complete_patches)
#                np.save(save_dir + 'motion_patches_batch_{}'.format(batch_num),
#                        complete_patches)
#                complete_patches = []
#            batch_num += 1
#    
#    if len(complete_patches) != 0:
#        if patch_transform is not None:
#            complete_patches = patch_transform(complete_patches)
#        if save_dir is not None:
#            np.save(save_dir + 'motion_patches_batch_{}'.format(batch_num),
#                    complete_patches)

    complete_patches = patch_transform(complete_patches)

    if return_idxs is True:
        return complete_patches, idxs
    else:
        return complete_patches


def main():
    vid_path = os.getcwd() + '/../data/input/test_video.mp4'
                           # '/../data/input/subj5-4hr.avi'
    patch_dims = (3, 3, 2)
    n_patches = 1000  # 52161000
    n_frames = 100    # 104322
    save_dir = os.getcwd() + '/../data/working/motion_patches/'
    patches = extract_patches(vid_path, patch_dims, n_patches, n_frames=n_frames,
                              patch_transform=patch_list_to_ndarray,
                              save_dir=save_dir)


if __name__ == '__main__':
    main()
