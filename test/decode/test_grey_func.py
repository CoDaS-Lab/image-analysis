# # Copyright 2017 Codas Lab
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at

# #   http://www.apache.org/licenses/LICENSE-2.0

# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================


# import numpy as np
# import unittest
# import os
# from preprocess import extract_patches as ep
# import skvideo.io


# class TestGreyFunc(unittest.TestCase):
    
#     def setUp(self):
        
#         self.patch_dims = (3, 3, 2)
#         self.vid_dims = (240, 320, 101)  # height x width x n_frames
#         self.n_patches = 2

#        # self.idxs_path = os.getcwd() + '/test/test_data/'
#        # self.idxs = np.load(self.idxs_path)      
#         self.vid_dir = os.getcwd() + '/test/test_data/'
#         self.vid_name = 'test_video.mp4'
#         self.gray_vid = skvideo.io.vread(self.vid_dir + self.vid_name)
#         self.gray_vid = np.squeeze(skvideo.utils.rgb2gray(self.gray_vid))
        
#     def test_grey_func(self):
#         #new function with rgb2grey within
        
#         np.random.seed(12) 
#         grey_func = ep.extract_patches(self.vid_dir + 
#                                               self.vid_name,
#                                               self.patch_dims,
#                                               self.n_patches,
#                                              patch_transform=ep.grey_patch_list_to_ndarray)
        
#         np.random.seed(12) 
#         default_func = ep.extract_patches(self.vid_dir + 
#                                               self.vid_name,
#                                               self.patch_dims,
#                                               self.n_patches,
#                                              patch_transform=ep.grey_patch_list_to_ndarray)
        
#         #working fuction without r2b2grey  
#         '''
#         np.random.seed(12)     
#         default_func = ep.extract_patches(self.vid_dir + 
#                                               self.vid_name,
#                                               self.patch_dims,
#                                               self.n_patches,
#                                               patch_transform=ep.patch_list_to_ndarray)
        
#         '''
#         '''
#         np.random.seed(1) 
#         default_func2 = ep.extract_patches(self.vid_dir + 
#                                               self.vid_name,
#                                               self.patch_dims,
#                                               self.n_patches,
#                                               patch_transform=ep.patch_list_to_ndarray)
#         '''
        
        
        
#         #self.are_same = np.array_equal(grey_func, default_func)
#         for x in range(0, self.n_patches):
#             are_same = np.array_equal(default_func[x], grey_func[x])
#             print(default_func[x] == grey_func[x])
#             print(default_func[x])
#             print(grey_func[x])
#             print(default_func[x].shape)
#             print(grey_func[x].shape)
#             self.assertTrue(are_same)
#         #self.assertEqual(grey_func.tolist(), default_func.tolist()))
        
        
        
# if __name__ == '__main__':
#     unittest.main()
