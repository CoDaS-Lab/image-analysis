# Image analysis project

## Core Pipeline
The pipeline is organized in a way to allow multiple operations to be defined in a few lines of code. For example:

```
motion_analysis = Pipeline(data=batch_list,
                           ops=[Normalize()],
                           seq=[RGBToGray(), DownSample()],
                           models={'SVM': SVM(), 'PCA': PCA()}
                           save_all=True)
```

The Pipeline class takes 5 inputs: the data, list of operations to run in parallel(without dependencies on each other), list of sequential operations, dictionary of models, and boolean check to save all features to a dictionary or not. The seq list contains all the operations we will run sequentially. The outputs of one operation gets fed into the following operation in the list.  

The models dictionary contains the statistical models to run on the data. The pipeline runs each model, in this case SVM(support vector machine) or PCA(principal component analysis) on the data. By using this structure you can swap out and insert operations at will.


## Load videos (function name might change)

```
batch_list = vd.decode_mpeg(vid_path + 'test_video.mp4', stride=2
                            batch_size=2, start_idx=2, end_idx=10, pad=True)
```

To load videos, use the decode_mpeg function. It loads videos into batches for easier processing. The function takes 6 parameters: the video path, the stride to load frames (ex. every 2 frames or every 3, 1 is default), the size of each batch (if it's 1 then we want every frame in the video), the start frame index and end frame index, and, a boolean check to pad the last batch if stride and batch size yield uneven last batch.

## Folder Structure

* /decode - contains stuff realated to load and saving images/videos
* /demo - contains demos
* /pipeline - contains main pipeline code and implemented  feature/model classes
* /test - contains test
* /utils - utility functions like loading images from directories