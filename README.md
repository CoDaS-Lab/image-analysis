# imageanalysis

### Optional Dependencies (Demos ect.)
*  matplotlib >= 2.0.0


### Python Dependencies
* numpy >= 1.12.0
* sk-video >=1.17
* scikit-image >=0.12.0
* python 3.5
* pyfftw

### External Dependencies
* ffmpeg >=3.2.2 or libav (10 or 11)
* fftw >= 3

## Installation

### Requirements (OSX)

Installing ffmpeg:
```
brew install ffmpeg
```

### Requirements (Ubuntu)
Installing ffmpeg:
```
sudo apt-get install ffmpeg
```
Installing fftw:
``` 
sudo sudo apt-get install fftw3 libfftw3-dev libfftw3-doc
```

### Requirements (Windows)
Installing ffmpeg:  
```
1. Download a static build from [here](http://ffmpeg.zeranoe.com/builds/).
2. unpack it in the folder of your choice.
3. [Open a command prompt with administrator's rights](Just-Enough-Command-Line-for-Installing).
4. Run the command: setx /M PATH "path\to\ffmpeg\bin;%PATH%"
Be sure to alter the command so that ``path\to`` reflects the folder path from your root to ``ffmpeg\bin``.  
```

Installing sk-video:  
pip version of sk-video doesn't have support for windows yet! Download the github lastest version which was recently updated for windows:
```
https://github.com/scikit-video/scikit-video
```
and follow the dependecies and installation instructions (github installation)

## Testing
change directory to directory of test folder:
```
cd /path/to/directory/image-analysis/imageanalysis
```

Run unittest discover
```
python -m unittest discover
```

Run a single test case
```
python -m unittest test.module_name.test_file_name
```

Or for a specific function
```
python -m unittest test.module_name.test_file_name.test_class_name.test_function
```

## Overview and Tutorial
[overview](https://github.com/CoDaS-Lab/image_analysis/blob/anderson/overview.md)