# Code 
<aside class="warning">
Warning: training code segfaults when run under Anaconda virtual environment. You can try to debug it, I tried but failed. To run code ask admin to install >python3.5. Other libraries you can ask admin to install, or compile and install to your home directory. Compileing by self generally involves downloading source code, `./configure --prefix=$HOME/local/ ENV=val ENV2=val2`, `make` and `make install`. This is one time annoyance(one time for each server :( ).
</aside>

`eval.py`: code that runs on runtime or validation dataset.

`eval-wmap.py`: driver that calls eval.py

`train-unet-dice.py`: train unet with dice loss function

`train-unet-dice-siou.py`: validation runs on siou metrics along with F-ARI metrics(See Report)

`train-unet-wmap.py`: train unet with weighted cross entropy

`unet.py`: original unet (with atrous convolutions)

`unet-v1.py`: unet with input channel(s) concatenated before prefinal layer.

`unet-v2.py`: work in progress. Idea is to make sure that for different perturbed versions of same image(you can view another frame as perturbation), outputs of first few layers are almost same. For training you will have to write new step function and pass it to trainer.Trainer object. Read train-unet- for more info on how to train.

`trainer.py`: contains main class. It has train-for method that runs for particular number of epochs. I have tried to make this class as generic as possible, but in certain cases you might have to make changes to it.

`scripts/augment.py`: If you have both rois and brightfield images use this script to prepare data for training. It has options to rotate, scale and split. It will generate new images with masks and boundaries(boundaries are needed by weightmap datareaders)

`scripts/ziproi_to_mask.py`: reads zipped rois (image.zip) and creates corresponding mask.

`scripts/split.py`: If rois are not available, use this script to split data.

`scripts/stitch.py`: Stiches multiple probability maps to single map.

`utils/wmap-dataset.py`: datareader for PyTorch which returns image, mask and weightmap

`utils/dataset.py`: datareader without weightmaps
`utils/wand-transforms.py`: data augmentation pipeline functions for python-wand library.

`utils/cv-transforms.py`: same as above, except when opencv is used instead of python-wand. I found that perturbations(Shepard's interpolation) from imagemagick behaved nicely than opencv. magickwand is python interface to imagemagick.

`utils/roi.py`: reader and parser for .roi files.

`utils/track.py`: simple hueristic to track cells across frames, might not work if there is too much shift. It basically keeps track of unique cells sofar. for next frame, for each cell finds one of the previous cells with maximum iou. If iou is zero, this is new cell. 

`metric/separate-iou.py`: code for sIoU from report.

### clustering

Here most work I did was on the fly. Often I wrote code in jupyter individually for each experiment. However repititive code for plotting has been written to plotting.py file. 

## Data
All the data we receive has been stored in data directory. Directory is orgainzed by subdirectories with dates as names. So data received on 28 June would be in 19-06-28. 

Within each directory there is raw folder, it contains data as received (sometimes I will keep only brightfield images, and keep other to some other folder). 

To prepare data,

1. You run 
