# CK-CNN Deep learning based Corn Kernel classification

<!-- ```diff
- Sorry for any inconvenience, we are updating the repo
``` -->

This work presents a full pipeline to classify sample sets of corn kernels. The proposed approach follows a segmentation-classification scheme. The image segmentation is performed through a well known deep learning-based approach, the Mask R-CNN architecture, while the classification is performed through a novel-lightweight network specially designed for this task: CK-CNN ---good corn kernel, defective corn kernel and impurity categories are considered. To know more about CK-CNN, read our camera ready version in [ckcnn](http://www.cidis.espol.edu.ec/es/content/deep-learning-based-corn-kernel-classification), this paper will be presented in The 1st International Workshop and Prize Challenge on Agriculture of CVPR2020.

<div style="text-align:center"><img src='figs/ckcnn_architecture.jpg' style="width:70%; display: block;
  margin-left: auto;
  margin-right: auto;">

## Table of Contents
* [Keras](#tensorflow)
* [Datasets](#datasets)
* [Performance](#performance)
* [Citation](#citation)

# Keras

 Before starting to use this model,  there are some requirements to fullfill.
 
## Requirements

* [Python 3.7.3](https://www.python.org/downloads/release/python-370/g)
* [TensorFlow=1.14](https://www.tensorflow.org) 
* [Keras 2.3.1](https://keras.io/#installation)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Matplotlib](https://matplotlib.org/3.1.1/users/installing.html)
* [Numpy](https://numpy.org/devdocs/user/install.html)

Once the packages are installed,  clone this repo as follow: 

    git clone https://github.com/cidis/CK-CNN.git
    cd code

## Project Architecture

```
├── data                           # sample images for training and testing
|   ├── clusters                   # samples for segmentation experiment
|   └── individual                 # samples for classification experiment
├── figs                           # Images used in README.md
|   └── ckcnn_architecture.jpg     # ckcnn banner for model architecture
|   └── dataset_example.png        # dataset example banner
|   └── maskrcnn_architecture.png  # mask r-cnn banner for model architecture
├── models                         # keras model file  
|   └── weights.h5                 # weights saved for the ck-cnn test in paper
├── code                           # a series of tools used in this repo
|   └── ckcnn_2class.py            # python file with main functions and parameter settings for 2 class classification
|   └── ckcnn_3class.py            # python file with main functions and parameter settings for 3 class classification 
|   └── Prediction.py              # the script to run testing experiment and generation of statistics
```

As described above, ckcnn_2class.py and ckcnn_3class.py has the parameters settings, for training ckcnn for 2 class and 3 class classification experiments respectively, before those processes the parameters need to be set. CKCNN is trained with our proposed dataset attached uin the respective section, so make sure to change the route to the right one; however, in the testing stage (Prediction.py), any dataset can be used. However, we evaluated with our proposed "validation" dataset and the arguments have to be well referenced. If you want to experiment with our trained weights, just change the load_model function parameter and set the one is in the model directory on this repo. Pay attention in the parameters' settings, and change whatever you want. 

```
saved_model = load_model("weights.h5")
test_data_generator = test_generator.flow_from_directory(
    directory="datasetD/validation_set/validation_set", # Put your path here
    target_size=(224,224),
    #batch_size=32,
    shuffle=False)
```

# Datasets

<div style="text-align:center"><img src='figs/dataset_example.png' style="width:70%; display: block;
  margin-left: auto;
  margin-right: auto;">
 
## Dataset used for both Training and Testing

This dataset is collected and annotated in our laboratories following high quality standards for image adquisition.
See more details and download in: [Option1](http://www.cidis.espol.edu.ec/es/content/deep-learning-based-corn-kernel-classification)

# Performance

The results below are from the final version of CKCNN compared with other well-known architectures trained with our datasets. 

GCA: Good corn accuracy\
DFA: Defective corn accuracy\
IMP: Impurities accuracy\
AVG: overall accuracy\
PAR: Total number of parameters in K\

<center>

|     Network    |    GCA   |    DFA   |    IMP   |    AVG   |   PAR(K)  |
| -------------- | ---------| -------- | -------- | ---------| --------- |
| [CKCNN(ours)](https://github.com/cidis/CK-CNN/)            | `.979` | `.900` | `.973` | `.956` |    `3306` |
| [VGG-16](https://github.com/1297rohit/VGG16-In-Keras)      | `.974` | `.876` | `.819` | `.890` | `134.272` |
| [RESNET-50](https://github.com/s9xie/hed)                  | `.986` | `.860` | `.931` | `.925` |   `23593` |
| [MASK-RCNN](https://github.com/matterport/Mask_RCNN)       | `.960` | `.695` | `.286` | `.647` |   `63738` |

</center>
Evaluation performed with our provided dataset. Fine-tuning is in progress, we will update the results soon.

# Citation
Please cite our paper if you find helpful,
```
@inproceedings{corn2020ckcnn,
  title={Deep Learning based Corn Kernel Classification},
  author={Velesaca, Henry O. and Mira, Raúl and Suarez, Patricia L. and Larrea, Christian X. and Sappa, Angel D.},
  booktitle={The 1st International Workshop on Agriculture-Vision: Challenges & Opportunities for Computer Vision in Agriculture},
  pages={},
  year={2020},
  organization={}
}
```



