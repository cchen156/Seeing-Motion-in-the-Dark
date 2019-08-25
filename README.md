# Seeing Motion in the Dark

This is a Tensorflow implementation of Seeing Motion in the Dark in ICCV 2019, by [Chen Chen](http://cchen156.web.engr.illinois.edu/), [Qifeng Chen](http://cqf.io/), [Minh N. Do](http://minhdo.ece.illinois.edu), and [Vladlen Koltun](http://vladlen.info/).  

[Project Website](http://cchen156.web.engr.illinois.edu/SMID.html)<br/>
[Paper](http://vladlen.info/papers/DRV.pdf)<br/>


## Demo Video


## Setup

### Requirement
Required python (version 2.7) libraries: Tensorflow (>=1.8.0) + Scipy + Numpy + Rawpy.

Tested in Ubuntu 16.04 + Nvidia Tesla V100 32 GB with Cuda (>=9.0) and CuDNN (>=7.1). CPU mode should also work with minor changes but not tested.

### Testing and training the models

To reproduce the results, you only need to download the [pre-processed data]() and put them in ```./DRV```. To try the [pre-trained model](), download the model in ```./checkpoint```.

To retrain a new model, run: 
```
python download_VGG_models.py
python train.py
```

To generate the 5th frame of each video, run 
```
python test_image.py
```

To generate the videos, run 
```
python test_video.py`
```

By default, the code takes the data in the "./DRV/" and the output folder is ```./result```.


### Original sensor raw data

The original raw data is much larger. If you need to process the data in a different way, you can download the [sensor raw data]() here.

## Citation
If you use our code and dataset for research, please cite our paper:

Chen Chen, Qifeng Chen, Minh N. Do, and Vladlen Koltun, "Seeing Motion in the Dark", in ICCV, 2019.

### License
MIT License.

## FAQ
1. Can I test my own data using the provided model? 

The proposed method is designed for sensor raw data. The pretrained model probably not work for data from another camera sensor. We do not have support for other camera data. It also does not work for images after camera ISP, i.e., the JPG or PNG data.

2. Will this be in any product?

This is a research project and a prototype to prove a concept. 

3. How can I train the model using my own raw data? 

Generally, you will need to pre-process your data in a similar way. That is black level subtraction, packing, applying target gain and run some pre-defined temporal filters. The test data should be pre-processed in the same way.


## Questions
If you have addtional questions after reading the FAQ, please email to cchen156@illinois.edu.

