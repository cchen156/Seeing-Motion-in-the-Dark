# Seeing Motion in the Dark 

This is a Tensorflow implementation of Seeing Motion in the Dark in ICCV 2019, by [Chen Chen](http://cchen156.web.engr.illinois.edu/), [Qifeng Chen](http://cqf.io/), [Minh N. Do](http://minhdo.ece.illinois.edu), and [Vladlen Koltun](http://vladlen.info/).  

[Paper](http://vladlen.info/papers/DRV.pdf)<br/>


## Demo Video

https://youtu.be/YeaHVrPLSro

## Code and Data

### Requirement
Required python (version 2.7) libraries: Tensorflow (1.8.0) + Scipy + Numpy + Rawpy + OpenCV (4.1.0).

Tested in Ubuntu 16.04 + Nvidia Tesla V100 32 GB with Cuda (>=9.0) and CuDNN (>=7.1). CPU mode should also work with minor changes but not tested.

### Testing and training the models

To reproduce the results, you only need to download the pre-processed data: [long](https://storage.googleapis.com/isl-datasets/DRV/long.zip), [short](https://storage.googleapis.com/isl-datasets/DRV/VBM4D_rawRGB.zip), and put them in ```./DRV```. To try the [pre-trained model](https://drive.google.com/drive/folders/1OO97dDJp9GlqijGfcvKpas2PZB0q9o8n?usp=sharing), download the model in ```./checkpoint```.

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
python test_video.py
```

By default, the code takes the data in the ```./DRV/``` and the output folder is ```./result```.


### Original sensor raw data

The original raw data is much larger. If you need to process the data in a different way, you can download the camera output raw data here: [Part1](https://storage.googleapis.com/isl-datasets/DRV/short1.zip), [Part2](https://storage.googleapis.com/isl-datasets/DRV/short2.zip), [Part3](https://storage.googleapis.com/isl-datasets/DRV/short3.zip), [Part4](https://storage.googleapis.com/isl-datasets/DRV/short4.zip), [Part5](https://storage.googleapis.com/isl-datasets/DRV/short5.zip) and [long](https://storage.googleapis.com/isl-datasets/DRV/long.zip).

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

4. What if my GPU memory is too small to train model?  

We provided a `pretrain_on_small.py` for small memory GPUs. After the training on small resolution, you will need to finetune it CPU using the `train.py`. 


## Questions
If you have addtional questions after reading the FAQ, please email to cchen156@illinois.edu.

