# DeepDeblur_release

Single image deblurring with deep learning.

This is a project page for our research.
Please refer to our arXiv paper for details: [arXiv](https://arxiv.org/abs/1612.02177)

## Dependencies
* torch7
* [torchx](https://github.com/nicholas-leonard/torchx)
* cudnn

## Code

To run demo, download the trained models into "experiment" folder.
[models](cv.snu.ac.kr/~snah\Deblur\DeepDeblur_models/experiment.zip)

Type following command in "code" folder.
```lua
qlua -i demo.lua -load -save scale3_adv_crf -blur_type gamma2.2
qlua -i demo.lua -load -save scale3_adv_lin -blur_type linear
```

To train a model, clone this repository and download below dataset in "dataset" directory.
Then run main.lua in 'code' folder with optional parameters.
```lua
-- Train for 450 epochs, save in 'experiment/scale3'
th main.lua -nEpochs 450 -save scale3
-- Load saved model
th main.lua -load -save scale3
> blur_dir, output_dir = ...
> deblur_dir(blur_dir, output_dir)
```
optional parameters are listed in opts.lua



## Dataset

In this work, we proposed a new dataset of realistic blurry and sharp image pairs using a high-speed camera.
However, we do not provide blur kernels. 

Statistics | Training | Test | Total 
-- | -- | -- | --
sequences | 22 | 11 | 33
image pairs | 2103 | 1111 | 3214


__*Download link*__ 

* [GOPRO_Large](http://cv.snu.ac.kr/~snah/Deblur/dataset/GOPRO_Large.zip)

[//]: # " * [GOPRO_Large_raw](http://cv.snu.ac.kr/~snah/Deblur/dataset/GOPRO_Large_raw.zip)"

Here are some examples.

Blurry image example 1
![Blurry image](images/Istanbul_blur1.png)

Sharp image example 1
![Sharp image](images/Istanbul_sharp1.png)

Blurry image example 2
![Blurry image](images/Flower_blur1.png)

Sharp image example 2
![Sharp image](images/Flower_sharp1.png)

