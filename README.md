# IROS2017_OnlyLookOnce
This repository contains code for our IROS 2017 paper "Only look once, mining distinctive landmarks from convnet for visual place recognition" by Zetao Chen, Fabiola Maffra, Inkyu Sa, Margarita Chli. https://ieeexplore.ieee.org/document/8202131

## Setup
- Linux or Windows;
- Matlab 2016b (the other versions not tested);

## Getting Started
- Install MatConvNet from the http://www.vlfeat.org/matconvnet/quick/ or simply follow the following commands on the Matlab command line before executing the other script:
```
% Install and compile MatConvNet (needed once).
untar('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz') ;
cd matconvnet-1.0-beta25
run matlab/vl_compilenn ;

% Download a pre-trained CNN from the web (needed once).
urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-1024.mat', ...
  'imagenet-vgg-m-1024.mat') ;

% Setup MatConvNet.
run matlab/vl_setupnn ;
```

- Clone this repo:
```
git clone https://github.com/scutzetao/IROS2017_OnlyLookOnce.git
```

- On the matlab command line, run the following command:
```
demo.m
```








