clc;clear;close all;
% This example illustrares how to compare two images using the feature
% encoding method from our paper "Only Look Once, mining distinctive
% landmarks from ConvNet for visual place recognition" by Zetao Chen,
% Fabiola Maffra, Inkyu Sa, and Margarita Chli. 

comment this line if you already downloaded the network
urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-1024.mat', ...
  'imagenet-vgg-m-1024.mat') ;

%load the network
net = load('imagenet-vgg-m-1024.mat') ;
net = vl_simplenn_tidy(net) ;

% load two images for comparison.
image1 = imread('example_images/000050.jpg');
img1 = single(image1);
img1 = imresize(img1, net.meta.normalization.imageSize(1:2)) ;
img1 = img1 - net.meta.normalization.averageImage ;

image2 = imread('example_images/000094.jpg');
img2 = single(image2);
img2 = imresize(img2, net.meta.normalization.imageSize(1:2)) ;
img2 = img2 - net.meta.normalization.averageImage ;

% Run the CNN and extract the feats
res1 = vl_simplenn(net, img1) ;
res2 = vl_simplenn(net, img2) ;

feat1 = res1(14).x; % 13*13*512
feat1 = permute(feat1,[3 1 2]); %512*13*13;
mask1 = res1(15).x; % 13*13*512
mask1 = permute(mask1,[3 1 2]); % 512*13*13

feat2 = res2(14).x; % 13*13*512
feat2 = permute(feat2,[3 1 2]); %512*13*13;
mask2 = res2(15).x; % 13*13*512
mask2 = permute(mask2,[3 1 2]); % 512*13*13

encodef1 = encode_feat(feat1,mask1);
encodef1 = encodef1';

encodef2 = encode_feat(feat2,mask2);
encodef2 = encodef2';

score = compare_two(encodef1,encodef2); % score between image1 and image2, 0: not similar at all. 1: the same.
