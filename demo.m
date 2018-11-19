clc;clear;close all;

urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
  'imagenet-vgg-f.mat') ;

%load the network
net = load('imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;

% load two images for comparison.
image1 = imread('peppers.png');
img1 = single(image1);
img1 = imresize(img1, net.meta.normalization.imageSize(1:2)) ;
img1 = img1 - net.meta.normalization.averageImage ;

image2 = imread('onion.png');
img2 = single(image2);
img2 = imresize(img2, net.meta.normalization.imageSize(1:2)) ;
img2 = img2 - net.meta.normalization.averageImage ;

% Run the CNN and extract the feats
res1 = vl_simplenn(net, img1) ;
res2 = vl_simplenn(net, img2) ;

feat1 = res1(14).x; % 13*13*256
feat1 = permute(feat1,[3 1 2]); %256*13*13;
mask1 = res1(15).x; % 13*13*256
mask1 = permute(mask1,[3 1 2]); % 256*13*13

feat2 = res2(14).x; % 13*13*256
feat2 = permute(feat2,[3 1 2]); %256*13*13;
mask2 = res2(15).x; % 13*13*256
mask2 = permute(mask2,[3 1 2]); % 256*13*13

encodef1 = encode_feat(feat1,mask1);

encodef2 = encode_feat(feat2,mask2);

score = compare_two(encodef1,encodef2); % score between image1 and image2, 0: not similar at all. 1: the same
