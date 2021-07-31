# vol-mlmr
This code is the implementation of the following paper: "Three Dimensional Root CT Segmentation using Multi-Resolution Encoder-Decoder Networks"

Instructions:

Convertin gthe stack of images to volumetric th7 data:

th createfromimages.lua 

Train a model from scratch:

th main.lua

Train from a saved model:
th main.lua -model 'snapshots/model_**.t7' -optimState 'snapshots/optimState_**.t7'

To do segmentation:

th testmodelfull_multiLoss.lua

More info to appear soon.
