# Adversarial Training for Attacks on Monocular Depth Estimation Networks

Adversarial attacks are imperceptible attacks that can be performed on images to fool an ML model into predicting incorrectly.
The dissertation focuses on defending these attacks using a method of training called adversarial training, specifically on depth estimation networks on the KITTI dataset(road scene dataset).

See figure 1 below.

<img src="https://raw.githubusercontent.com/NNachiappan/mscProject/main/fig1.png" width="350">


## To train
To train the model, first you need to use the implementation from Wong et al.[1]

Then you need to replace the files in the fileToReplace directory under

'master/external_src/monodepth/src'

to train using adversarial training.

## Code Reference:
[1] A. Wong, S. Cicek, and S. Soatto. Targeted Adversarial Perturbations for Monocular Depth Prediction.
