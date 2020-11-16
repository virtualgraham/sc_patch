# Jigsaw Patch Descriptor

Experiments with representation learning based on:

[Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)  
[Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)  
[Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)  
[Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/abs/1901.09005)  


## `src/sc_patch_b.py`
Method based [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)
Main differences from original method in paper:
- Uses VGG16 architecture ([Author's supplementarily provided VGG16 architecture](http://graphics.cs.cmu.edu/projects/deepContext/nets/vgg_style.prototxt)) instead of the original AlexNet architecture
- VGG16 includes normalization layers that author's version of VGG does not have
- Trained on Objects365 dataset instead of ImageNet dataset. Objects365 is newer and has somewhat larger resolution images.
- Skips resize image step. Most of the time the resize image preprocessing step from the original paper upscales the ImageNet images to the ranges of resolutions that Objects365 is already at. For the objective of making the network robust to pixelation, there is still the preprocessing step that resizes patches. 
- In the original method all possible pairs of patches are extracted together from a grid on each image. Here, pairs of patches are selected one at a time from a random image at a random location.

## `src/shuffle_patch_p.py`
Method based on [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)
- Uses VGG16 instead of AlexNet
- Uses the larger, higher resolution open-image dataset
- Uses 4 patches instead of 9, this has only 24 permutations so there in no need to sample from the complete set of permutations. Potentially having 4 patches may be better than 9 patches. 9 patches seems like a more difficult task than 4 patches. However because with the 9 patch method the permutations are sampled, the task can be simplified to only finding the correct orientation between any two patches. Once certainty of the orientation of any two patches is achieved, the probability that there is only a single available permutation that includes the known single orientation is extremely high. 

## `src/rotation_jigsaw.py`
- Combines 4 patch jigsaw with rotation from [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)  
- Uses Google's Open Images dataset


## `src/rotation_jigsaw_resnet50.py`
TODO: implement rotation jigsaw with ResNet50 or RevNet50 from [Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/abs/1901.09005)