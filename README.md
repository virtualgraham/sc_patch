# Jigsaw Patch Descriptor

Experiments with representation learning based on:

[Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192) 
[Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)  
```

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
- Uses 4 patches instead of 9, this has only 24 permutations so there in no need to sample from the complete set of permutations. Potentially having 4 patches and larger images could be better than 9 patches with smaller images. With the original method with 9 patches there is a hidden assumption that each image has a single subject. Most of the sample pictures have a cat or a car thats fills up the frame. Even though this assumption is often false, the 9 patches are meant to represent parts of a single subject. This is not neccerilly the case in more complex scenes without well framed subjects. With 4 patches and larger images, the patches are all neighbors in a smaller region. So even if the image is composed of multiple objects, the relationships between adjacent parts may be better represented. 
