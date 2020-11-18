# Jigsaw Patch Descriptor

Experiments with unsupervised visual representation learning based on:

[Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)  
[Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)  
[Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)  
[Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/abs/1901.09005)  


The objective here is to create an object part patch descriptor in an unsupervised manner. The patch descriptor is needed for the [fgraph](https://github.com/virtualgraham/fgraph) project.  

The patch based unsupervised visual representation learning methods used here have mostly not been tested as patch descriptors (despite it being called "representation" learning) but rather the evaluation of these methods have been mostly tested using transfer learning to traditional object classification tasks.  

The working hypothesis behind this project is that these patch based methods are well suited as local object part patch feature descriptors, potentially exceeding other local feature descriptors such as HOG and SIFT for object part representation, and are not as effective at global image descriptors. An object part model that uses patch descriptors would be more effective using cnn features produced by unsupervised representation learning than using the features from the same cnn architecture trained on a supervised classification task.  

The use of unsupervised visual representation learning as object part representations was discussed and examined early on by [Doersch et al](https://arxiv.org/abs/1505.05192). However in the followup papers expanding and improving the methods, this research angle was overshadowed by the transfer learning task.  

Transfer learning from the unsupervised to the supervised classification task may be a misleading test of the effectiveness of unsupervised visual representation learning methods. This is because the image classification task not only finds object part features but also finds relationships between object features. Patch descriptors will fall short in whole image classification tasks because they do not represent relationships between object parts, because they do not contain representations of part relationships. A good patch descriptor however does provide a good way to represent features at many scales and at higher resolutions. For example a patch descriptor can be trained on high resoluton details of objects that size constraints would prevent a cnn classification model from fitting all the high resolution parts into a single input.  


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