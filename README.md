# sc_patch

Experiments with representation learning based on:

https://github.com/cdoersch/deepcontext
and
https://github.com/abhisheksambyal/Self-supervised-learning-by-context-prediction

```
@inproceedings{doersch2015unsupervised,
    Author = {Doersch, Carl and Gupta, Abhinav and Efros, Alexei A.},
    Title = {Unsupervised Visual Representation Learning by Context Prediction},
    Booktitle = {International Conference on Computer Vision ({ICCV})},
    Year = {2015}
}
```

## `src/sc_patch_b.py`
Main differences from original method in paper:
- Uses VGG16 architecture ([Author's supplementarily provided VGG16 architecture](http://graphics.cs.cmu.edu/projects/deepContext/nets/vgg_style.prototxt)) instead of the original AlexNet architecture
- VGG16 includes normalization layers that author's version of VGG does not have
- Trained on Objects365 dataset instead of ImageNet dataset. Objects365 is newer and has somewhat larger resolution images.
- Skips resize image step. Most of the time the resize image preprocessing step from the original paper upscales the ImageNet images to the ranges of resolutions that Objects365 is already at. For the objective of making the network robust to pixelation, there is still the preprocessing step that resizes patches. 
- In the original method all possible pairs of patches are extracted together from a grid on each image. Here, pairs of patches are selected one at a time from a random image at a random location.

## `src/shuffle_patch_a.py`
- Instead of finding the relative orientation between two patches, this network finds the correct arrangement of 4 shuffled tiled patches. The drawback is that the first fully connected layer is much larger on this network than the original because it concatenates 4 networks instead of 2. The benefit is that this provides more context information to the network. Unlike the task of the original network, this task is something a person could probably do fairly well at.