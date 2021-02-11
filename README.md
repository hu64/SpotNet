# SpotNet
Repository for the paper SpotNet: Self-Attention Multi-Task Network for Object Detection
<br> by Hughes Perreault<sup>1</sup>, Guillaume-Alexandre Bilodeau<sup>1</sup>, Nicolas Saunier<sup>1</sup> and Maguelonne Héritier<sup>2</sup>.
<br>
<sup>1</sup> Polytechnique Montréal
<sup>2</sup> Genetec <br>
Paper: https://arxiv.org/abs/2002.05540 <br>
Video: https://www.youtube.com/watch?v=JatQ-lziHO4

## Abstract
Humans are very good at directing their visual attention toward relevant areas when they search for different types of objects. For instance, when we search for cars, we will look at the streets, not at the top of buildings. The motivation of this paper is to train a network to do the same via a multi-task learning approach. To train visual attention, we produce foreground/background segmentation labels in a semi-supervised way, using background subtraction or optical flow. Using these labels, we train an object detection model to produce foreground/background segmentation maps as well as bounding boxes while sharing most model parameters. We use those segmentation maps inside the network as a self-attention mechanism to weight the feature map used to produce the bounding boxes, decreasing the signal of non-relevant areas. We show that by using this method, we obtain a significant mAP improvement on two traffic surveillance datasets, with state-of-the-art results on both UA-DETRAC and UAVDT.

## Model
![Model](imgs/SpotNet.png "")

Overview of SpotNet: the input image first passes through a double-stacked hourglass network; the segmentation head then produces an attention map that multiplies the final feature map of the backbone network; the final center keypoint heatmap is then produced as well as the size and coordinate offset regressions for each object.

## Organization of the Repository
* object detection: code used to perform object detection. Mainly borrowed from [CenterNet](https://github.com/xingyizhou/CenterNet), please refer to their repo for installation instructions.
* semi-supervised segmentation: code used to produce our semi-supervised ground-truth.
* results: our results for the two evaluated datasets.
* CRV 2020 conference material: poster and presentation for the CRV 2020 conference.
* imgs: images used in the repo.

## Acknowledgements
The code for this paper is mainly built upon [CenterNet](https://github.com/xingyizhou/CenterNet), we would therefore like to thank the authors for providing the source code of their paper. We also acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), [RDCPJ 508883 - 17], and the support of Genetec.
