class: middle, center, title-slide

# Deep Learning

Lecture 6: Computer vision

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

---

# Today 

How to build neural networks for (some) advanced computer vision tasks.
- Classification
- Object detection
- Segmentation

---

class: middle

.width-90.center[![](figures/lec6/tasks.jpg)]

.footnote[Credits: [Aurélien Géron](https://www.oreilly.com/content/introducing-capsule-networks/), 2018.]

???

Each of these tasks requires a different neural network architecture.

---

class: middle

# Classification

A few tips when using convnets for classifying images.

---

class: middle

## Convolutional neural networks

- Convolutional neural networks combine convolution, pooling and fully connected layers.
- They achieve state-of-the-art results for **spatially structured** data, such as images, sound or text.

.center.width-110[![](figures/lec5/lenet.svg)]

.footnote[Credits: [Dive Into Deep Learning](https://d2l.ai/), 2020.]

---

class: middle

For classification,
- the activation in the output layer is a Softmax activation producing a vector $\mathbf{h} \in \bigtriangleup^C$ of probability estimates $P(Y=i|\mathbf{x})$, where $C$ is the number of classes;
- the loss function is the cross-entropy loss.

---

class: middle

## Image augmentation

The lack of data is the biggest limit to the performance of deep learning models.
- Collecting more data is usually expensive and laborious.
- Synthesizing data is complicated and may not represent the true distribution.
- **Augmenting** the data with base transformations is simple and efficient.

---

class: middle

.center.width-80[![](figures/lec6/augmentation.png)]
.center.width-85[![](figures/lec6/deepaugment.png)]

.footnote[Credits: [DeepAugment](https://github.com/barisozmen/deepaugment), 2020.]

---

class: middle

## Pre-trained models

- Training a model on natural images, from scratch, takes days or weeks.
- Many models pre-trained on large datasets are publicly available for download. These models can be used as *feature extractors* or for smart **initialization**.
- The models themselves should be considered as generic and re-usable assets.

???

Insist that this is becoming a standard practice in deep learning. Very few people train from scratch. Even fewer now with the rise of foundation models.

---

class: middle

## Transfer learning

- Take a pre-trained network, remove the last layer(s) and then treat the rest of the network as a **fixed** feature extractor.
- Train a model from these features on a new task.
- Often better than handcrafted feature extraction for natural images, or better than training from data of the new task only.

<br>
.center.width-100[![](figures/lec6/feature-extractor.png)]

.footnote[Credits: Mormont et al, [Comparison of deep transfer learning strategies for digital pathology](http://hdl.handle.net/2268/222511), 2018.]

---

class: middle

.center.width-65[![](figures/lec6/finetune.svg)]

## Fine-tuning

Same as for transfer learning, but also *fine-tune* the weights of the pre-trained network by training the whole network on the new task.

.footnote[Credits: [Dive Into Deep Learning](https://d2l.ai/), 2020.]

---

class: middle

For models pre-trained on ImageNet, transferred/fine-tuned networks usually work even when the input images are from a different domain (e.g., biomedical images, satellite images or paintings).

.center.width-75[![](figures/lec6/fine-tuning-results.png)]

.footnote[Credits: Matthia Sabatelli et al, [Deep Transfer Learning for Art Classification Problems](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Sabatelli_Deep_Transfer_Learning_for_Art_Classification_Problems_ECCVW_2018_paper.pdf), 2018.]

---

class: middle

# Object detection

---

class: middle

The simplest strategy to move from image classification to object detection is to classify local regions, at multiple scales and locations.

.center.width-80[![](figures/lec6/sliding.gif)]

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

exclude: true
class: middle

## Intersection over Union (IoU)

A standard performance indicator for object detection is to evaluate the **intersection over union** (IoU) between a predicted bounding box $\hat{B}$ and an annotated bounding box $B$,
$$\text{IoU}(B,\hat{B}) = \frac{\text{area}(B \cap \hat{B})}{\text{area}(B \cup \hat{B})}.$$

.center.width-45[![](figures/lec6/iou.png)]

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

exclude: true
class: middle 

## Mean Average Precision (mAP)

If $\text{IoU}(B,\hat{B})$ is larger than a fixed threshold (usually $\frac{1}{2}$), then the predicted bounding-box is valid (true positive) and wrong otherwise (false positive).

TP and FP values are accumulated for all thresholds on the predicted confidence.
The area under the resulting precision-recall curve is the *average precision* for the considered class.

The mean over the classes is the **mean average precision** (mAP).

.center.width-50[![](figures/lec6/interpolated_precision.png)]

.footnote[Credits: [Rafael Padilla](https://github.com/rafaelpadilla/Object-Detection-Metrics), 2018.]

???

- Precision = TP / all detections 
- Recall = TP / all ground truths

---

class: middle 

The sliding window approach evaluates a classifier at a large number of locations and scales. 

This approach is usually .bold[computationally expensive] as performance directly depends on the resolution and number of the windows fed to the classifier (the more the better, but also the more costly). 

---

# OverFeat 

.grid[
.kol-2-3[

The complexity of the sliding window approach was mitigated in the pioneer OverFeat network (Sermanet et al, 2013) by adding a **regression head** to predict the object bounding box $(x,y,w,h)$.

]
.kol-1-3[.center.width-100[![](figures/lec6/overfeat.png)]]
]

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

class: middle

.center[.width-45[![](figures/lec6/overfeat-grid.png)] .width-45[![](figures/lec6/overfeat-predictions.png)]]

For each location and scale pre-defined from a .bold[coarse] grid,
- the classifier head outputs a class and a confidence (left);
- the regression head predicts the location of the object (right).

.footnote[Credits: Sermanet et al, 2013.]

---

class: middle

.center.width-60[![](figures/lec6/overfeat-merge.png)]

These bounding boxes are finally merged by .bold[Non-Maximum Suppression] to produce the final predictions over a small number of objects.

.footnote[Credits: Sermanet et al, 2013.]

???

NMS:
1. Start with all detection boxes, each with a confidence score
2. Sort all boxes by confidence score (highest to lowest)
3. Select the box with highest confidence score, add it to the final detection list
4. Calculate Io between this box and all remaining boxes
5. Discard boxes with IoU greater than a predefined threshold (typically 0.5-0.7)
6. Repeat steps 3-5 until no boxes remain

IoU = Area of Intersection / Area of Union
(range from 0 (no overlap) to 1 (perfect overlap))

---

class: middle

The OverFeat architecture comes with several **drawbacks**:
- it is a disjoint system (2 disjoint heads with their respective losses, ad-hoc merging procedure);
- it optimizes for localization rather than detection;
- it cannot reason about global context and thus requires significant post-processing to produce coherent detections.

???

Localization is the task of predicting the bounding box of an object that is known to be present in the image, while detection is the task of predicting the bounding box of an object that may or may not be present in the image.

---

# YOLO

.center.width-65[![](figures/lec6/yolo-model.png)]

YOLO (Redmon et al, 2015) models detection as a regression problem. 

The image is divided into an $S\times S$ grid and for each grid cell predicts $B$ bounding boxes, confidence for those boxes, and $C$ class probabilities. These predictions are encoded as an $S \times S \times (5B + C)$ tensor.

.footnote[Credits: Redmon et al, 2015.]

---

class: middle

For $S=7$, $B=2$, $C=20$, the network predicts a vector of size $30$ for each cell.

.center.width-100[![](figures/lec6/yolo-architecture.png)]

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

class: middle

The network predicts class scores and bounding-box regressions, and .bold[although the output comes from fully connected layers, it has a 2D structure].

- Unlike sliding window techniques, YOLO is therefore capable of reasoning globally about the image when making predictions. 
- It sees the entire image during training and test time, so it implicitly encodes contextual information about classes as well as their appearance.

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

class: middle

During training, YOLO makes the assumptions that any of the $S\times S$ cells contains at most (the center of) a single object. We define for every image, cell index $i=1, ..., S\times S$, predicted box $j=1, ..., B$ and class index $c=1, ..., C$,
- $\mathbb{1}\_i^\text{obj}$ is $1$ if there is an object in cell $i$, and $0$ otherwise;
- $\mathbb{1}\_{i,j}^\text{obj}$ is $1$ if there is an object in cell $i$ and predicted box $j$ is the most fitting one, and $0$ otherwise;
- $p\_{i,c}$ is $1$ if there is an object of class $c$ in cell $i$, and $0$ and otherwise;
- $x\_i, y\_i, w\_i, h\_i$ the annoted bouding box (defined only if $\mathbb{1}\_i^\text{obj}=1$, and relative in location and scale to the cell);
- $c\_{i,j}$ is the IoU between the predicted box and the ground truth target.

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

class: middle

The training procedure first computes on each image the value of the $\mathbb{1}\_{i,j}^\text{obj}$'s and $c\_{i,j}$, and then does one step to minimize the multi-part loss function
.smaller2[
$$
\begin{aligned}
& \lambda\_\text{coord} \sum\_{i=1}^{S \times S} \sum\_{j=1}^B \mathbb{1}\_{i,j}^\text{obj} \left( (x\_i - \hat{x}\_{i,j})^2 + (y\_i - \hat{y}\_{i,j})^2 + (\sqrt{w\_i} - \sqrt{\hat{w}\_{i,j}})^2 + (\sqrt{h\_i} - \sqrt{\hat{h}\_{i,j}})^2\right)\\\\
& + \lambda\_\text{obj} \sum\_{i=1}^{S \times S} \sum\_{j=1}^B \mathbb{1}\_{i,j}^\text{obj} (c\_{i,j} - \hat{c}\_{i,j})^2 + \lambda\_\text{noobj} \sum\_{i=1}^{S \times S} \sum\_{j=1}^B (1-\mathbb{1}\_{i,j}^\text{obj}) \hat{c}\_{i,j}^2  \\\\
& + \lambda\_\text{classes} \sum\_{i=1}^{S \times S} \mathbb{1}\_i^\text{obj} \sum\_{c=1}^C (p\_{i,c} - \hat{p}\_{i,c})^2 
\end{aligned}
$$
]

where $\hat{p}\_{i,c}$, $\hat{x}\_{i,j}$, $\hat{y}\_{i,j}$, $\hat{w}\_{i,j}$, $\hat{h}\_{i,j}$ and $\hat{c}\_{i,j}$ are the network outputs.

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

class: middle 

Training YOLO relies on .bold[many engineering choices] that illustrate well how involved is deep learning in practice:
- pre-train the 20 first convolutional layers on ImageNet classification;
- use $448 \times 448$ input for detection, instead of $224 \times 224$;
- use Leaky ReLUs for all layers;
- dropout after the first convolutional layer;
- normalize bounding boxes parameters in $[0,1]$;
- use a quadratic loss not only for the bounding box coordinates, but also for the confidence and the class scores;
- reduce weight of large bounding boxes by using the square roots of the size in the loss;
- reduce the importance of empty cells by weighting less the confidence-related loss on them;
- data augmentation with scaling, translation and HSV transformation.

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

class: middle, center, black-slide

<iframe width="600" height="450" src="https://www.youtube.com/embed/YmbhRxQkLMg" frameborder="0" allowfullscreen></iframe>

YOLO (Redmon, 2017).

---

exclude: true
class: middle

## SSD

The Single Shot Multi-box Detector (SSD; Liu et al, 2015) improves upon YOLO by using a fully-convolutional architecture and multi-scale maps.

.center.width-80[![](figures/lec6/ssd.png)]

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

# Region-based CNNs

An alternative strategy to having a huge predefined set of box proposals is to rely on *region proposals* first extracted from the image.

The main family of architectures following this principle are **region-based** convolutional neural networks:
- (Slow) R-CNN (Girshick et al, 2014)
- Fast R-CNN (Girshick et al, 2015)
- Faster R-CNN (Ren et al, 2015)
- Mask R-CNN (He et al, 2017)

---

class: middle

## R-CNN

This architecture is made of four parts:
1. Selective search is performed on the input image to select multiple high-quality region proposals.
2. A pre-trained CNN (the **backbone**) is selected and put before the output layer. It resizes each proposed region into the input dimensions required by the network and uses a forward pass to output features for the proposals.
3. The features are fed to an SVM for predicting the class.
4. The features are fed to a linear regression model for predicting the bounding-box.

.center.width-90[![](figures/lec6/r-cnn.svg)]

.footnote[Credits: [Dive Into Deep Learning](https://d2l.ai/), 2020.]

---

class: middle

.center.width-80[![](figures/lec6/selective-search.png)]

Selective search (Uijlings et al, 2013) groups adjacent pixels of similar texture, color, or intensity by analyzing windows of different sizes in the image.

---

class: middle

.grid[
.kol-3-5[
<br><br>

## Fast R-CNN

- The main performance bottleneck of R-CNN is the need to independently extract features for each proposed region.
- Fast R-CNN uses the entire image as input to the CNN for feature extraction, rather than each proposed region.
- Fast R-CNN introduces RoI pooling for producing feature vectors of fixed size from region proposals of different sizes.

]
.kol-2-5[.width-100[![](figures/lec6/fast-rcnn.svg)]

.width-100[![](figures/lec6/fast-rcnn.png)]]
]

.footnote[Credits: [Dive Into Deep Learning](https://d2l.ai/), 2020.]

---

class: middle

.center.width-70[![](figures/lec6/faster-rcnn.svg)]

## Faster R-CNN

- The performance of both R-CNN and Fast R-CNN is tied to the quality of the region proposals from selective search.
- Faster R-CNN replaces selective search with a region proposal network.
- This network reduces the number of proposed regions generated, while ensuring precise object detection.

.footnote[Credits: [Dive Into Deep Learning](https://d2l.ai/), 2020.]

---

class: middle, center, black-slide

<iframe width="600" height="450" src="https://www.youtube.com/embed/V4P_ptn2FF4" frameborder="0" allowfullscreen></iframe>

YOLO (v2) vs YOLO 9000 vs SSD vs Faster RCNN

---

class: middle

## Takeaways

- One-stage detectors (YOLO, SSD, RetinaNet, etc) are fast for inference but are usually not the most accurate object detectors.
- Two-stage detectors (Fast R-CNN, Faster R-CNN, R-FCN, Light head R-CNN, etc) are usually slower but are often more accurate.
- All networks depend on lots of engineering decisions.

---

class: middle, center

([demo](https://colab.research.google.com/gist/kirisakow/325a557d89262e8d6a4f2918917e82b4/real-time-object-detection-in-webcam-video-stream-using-ultralytics-yolov8.ipynb))

???

Use Lucie's kitchen set.
- Far vs. near detections
- Individual vs. packed detections
- Rotation, flip, etc

---

class: middle

# Segmentation

---

class: middle

.center.width-70[![](figures/lec6/segmentation.png)]

Segmentation is the task of partitioning an image, at the pixel level, into regions:
- .bold[Semantic segmentation]: All pixels in an image are labeled with their class (e.g., car, pedestrian, road).
- .bold[Instance segmentation]: Pixels of detected objects are labeled with an instance ID (e.g., car 1, car 2, pedestrian 1).
- Panoptic segmentation: Combines semantic and instance segmentation. All pixels in an image are labeled with a class and an instance ID (if applicable).

.footnote[Credits: [Dive Into Deep Learning](https://d2l.ai/), 2020.]

---

class: middle

The deep learning approach casts semantic segmentation as pixel classification. Convolutional networks can be used for that purpose, but with a few adaptations.

---

class: middle

.center.width-100[![](figures/lec6/fcn-1.png)]

.footnote[Credits: [CS231n, Lecture 11](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture11.pdf), 2018.]

---

class: middle

.center.width-100[![](figures/lec6/fcn-2.png)]

.footnote[Credits: [CS231n, Lecture 11](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture11.pdf), 2018.]

???

Convolution and pooling layers reduce the input width and height, or keep them unchanged.

Semantic segmentation requires to predict values for each pixel, and therefore needs to increase input width and height.

Fully connected layers could be used for that purpose but would face the same limitations as before (spatial specialization, too many parameters).

Ideally, we would like layers that implement the inverse of convolutional
 and pooling layers.

---

class: middle

## Transposed convolution

A **transposed convolution** is a convolution where the implementation of the forward and backward passes
are swapped.

Given a convolutional kernel $\mathbf{u}$,
- the forward pass is implemented as $v(\mathbf{h}) = \mathbf{U}^T v(\mathbf{x})$ with appropriate reshaping, thereby effectively up-sampling an input $v(\mathbf{x})$ into a larger one;
- the backward pass is computed by multiplying the loss by $\mathbf{U}$ instead of $\mathbf{U}^T$.

???

In a regular convolution,
- the forward pass is equivalent to $v(\mathbf{h}) = \mathbf{U} v(\mathbf{x})$;
- the backward pass is computed by multiplying the loss by $\mathbf{U}^T$.

Transposed convolutions are also referred to as fractionally-strided convolutions or deconvolutions (mistakenly).

---

class: middle

.pull-right[<br><br>![](figures/lec6/no_padding_no_strides_transposed.gif)]

$$
\begin{aligned}
\mathbf{U}^T v(\mathbf{x}) &= v(\mathbf{h}) \\\\
\begin{pmatrix}
1 & 0 & 0 & 0 \\\\
4 & 1 & 0 & 0 \\\\
1 & 4 & 0 & 0 \\\\
0 & 1 & 0 & 0 \\\\
1 & 0 & 1 & 0 \\\\
4 & 1 & 4 & 1 \\\\
3 & 4 & 1 & 4 \\\\
0 & 3 & 0 & 1 \\\\
3 & 0 & 1 & 0 \\\\
3 & 3 & 4 & 1 \\\\
1 & 3 & 3 & 4 \\\\
0 & 1 & 0 & 3 \\\\
0 & 0 & 3 & 0 \\\\
0 & 0 & 3 & 3 \\\\
0 & 0 & 1 & 3 \\\\
0 & 0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
2 \\\\
1 \\\\
4 \\\\
4
\end{pmatrix} &=
\begin{pmatrix}
2 \\\\
9 \\\\
6 \\\\
1 \\\\
6 \\\\
29 \\\\
30 \\\\
7 \\\\
10 \\\\
29 \\\\
33 \\\\
13 \\\\
12 \\\\
24 \\\\
16 \\\\
4
\end{pmatrix}
\end{aligned}$$

.footnote[Credits: Dumoulin and Visin, [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285), 2016.]

---

class: middle

## Fully convolutional networks (FCNs)

.grid[
.kol-3-4[

A fully convolutional network (FCN) is a convolutional network that replaces the fully connected layers with convolutional layers and transposed convolutional layers. 

For semantic segmentation, the simplest design of a fully convolutional network consists in:
- using a (pre-trained) convolutional network for downsampling and extracting image features;
- replacing the dense layers with a  $1 \times 1$ convolution layer to  transform the number of channels into the number of categories;
- upsampling the feature map to the size of the input image by using one (or several) transposed convolution layer(s).
]
.kol-1-4[.center.width-90[![](figures/lec6/fcn.svg)]]
]

---

class: middle

Contrary to fully connected networks, the dimensions of the output of a fully convolutional network is not fixed. It directly depends on the dimensions of the input, which can be images of arbitrary sizes.

---

exclude: true
class: middle

.center.width-100[![](figures/lec6/fcn-convdeconv.png)]

.footnote[Credits: [Noh et al](https://arxiv.org/abs/1505.04366), 2015.]

---

class: middle

.center.width-100[![](figures/lec6/ConvSemSeg.svg)]

The previous .bold[encoder-decoder architecture] is a simple and effective way to perform semantic segmentation. 

However, the low-resolution representation in the middle of the network can be a bottleneck for the segmentation performance, as it must retain enough information to reconstruct the high-resolution segmentation map.

.footnote[Credits: Simon J.D. Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), 2023.]

---

class: middle

## UNet

The .bold[UNet] architecture is an encoder-decoder architecture with skip connections (usually concatenations) that directly connect the encoder and decoder layers at the same resolution. In this way, the decoder can use both
- the corresponding high-resolution features from the encoder, and
- the lower-resolution features from the previous layers.

.center.width-80[![](figures/lec6/ResidualUNet.svg)]

.footnote[Credits: Simon J.D. Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), 2023.]

???

Take the time to explain that that same architecture can be used for image to image mappings, as in some of their projects.

Insist once again on the increasing number of kernels (=out_channels) in the encoder and the decreasing number of kernels in the decoder.

Mention the final 1x1 convolution to reduce the number of channels to the number of classes.

---

class: middle

.center.width-100[![](figures/lec6/ResidualUNetResults.svg)]

.center[3d segmentation results using a UNet architecture.<br> (a) Slices of a 3d volume of a mouse cortex, (b) A UNet is used to classify voxels as either inside or outside neutrites. Connected regions are shown with different colors, (c) 5-member ensemble of UNets.]


.footnote[Credits: Simon J.D. Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), 2023.]

---

class: middle

.center[(demo)]

---

class: middle

## Mask R-CNN

.grid[
.kol-1-2[

Segmentation is a natural extension of object detection. For example, Mask R-CNN extends the Faster R-CNN model for semantic segmentation: 
- The RoI pooling layer is replaced with an RoI alignment layer. 
- It branches off to an FCN for predicting a semantic segmentation mask.
- Object detection combined with mask prediction enables *instance segmentation*.


]
.kol-1-2[.center.width-95[![](figures/lec6/mask-rcnn.svg)]]
]

.footnote[Credits: [Dive Into Deep Learning](https://d2l.ai/), 2020.]

---

class: middle

.center.width-100[![](figures/lec6/mask-rcnn-results.png)]

.footnote[Credits: [He et al](https://arxiv.org/abs/1703.06870), 2017.]

---

class: middle, center, black-slide

<iframe width="600" height="450" src="https://www.youtube.com/embed/OOT3UIXZztE" frameborder="0" allowfullscreen></iframe>

---

class: middle

It is noteworthy that for detection and segmentation, there is an heavy
re-use of large networks trained for classification.

.bold[The models themselves, as much as the source code of the algorithm that
produced them, or the training data, are generic and re-usable assets.]

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

class: end-slide, center
count: false

The end.

???

Quiz:
- What architecture would you use on images?
- Would you train from scratch?
- What is the difference between object detection and segmentation?
- Name one architecture for object detection.
- Name one architecture for semantic segmentation.
- What kind of layer can you use to upscale a feature map?