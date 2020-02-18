class: middle, center, title-slide

# Deep Learning

Lecture 4: Computer vision

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](g.louppe@uliege.be)

???

R: Beyond classification -> e.g., UNet for segmentation

R: check https://d2l.ai/chapter_computer-vision/index.html

---

class: middle, center

Convolutional networks are now **used everywhere in vision**.

.grid[
.kol-1-2[<br><br>.width-100[![](figures/lec3/yolo.png)]<br>
Object detection<br>(Redmon et al, 2015)]
.kol-1-2[.width-70[![](figures/lec3/geometric-matching.png)]<br>
Geometric matching<br>(Rocco et al, 2017)]
]
.grid[
.kol-1-2[.width-70[![](figures/lec3/semantic-segmentation.png)]<br>
Semantic segmentation<br>(Long et al, 2015)]
.kol-1-2[.width-70[![](figures/lec3/instance-segmentation.png)]<br>
Instance segmentation<br>(He et al, 2017)]
]


---

class: middle

... but also in many other applications, including:
- speech recognition and synthesis
- natural language processing
- protein/DNA binding prediction
- or more generally, any problem *with a spatial* (or sequential) *structure*.

---


---

# Pre-trained models

- Training a model on natural images, from scratch, takes **days or weeks**.
- Many models trained on ImageNet are publicly available for download. These models can be used as *feature extractors* or for smart *initialization*.

---

class: middle

## Transfer learning

- Take a pre-trained network, remove the last layer(s) and then treat the rest of the the network as a **fixed** feature extractor.
- Train a model from these features on a new task.
- Often better than handcrafted feature extraction for natural images, or better than training from data of the new task only.

## Fine tuning

- Same as for transfer learning, but also *fine-tune* the weights of the pre-trained network by continuing backpropagation.
- All or only some of the layers can be tuned.

---

class: middle

In the case of models pre-trained on ImageNet, this often works even when input images for the new task are not photographs of objects or animals, such as biomedical images, satellite images or paintings.

<br>

.center.width-100[![](figures/lec3/feature-extractor.png)]

.footnote[Credits: Mormont et al, [Comparison of deep transfer learning strategies for digital pathology](http://hdl.handle.net/2268/222511), 2018.]