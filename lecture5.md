class: middle, center, title-slide

# Deep Learning

Lecture 5: Recurrent neural networks

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](g.louppe@uliege.be)

---

# Today

How to deal we sequential data?

- Recurrent neural networks
- Gating
- Machine translation
- Differentiable computers

---

class: middle

Many real-world problems require to process a signal with a **sequence** structure.

- Sequence classification:
    - sentiment analysis
    - activity/action recognition
    - DNA sequence classification
    - action selection
- Sequence synthesis:
    - text synthesis
    - music synthesis
    - motion synthesis
- Sequence-to-sequence translation:
    - speech recognition
    - text translation
    - part-of-speech tagging

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

class: middle

Given a set $\mathcal{X}$, if $S(\mathcal{X})$ denotes the set of sequences of elements from $\mathcal{X}$,
$$S(\mathcal{X}) = \cup\_{t=1}^\infty \mathcal{X}^t,$$
then we formally define:

.grid.center[
.kol-1-2.bold[Sequence classification]
.kol-1-2[$f: S(\mathcal{X}) \to \\\\{ 1, ..., C\\\\}$]
]
.grid.center[
.kol-1-2.bold[Sequence synthesis]
.kol-1-2[$f: \mathbb{R}^d \to S(\mathcal{X})$]
]
.grid.center[
.kol-1-2.bold[Sequence-to-sequence translation]
.kol-1-2[$f: S(\mathcal{X}) \to S(\mathcal{Y})$]
]

<br>
In the rest of the slides, we consider only time-indexed signal, although it generalizes to arbitrary sequences.

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

# Temporal convolutions

One of the simplest approach to sequence processing is to use **temporal convolutional networks** (TCNs).
- TCNs correspond to standard 1D convolutional networks.
- They process input sequences as sequences of the maximum possible length.

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

class: middle

.center.width-80[![](figures/lec5/tcn.png)]

- Increasing exponentially the window size $T$ makes the required number of layers grow as $O(\log T)$.
- Thanks to dilated convolutions, the model size is $O(\log T)$.
- The memory footprint put and computation are $O(T \log T)$.

.footnote[Credits: Francois Fleuret, [EE559 Deep Learning](https://fleuret.org/ee559/), EPFL.]

---

class: middle

.center.width-100[![](figures/lec5/tcn-results.png)]


.footnote[Credits: Bai et al, [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271), 2018.]

---

class: middle

# Recurrent neural networks

---


---

bptt

---

class: middle

# Gating

---

# Gating

---

# LSTM

---

# GRU

---

class:  middle

# Machine translation

---

xxx

---

class: middle

# Differentiable computers

---

# Turing completeness

---

# DNC

???

https://openreview.net/pdf?id=HyGEM3C9KQ

---

class: end-slide, center
count: false

The end.

---

# References

xxx
