class: middle, center, title-slide

# Deep Learning

Lecture 1: Fundamentals of machine learning

<br><br>

.bold[Gilles Louppe]<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

???

R: add train/test datasets discussion

---

# Course outline

Theory:
- Lecture 1: Fundamentals of machine learning
- Lecture 2: Neural networks
- Lecture 3: Convolutional networks
- Lecture 4: Adversarial attacks and defenses
- Lecture 5: Variational auto-encoders
- Lecture 6: Generative adversarial networks

Practice:
- Building and training neural networks with PyTorch

---

# Outline

Goal: Set the fundamentals of machine learning.

- Why learning?
- Learning from data
- Empirical risk minimization
- Bias-variance dilemma

---

class: middle

# Why learning?

---

The automatic extraction of **semantic information** from raw signal is at the
core of many applications (e.g., object recognition, speech processing, natural
language processing, planning, etc).

Can we write a computer program that does that?

<br>

.center[![](figures/lec1/mushrooms.png)]

.center[(ImageNet)]

---

class: middle

The (human) brain is so good at interpreting visual information that the gap between raw
data and its semantic interpretation is difficult to assess intuitively:

<br>

.center[
![](figures/lec1/mushroom-small.png)

This is a mushroom.
]

---

class: middle

.center.width-70[![](figures/lec1/mushroom-big.png)]

.center[This is a mushroom.]

---

class: middle

.center.width-100[![](figures/lec1/mushroom-small-nb.png)]

.center[This is a mushroom.]

---

Extracting semantic information requires models of **high complexity**.
Therefore one cannot write by hand a computer program that reproduces this process.

However, one can write a program that **learns** the task of extracting semantic information. A common strategy to solve this issue consists in:
- defining a parametric model with high capacity,
- optimizing its parameters, by "making it work" on the training data.

<br>
.center.width-50[![](figures/lec1/knobs.jpg)]

.center[Learning $\approx$ tuning the many parameters of a model.]

---

class: middle

# Learning from data

---

# Data generative model

Consider an unknown joint probability distribution $P(X,Y)$ over observations or values of interest.

Assume training data drawn from this distribution:
$$(\mathbf{x}\_i,y\_i) \sim P(X,Y),$$
with $\mathbf{x}\_i \in \mathcal{X}$, $y \in \mathcal{Y}$, $i=1, ..., N$.

- In most cases,
    - $\mathbf{x}\_i$ is a $p$-dimensional vector of *features* or *descriptors*,
    - $y$ is a scalar (e.g., a category or a real value).
- The training data is generated i.i.d.
- The training data can be of any finite size $N$.
- In general, we do not have any prior information about $P(X,Y)$.

---

class: middle

.center.width-70[![](figures/lec1/data-generation.png)]

.center[Probability space and random variables interpretation<br> of the data generative process.

$(\mathbf{x}\_i, y\_i) \sim P(X,Y) \Leftrightarrow \omega\_i \sim P\_\Omega$, $\mathbf{x}\_i = X(\omega\_i)$, $y\_i = Y(\omega\_i)$
]

---

class: middle

Alternatively, the joint distribution can be interpreted as a two-step generative process such that
 $$P(X,Y) = P(X|Y)P(Y) = P(Y|X)P(X)$$
where
- for $P(X|Y)P(Y)$:
    - first, we draw $y\sim P(Y)$
    - then, generate $\mathbf{x} \sim P(X|Y=y)$.
- for $P(Y|X)P(X)$:
    - first, we draw $\mathbf{x} \sim P(X)$
    - then, generate $y \sim P(Y|X=\mathbf{x})$.

---

class: middle

.center.width-60[![](figures/lec1/data-generation-two.png)]

.center[Two-step generative interpretation of $P(X,Y)$ as $P(X)P(Y|X)$.]

.center[e.g., $\mathbf{x} \sim P(X), y = f(\mathbf{x}) + \epsilon$ for $\epsilon \sim \mathcal{N}$.]

---

# Inference

In supervised learning, we are usually interested in the two following inference problems:
- **Classification**:<br>
Given $(\mathbf{x}, y) \in \mathcal{X}\times\mathcal{Y} = \mathbb{R}^p \times \\\{1, ..., C\\\}$,
we want to estimate $$\arg \max\_y P(Y=y|X=\mathbf{x}).$$
- **Regression**:<br>
Given $(\mathbf{x}, y) \in \mathcal{X}\times\mathcal{Y} =  \mathbb{R}^p \times \mathbb{R}$,
we want to estimate $$\mathbb{E}\left[ Y|X=\mathbf{x} \right].$$

---

class: middle, center

![](figures/lec1/classification.png)

Classification consists in identifying<br>
a decision boundary between objects of distinct classes.

---

class: middle, center

![](figures/lec1/regression.png)

Regression aims at estimating relationships among variables.

---

class: middle

The boundary between these inference problems is fuzzy, as one often reduces to the other.

- Regression enables classification through class scores.
- Classification can be viewed as discretized regression.

These inference problems also closely relate to the more general (conditional) density estimation problem.

---

# Empirical risk minimization

Consider a function $f : \mathcal{X} \to \mathcal{Y}$ produced by some learning algorithm. The predictions
of this function can be evaluated through a loss
$$\ell : \mathcal{Y} \times  \mathcal{Y} \to \mathbb{R}$$
such that $\ell(y, f(\mathbf{x})) \geq 0$ measures how close is the prediction $f(\mathbf{x})$ from $y$.

For example,
- for classification: $$\ell(y,f(\mathbf{x})) = \mathbf{1}\_{y \neq f(\mathbf{x})}$$
- for regression: $$\ell(y,f(\mathbf{x})) = (y - f(\mathbf{x}))^2$$

---

class: middle

Let us denote as $\mathcal{F}$ the hypothesis space, i.e. the set of all functions $f$ than can be produced by the chosen learning algorithm.

We are looking for a function $f \in \mathcal{F}$ with a small **expected risk** (or generalization error)
$$R(f) = \mathbb{E}\_{(\mathbf{x},y)\sim P(X,Y)}\left[ \ell(y, f(\mathbf{x})) \right].$$

This means that for a given data generating distribution and for a given hypothesis space,
the optimal model is
$$f\_\* = \arg \min\_{f \in \mathcal{F}} R(f).$$

---

class: middle

Unfortunately, since $P(X,Y)$ is unknown, the expected risk cannot be evaluated and the optimal
model cannot be determined.

However, given training data $\mathbf{d} = \\\{(\mathbf{x}\_i, y\_i) | i=1,\ldots,N\\\}$, we can
compute an estimate, the **empirical risk** (or training error)
$$\hat{R}(f, \mathbf{d}) = \frac{1}{N} \sum\_{(\mathbf{x}\_i, y\_i) \in \mathbf{d}} \ell(y\_i, f(\mathbf{x}\_i)).$$

This estimate can be used for finding a good enough approximation of $f\_\*$, giving rise
to the **empirical risk minimization principle**:
$$f\_\*^{\mathbf{d}} = \arg \min\_{f \in \mathcal{F}} \hat{R}(f, \mathbf{d})$$

---

class: middle

Most machine learning algorithms, including neural networks, implement empirical risk minimization.

Under regularity assumptions, empirical risk minimizers converge:

$$\lim\_{N \to \infty} f\_\*^{\mathbf{d}} = f\_\*$$

???

This is why tuning the parameters of the model to make it work on the training data is a reasonable thing to do.

---

# Polynomial regression

.center[![](figures/lec1/data.png)]

Consider the joint probability distribution $P(X,Y)$ induced by the data generating
process
$$x,y \sim P(X,Y) \Leftrightarrow x \sim U[-10;10], \epsilon \sim \mathcal{N}(0, \sigma^2), y = g(x) + \epsilon$$
where $x \in \mathbb{R}$, $y\in\mathbb{R}$ and $g$ is an unknown polynomial of degree 3.

---

class: middle

Our goal is to find a function $f$ that makes good predictions on average over $P(X,Y)$.

Consider the hypothesis space $f \in \mathcal{F}$ of polynomials of degree 3 defined through their parameters $\mathbf{w} \in \mathbb{R}^4$ such that
$$\hat{y} \triangleq f(x; \mathbf{w}) = \sum\_{d=0}^3 w\_d x^d$$  

---

class: middle

For this regression problem, we use the squared error loss
$$\ell(y, f(x;\mathbf{w})) = (y - f(x;\mathbf{w}))^2$$
to measure how wrong are the predictions.

Therefore, our goal is to find the best value $\mathbf{w}\_\*$ such
$$\begin{aligned}
\mathbf{w}\_\* &= \arg\min\_\mathbf{w} R(\mathbf{w}) \\\\
&= \arg\min\_\mathbf{w}  \mathbb{E}\_{(x,y)\sim P(X,Y)}\left[ (y-f(x;\mathbf{w}))^2 \right]
\end{aligned}$$

---

class: middle

Given a large enough training set $\mathbf{d} = \\\{(x\_i, y\_i) | i=1,\ldots,N\\\}$, the
empirical risk minimization principle tells us that a good estimate $\mathbf{w}\_\*^{\mathbf{d}}$ of $\mathbf{w}\_\*$ can be found by minimizing the empirical risk:
$$\begin{aligned}
\mathbf{w}\_\*^{\mathbf{d}} &= \arg\min\_\mathbf{w} \hat{R}(\mathbf{w},\mathbf{d}) \\\\
&= \arg\min\_\mathbf{w} \frac{1}{N}  \sum\_{(x\_i, y\_i) \in \mathbf{d}} (y\_i - f(x\_i;\mathbf{w}))^2 \\\\
&= \arg\min\_\mathbf{w} \frac{1}{N}  \sum\_{(x\_i, y\_i) \in \mathbf{d}} (y\_i - \sum\_{d=0}^3 w\_d x\_i^d)^2 \\\\
&= \arg\min\_\mathbf{w} \frac{1}{N} \left\lVert
\underbrace{\begin{pmatrix}
y\_1 \\\\
y\_2 \\\\
\ldots \\\\
y\_N
\end{pmatrix}}\_{\mathbf{y}} -
\underbrace{\begin{pmatrix}
x\_1^0 \ldots x\_1^3 \\\\
x\_2^0 \ldots x\_2^3 \\\\
\ldots \\\\
x\_N^0 \ldots x\_N^3
\end{pmatrix}}\_{\mathbf{X}}
\begin{pmatrix}
w\_0 \\\\
w\_1 \\\\
w\_2 \\\\
w\_3
\end{pmatrix}
\right\rVert^2
\end{aligned}$$

---

class: middle

This is **ordinary least squares** regression, for which the solution is known analytically:
$$\mathbf{w}\_\*^{\mathbf{d}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

.center[![](figures/lec1/poly-3.png)]

---

class: middle

The expected risk minimizer within our hypothesis space is $g$ itself.

Therefore, on this toy problem, we can verify that
$f(x;\mathbf{w}\_\*^{\mathbf{d}}) \to f(x;\mathbf{w}\_\*) = g(x)$ as $N \to \infty$.

---

class: middle

.center[![](figures/lec1/poly-N-5.png)]

---

class: middle
count: false

.center[![](figures/lec1/poly-N-10.png)]

---

class: middle
count: false

.center[![](figures/lec1/poly-N-50.png)]

---

class: middle
count: false

.center[![](figures/lec1/poly-N-100.png)]

---

class: middle
count: false

.center[![](figures/lec1/poly-N-500.png)]

---

class: middle

# Under-fitting and over-fitting

---

# Under-fitting and over-fitting

What if we consider a hypothesis space $\mathcal{F}$ in which candidate functions $f$ are either too "simple" or too "complex" with respect to the true data generating process?

---

class: middle

.center[![](figures/lec1/poly-1.png)]

.center[$\mathcal{F}$ = polynomials of degree 1]

---

class: middle
count: false

.center[![](figures/lec1/poly-2.png)]

.center[$\mathcal{F}$ = polynomials of degree 2]

---

class: middle
count: false

.center[![](figures/lec1/poly-3.png)]

.center[$\mathcal{F}$ = polynomials of degree 3]

---

class: middle
count: false

.center[![](figures/lec1/poly-4.png)]

.center[$\mathcal{F}$ = polynomials of degree 4]

---

class: middle
count: false

.center[![](figures/lec1/poly-5.png)]

.center[$\mathcal{F}$ = polynomials of degree 5]

---

class: middle
count: false

.center[![](figures/lec1/poly-10.png)]

.center[$\mathcal{F}$ = polynomials of degree 10]

---

class: middle

Let $\mathcal{Y}^{\mathcal X}$ be the set of all functions $f : \mathcal{X} \to \mathcal{Y}$.

We define the **Bayes risk** as the minimal expected risk over all possible functions,
$$R\_B = \min\_{f \in \mathcal{Y}^{\mathcal X}} R(f),$$
and call **Bayes model** the model $f_B$ that achieves this minimum.

No model $f$ can perform better than $f\_B$.

---

class: middle

The **capacity** of an hypothesis space induced by a learning algorithm intuitively represents the ability to
find a good model $f \in \mathcal{F}$ for any function, regardless of its complexity.

- If the capacity of $\mathcal{F}$ is low, then $f\_B \notin \mathcal{F}$ and $R(f) - R\_B$ is large for any $f \in \mathcal{F}$, including $f\_\*$ and $f\_\*^{\mathbf{d}}$. Such models $f$ are said to **underfit** the data.
- If the capacity of $\mathcal{F}$  is high, then $f\_B \in \mathcal{F}$ or $R(f\_\*) - R\_B$ is small.<br>
However, because of the high capacity of the hypothesis space, the empirical risk minimizer $f\_\*^{\mathbf{d}}$ could fit the training data arbitrarily well such that $$R(f) \geq R\_B \geq \hat{R}(f\_\*^{\mathbf{d}}, \mathbf{d}) \geq 0.$$
In this situation, $f\_\*^{\mathbf{d}}$ becomes too complex with respect to the true data generating process and a large reduction of the empirical risk (often) comes at the price
of an increase of the  expected risk of the empirical risk minimizer $R(f\_\*^{\mathbf{d}})$.
In this situation, $f\_\*^{\mathbf{d}}$ is said to **overfit** the data.

---

class: middle

Therefore, our goal is to adjust the capacity of the hypothesis space such that
the expected risk of the empirical risk minimizer gets as low as possible.

.center[![](figures/lec1/underoverfitting.png)]

---

class: middle

In practice, the capacity of the hypothesis space can be controlled through hyper-parameters of the learning algorithm. For example:
- The degree of polynomials;
- The number of layers in a neural network;
- The number of training iterations;
- Regularization terms.

---

class: middle


When overfitting,
$$R(f) \geq R\_B \geq \hat{R}(f\_\*^{\mathbf{d}}, \mathbf{d}) \geq 0.$$

This indicates that the empirical risk $\hat{R}(f\_\*^{\mathbf{d}}, \mathbf{d})$ is a poor estimator of the expected risk $R(f\_\*^{\mathbf{d}})$.

Nevertheless, an unbiased estimate of the expected risk can be obtained by evaluating $f\_\*^{\mathbf{d}}$ on data $\mathbf{d}\_\text{test}$ independent from the training samples $\mathbf{d}$:
$$\hat{R}(f\_\*^{\mathbf{d}}, \mathbf{d}\_\text{test}) =  \frac{1}{N} \sum\_{(\mathbf{x}\_i, y\_i) \in \mathbf{d}\_\text{test}} \ell(y\_i, f\_\*^{\mathbf{d}}(\mathbf{x}\_i))$$

This **test error** estimate can be used to evaluate the actual performance of model. However, it should not be used, at the same time, for model selection.

---

# Bias-variance decomposition

Consider a fixed point $x$ and the prediction $\hat{Y}=f\_*^\mathbf{d}(x)$ of the empirical risk minimizer at $x$.

Then the local expected risk of $f\_\*^{\mathbf{d}}$ is
$$\begin{aligned}
R(f\_\*^{\mathbf{d}}|x) &= \mathbb{E}\_{y \sim P(Y|x)} \left[ (y - f\_\*^{\mathbf{d}}(x))^2 \right] \\\\
&= \mathbb{E}\_{y \sim P(Y|x)} \left[ (y - f\_B(x) + f\_B(x) - f\_\*^{\mathbf{d}}(x))^2 \right]  \\\\
&= \mathbb{E}\_{y \sim P(Y|x)} \left[ (y - f\_B(x))^2 \right] + \mathbb{E}\_{y \sim P(Y|x)} \left[ (f\_B(x) - f\_\*^{\mathbf{d}}(x))^2 \right] \\\\
&= R(f\_B|x) + (f\_B(x) - f\_\*^{\mathbf{d}}(x))^2
\end{aligned}$$
where
- $R(f\_B|x)$ is the local expected risk of the Bayes model. This term cannot be reduced.
- $(f\_B(x) - f\_\*^{\mathbf{d}}(x))^2$ represents the discrepancy between $f\_B$ and $f\_\*^{\mathbf{d}}$.

---

class: middle

If $\mathbf{d} \sim P(X,Y)$ is itself considered as a random variable, then $f\_*^\mathbf{d}$ is also a random variable, along with its predictions $\hat{Y}$.

---

class: middle

.center[![](figures/lec1/poly-avg-degree-1.png)]

---

class: middle
count: false

.center[![](figures/lec1/poly-avg-degree-2.png)]

---

class: middle
count: false

.center[![](figures/lec1/poly-avg-degree-3.png)]

---

class: middle
count: false

.center[![](figures/lec1/poly-avg-degree-4.png)]

---

class: middle
count: false

.center[![](figures/lec1/poly-avg-degree-5.png)]

---

class: middle

Formally, the expected local expected risk yields to:
$$\begin{aligned}
&\mathbb{E}\_\mathbf{d} \left[ R(f\_\*^{\mathbf{d}}|x) \right] \\\\
&= \mathbb{E}\_\mathbf{d} \left[ R(f\_B|x) + (f\_B(x) - f\_\*^{\mathbf{d}}(x))^2 \right]  \\\\
&=  R(f\_B|x) + \mathbb{E}\_\mathbf{d} \left[ (f\_B(x) - f\_\*^{\mathbf{d}}(x))^2 \right] \\\\
&= \underbrace{R(f\_B|x)}\_{\text{noise}(x)} + \underbrace{(f\_B(x) - \mathbb{E}\_\mathbf{d}\left[ f\_\*^\mathbf{d}(x) \right] )^2}\_{\text{bias}^2(x)}  + \underbrace{\mathbb{E}\_\mathbf{d}\left[ ( \mathbb{E}\_\mathbf{d}\left[ f\_\*^\mathbf{d}(x) \right] - f\_\*^\mathbf{d}(x))^2 \right]}\_{\text{var}(x)}
\end{aligned}$$

This decomposition is known as the **bias-variance** decomposition.
- The noise term quantities the irreducible part of the expected risk.
- The bias term measures the discrepancy between the average model and the Bayes model.
- The variance term quantities the variability of the predictions.

Typically,
- Models of low capacity have low variance but high bias.
- Models of high capacity how high variance but low bias.

---

class: end-slide, center
count: false

The end.

---

# References

- [EE-559 Deep learning](https://documents.epfl.ch/users/f/fl/fleuret/www/dlc/) (Francois Fleuret, EPFL)
- [Understanding Random Forests: From Theory to Practice](http://arxiv.org/abs/1407.7502) (Louppe, 2014)
