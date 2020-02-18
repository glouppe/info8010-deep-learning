class: middle, center, title-slide

# Deep Learning

Lecture 11: Theory of deep learning

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](g.louppe@uliege.be)

???

R: move out the GP part into a new lecture.
R: cover neural tangents there   https://rajatvd.github.io/NTK/
R: science of dl https://people.csail.mit.edu/madry/6.883/


mysteries of deep learning
-> better generalization than they should (over-param)
-> lottery ticket
-> adversarial examples
http://introtodeeplearning.com/materials/2019_6S191_L6.pdf

R: check generalization from https://m2dsupsdlclass.github.io/lectures-labs/slides/08_expressivity_optimization_generalization/index.html#87

---

# Universal approximation

.bold[Theorem.] (Cybenko 1989; Hornik et al, 1991) Let $\sigma(\cdot)$ be a
bounded, non-constant continuous function. Let $I\_p$ denote the $p$-dimensional hypercube, and
$C(I\_p)$ denote the space of continuous functions on $I\_p$. Given any $f \in C(I\_p)$ and $\epsilon > 0$, there exists $q > 0$ and $v\_i, w\_i, b\_i, i=1, ..., q$ such that
$$F(x) = \sum\_{i \leq q} v\_i \sigma(w\_i^T x + b\_i)$$
satisfies
$$\sup\_{x \in I\_p} |f(x) - F(x)| < \epsilon.$$

---

class: middle

The universal approximation theorem
- guarantees that even a single hidden-layer network can represent any classification
  problem in which the boundary is locally linear (smooth);
- does not inform about good/bad architectures, nor how they relate to the optimization procedure.
- generalizes to any non-polynomial (possibly unbounded) activation function, including the ReLU (Leshno, 1993).

---

class: middle

.bold[Theorem] (Barron, 1992) The mean integrated square error between the estimated network $\hat{F}$ and the target function $f$ is bounded by
$$O\left(\frac{C^2\_f}{q} + \frac{qp}{N}\log N\right)$$
where $N$ is the number of training points, $q$ is the number of neurons, $p$ is the input dimension, and $C\_f$ measures the global smoothness of $f$.

- Combines approximation and estimation errors.
- Provided enough data, it guarantees that adding more neurons will result in a better approximation.

---

class: middle

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-0.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-1.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-2.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-3.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-4.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-5.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-6.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-7.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-8.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-9.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-10.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-11.png)]

---

class: middle
count: false

Let us consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$  
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-12.png)]

---

# Effect of depth

.center.width-80[![](figures/lec2/folding.png)]

.bold[Theorem] (Montúfar et al, 2014) A rectifier neural network with $p$ input units and $L$ hidden layers of width $q \geq p$ can compute functions that have $\Omega((\frac{q}{p})^{(L-1)p} q^p)$ linear regions.

- That is, the number of linear regions of deep models grows **exponentially** in $L$ and polynomially in $q$.
- Even for small values of $L$ and $q$, deep rectifier models are able to produce substantially more linear regions than shallow rectifier models.