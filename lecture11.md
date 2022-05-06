class: middle, center, title-slide

# Deep Learning

Lecture 11: Generative adversarial networks

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

---

class: middle 

.center.width-55[![](figures/lec11/christies.jpg)]

.center.italic["Generative adversarial networks is the coolest idea<br> in deep learning in the last 20 years."] 

.pull-right[Yann LeCun, 2018.]

---

# Today

Learn a model of the data.

- Generative adversarial networks
- Numerics of GANs
- State of the art
- Applications

---

class: middle

# Generative adversarial networks

---

class: middle

.center.width-45[![](figures/lec11/catch-me.jpg)]

.center[.width-30[![](figures/lec11/check.jpg)] .width-30[![](figures/lec11/frank.jpg)]]

---

class: middle

## A two-player game

In **generative adversarial networks** (GANs), the task of learning a generative model is expressed as a two-player zero-sum game between two networks.

.center.width-100[![](figures/lec11/gan-setup.png)]

.footnote[Credits: Francois Fleuret, [Deep Learning](https://fleuret.org/dlc/), UNIGE/EPFL.]

---

class: middle

The first network is a *generator*  $g(\cdot;\theta) : \mathcal{Z} \to \mathcal{X}$, mapping a latent space equipped with a prior distribution $p(\mathbf{z})$ to the data space, thereby inducing a distribution
$$\mathbf{x} \sim q(\mathbf{x};\theta) \Leftrightarrow \mathbf{z} \sim p(\mathbf{z}), \mathbf{x} = g(\mathbf{z};\theta).$$

The second network $d(\cdot; \phi) : \mathcal{X} \to [0,1]$ is a **classifier** trained to distinguish between true samples $\mathbf{x} \sim p(\mathbf{x})$ and generated samples $\mathbf{x} \sim q(\mathbf{x};\theta)$.

---

class: middle

For a fixed generator $g$, the classifier $d$ can be trained by generating a two-class training set
$$\mathbf{d} = \\\{ (\mathbf{x}\_1, y=1), ..., (\mathbf{x}\_N, y=1), (g(\mathbf{z}\_1; \theta), y=0), ..., (g(\mathbf{z}\_N; \theta), y=0)  \\\},$$
where $\mathbf{x}\_i \sim p(\mathbf{x})$ and $\mathbf{z}\_i \sim p(\mathbf{z})$, and minimizing the cross-entropy loss
$$\begin{aligned}
\mathcal{L}(\phi) &= -\frac{1}{2N} \sum\_{i=1}^N \left[ \log d(\mathbf{x}\_i; \phi) + \log\left(1 - d(g(\mathbf{z}\_i;\theta); \phi)\right) \right] \\\\
&\approx -\mathbb{E}\_{\mathbf{x} \sim p(\mathbf{x})}\left[ \log d(\mathbf{x};\phi) \right] - \mathbb{E}\_{\mathbf{z} \sim p(\mathbf{z})}\left[ \log (1-d(g(\mathbf{z};\theta);\phi)) \right].
\end{aligned}$$

However, the situation is slightly more complicated since we also want to train $g$ to fool the discriminator. Fortunately, this is equivalent to maximizing $d$'s loss.

---

class: middle

Let us consider the **value function** 
$$V(\phi, \theta) = \mathbb{E}\_{\mathbf{x} \sim p(\mathbf{x})}\left[ \log d(\mathbf{x};\phi) \right] + \mathbb{E}\_{\mathbf{z} \sim p(\mathbf{z})}\left[ \log (1-d(g(\mathbf{z};\theta);\phi)) \right].$$

- For a fixed $g$, $V(\phi, \theta)$ is high if $d$ is good at recognizing true from generated samples.

- If $d$ is the best classifier given $g$, and if $V$ is high, then this implies that
the generator is bad at reproducing the data distribution.

- Conversely, $g$ will be a good generative model if $V$ is low when $d$ is a perfect opponent.

Therefore, the ultimate goal is
$$\theta^\* = \arg \min\_\theta \max\_\phi V(\phi, \theta).$$

---

class: middle

## Learning process

In practice, the minimax solution is approximated using *alternating* stochastic gradient descent:
$$
\begin{aligned}
\theta &\leftarrow \theta - \gamma \nabla\_\theta V(\phi, \theta) \\\\
\phi &\leftarrow \phi + \gamma \nabla\_\phi V(\phi, \theta),
\end{aligned}
$$
where gradients are estimated with Monte Carlo integration.

???

- For one step on $\theta$, we can optionally take $k$ steps on $\phi$, since we need the classifier to remain near optimal.
- Note that to compute $\nabla\_\theta V(\phi, \theta)$, it is necessary to backprop all the way through $d$ before computing the partial derivatives with respect to $g$'s internals.

---

class: middle

.center.width-100[![](figures/lec11/learning.png)]

.footnote[Credits: Goodfellow et al, [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), 2014.]

---


class: middle

## Game analysis

For a generator $g$ fixed at $\theta$, the classifier $d$ with parameters $\phi^\*\_\theta$ is optimal if and only if
$$\forall \mathbf{x}, d(\mathbf{x};\phi^\*\_\theta) = \frac{p(\mathbf{x})}{q(\mathbf{x};\theta) + p(\mathbf{x})}.$$

---

class: middle

Therefore,
$$\begin{aligned}
&\min\_\theta \max\_\phi V(\phi, \theta) = \min\_\theta V(\phi^\*\_\theta, \theta) \\\\
&= \min\_\theta \mathbb{E}\_{\mathbf{x} \sim p(\mathbf{x})}\left[ \log \frac{p(\mathbf{x})}{q(\mathbf{x};\theta) + p(\mathbf{x})} \right] + \mathbb{E}\_{\mathbf{x} \sim q(\mathbf{x};\theta)}\left[ \log \frac{q(\mathbf{x};\theta)}{q(\mathbf{x};\theta) + p(\mathbf{x})} \right] \\\\
&= \min\_\theta \text{KL}\left(p(\mathbf{x}) || \frac{p(\mathbf{x}) + q(\mathbf{x};\theta)}{2}\right) \\\\
&\quad\quad\quad+ \text{KL}\left(q(\mathbf{x};\theta) || \frac{p(\mathbf{x}) + q(\mathbf{x};\theta)}{2}\right) -\log 4\\\\
&= \min\_\theta 2\, \text{JSD}(p(\mathbf{x}) || q(\mathbf{x};\theta)) - \log 4
\end{aligned}$$
where $\text{JSD}$ is the Jensen-Shannon divergence.

---

class: middle

In summary,
$$
\begin{aligned}
\theta^\* &= \arg \min\_\theta \max\_\phi V(\phi, \theta) \\\\
&= \arg \min\_\theta \text{JSD}(p(\mathbf{x}) || q(\mathbf{x};\theta)).
\end{aligned}$$

Since $\text{JSD}(p(\mathbf{x}) || q(\mathbf{x};\theta))$ is minimum if and only if
$$p(\mathbf{x}) = q(\mathbf{x};\theta)$$ for all $\mathbf{x}$, this proves that the minimax solution
corresponds to a generative model that perfectly reproduces the true data distribution.

---

class: middle, center

.width-100[![](figures/lec11/ganlab.png)]

([demo](https://poloclub.github.io/ganlab))

---

class: middle

## Results

.center.width-90[![](figures/lec11/gan-gallery.png)]

.footnote[Credits: Goodfellow et al, [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), 2014.]

---

class: middle

.center.width-100[![](figures/lec11/bedrooms1.png)]

.footnote[Credits: Radford et al, [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), 2015.]

---

class: middle

.center.width-100[![](figures/lec11/bedrooms2.png)]

.footnote[Credits: Radford et al, [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), 2015.]


---

class: middle

## Open problems

Training a standard GAN often results in pathological behaviors:

- *Oscillations* without convergence: contrary to standard loss minimization,
  alternating stochastic gradient descent has no guarantee of convergence.
- **Vanishing gradients**: when the classifier $d$ is too good, the value function saturates
  and we end up with no gradient to update the generator.
- *Mode collapse*: the generator $g$ models very well a small sub-population,
  concentrating on a few modes of the data distribution.
- Performance is also difficult to assess in practice.

<br>
.center.width-100[![](figures/lec11/mode-collapse.png)]

.center[Mode collapse (Metz et al, 2016)]

---

class: middle

## Cabinet of curiosities

While early results (2014-2016) were already impressive, a close inspection of the fake samples distribution $q(\mathbf{x};\theta)$ often revealed fundamental issues highlighting architectural limitations.

---

class: middle

.center.width-90[![](figures/lec11/curiosity-cherrypicks.png)]

.center[Cherry-picks]

.footnote[Credits: Ian Goodfellow, 2016.]

---

class: middle

.center.width-90[![](figures/lec11/curiosity-counting.png)]

.center[Problems with counting]

.footnote[Credits: Ian Goodfellow, 2016.]

---

class: middle

.center.width-90[![](figures/lec11/curiosity-perspective.png)]

.center[Problems with perspective]

.footnote[Credits: Ian Goodfellow, 2016.]

---

class: middle

.center.width-90[![](figures/lec11/curiosity-global.png)]

.center[Problems with global structures]

.footnote[Credits: Ian Goodfellow, 2016.]

---

class: middle, inactive
count: false
exclude: true

# .inactive[Wasserstein GANs]

(optional)

---

class: middle
count: false
exclude: true

## Return of the Vanishing Gradients

For most non-toy data distributions, the fake samples $\mathbf{x} \sim q(\mathbf{x};\theta)$
may be so bad initially that the response of $d$ saturates.

At the limit, when $d$ is perfect given the current generator $g$,
$$\begin{aligned}
d(\mathbf{x};\phi) &= 1, \forall \mathbf{x} \sim p(\mathbf{x}), \\\\
d(\mathbf{x};\phi) &= 0, \forall \mathbf{x} \sim q(\mathbf{x};\theta).
\end{aligned}$$
Therefore,
$$V(\phi, \theta) =  \mathbb{E}\_{\mathbf{x} \sim p(\mathbf{x})}\left[ \log d(\mathbf{x};\phi) \right] + \mathbb{E}\_{\mathbf{z} \sim p(\mathbf{z})}\left[ \log (1-d(g(\mathbf{z};\theta);\phi)) \right] = 0$$
and $\nabla\_\theta V(\phi,\theta) = 0$, thereby **halting** gradient descent.

---

class: middle
count: false
exclude: true

Dilemma :
- If $d$ is bad, then $g$ does not have accurate feedback and the loss function cannot represent the reality.
- If $d$ is too good, the gradients drop to 0, thereby slowing down or even halting the optimization.

---

class: middle
count: false
exclude: true

## Jensen-Shannon divergence

For any two distributions $p$ and $q$,
$$0 \leq JSD(p||q) \leq \log 2,$$
where
- $JSD(p||q)=0$ if and only if $p=q$,
- $JSD(p||q)=\log 2$ if and only if $p$ and $q$ have disjoint supports.

.center[![](figures/lec11/jsd.gif)]

---

class: middle
count: false
exclude: true

Notice how the Jensen-Shannon divergence poorly accounts for the metric structure of the space.

Intuitively, instead of comparing distributions "vertically", we would like to compare them "horizontally".

.center[![](figures/lec11/jsd-vs-emd.png)]

---

class: middle
count: false
exclude: true

## Wasserstein distance

An alternative choice is the **Earth mover's distance**, which intuitively
corresponds to the minimum mass displacement to transform one distribution into
the other.

.center.width-100[![](figures/lec11/emd-moves.png)]

- $p = \frac{1}{4}\mathbf{1}\_{[1,2]} + \frac{1}{4}\mathbf{1}\_{[3,4]} + \frac{1}{2}\mathbf{1}\_{[9,10]}$
- $q = \mathbf{1}\_{[5,7]}$

Then,
$$\text{W}\_1(p,q) = 4\times\frac{1}{4} + 2\times\frac{1}{4} + 3\times\frac{1}{2}=3$$

.footnote[Credits: Francois Fleuret, [Deep Learning](https://fleuret.org/dlc/), UNIGE/EPFL.]

---

class: middle
count: false
exclude: true

The Earth mover's distance is also known as the Wasserstein-1 distance and is defined as:
$$\text{W}\_1(p, q) = \inf\_{\gamma \in \Pi(p,q)} \mathbb{E}\_{(x,y)\sim \gamma} \left[||x-y||\right]$$
where:
- $\Pi(p,q)$ denotes the set of all joint distributions $\gamma(x,y)$ whose marginals are respectively $p$ and $q$;
- $\gamma(x,y)$ indicates how much mass must be transported from $x$ to $y$ in order to transform the distribution $p$ into $q$.
- $||\cdot||$ is the L1 norm and $||x-y||$ represents the cost of moving a unit of mass from $x$ to $y$.

---

class: middle
count: false
exclude: true

.center[![](figures/lec11/transport-plan.png)]

---

class: middle
count: false
exclude: true

Notice how the $\text{W}\_1$ distance does not saturate. Instead, it
 increases monotonically with the distance between modes:

.center[![](figures/lec11/emd.png)]

$$\text{W}\_1(p,q)=d$$

For any two distributions $p$ and $q$,
- $W\_1(p,q) \in \mathbb{R}^+$,
- $W\_1(p,q)=0$ if and only if $p=q$.

---

class: middle
count: false
exclude: true

## Wasserstein GANs

Given the attractive properties of the Wasserstein-1 distance, Arjovsky et al (2017) propose
to learn a generative model by solving instead:
$$\theta^\* = \arg \min\_\theta \text{W}\_1(p(\mathbf{x})||q(\mathbf{x};\theta))$$
Unfortunately, the definition of $\text{W}\_1$ does not provide with an operational way of estimating it because of the intractable $\inf$.

On the other hand, the Kantorovich-Rubinstein duality tells us that
$$\text{W}\_1(p(\mathbf{x})||q(\mathbf{x};\theta)) = \sup\_{||f||\_L \leq 1} \mathbb{E}\_{\mathbf{x} \sim p(\mathbf{x})}\left[ f(\mathbf{x}) \right] - \mathbb{E}\_{\mathbf{x} \sim q(\mathbf{x};\theta)} \left[f(\mathbf{x})\right]$$
where the supremum is over all the 1-Lipschitz functions $f:\mathcal{X} \to \mathbb{R}$. That is, functions $f$ such that
$$||f||\_L = \max\_{\mathbf{x},\mathbf{x}'} \frac{||f(\mathbf{x}) - f(\mathbf{x}')||}{||\mathbf{x} - \mathbf{x}'||} \leq 1.$$

---

class: middle
count: false
exclude: true

.center.width-80[![](figures/lec11/kr-duality.png)]

For $p = \frac{1}{4}\mathbf{1}\_{[1,2]} + \frac{1}{4}\mathbf{1}\_{[3,4]} + \frac{1}{2}\mathbf{1}\_{[9,10]}$
and $q = \mathbf{1}\_{[5,7]}$,
$$\begin{aligned}
\text{W}\_1(p,q) &= 4\times\frac{1}{4} + 2\times\frac{1}{4} + 3\times\frac{1}{2}=3 \\\\
&= \underbrace{\left(3\times \frac{1}{4} + 1\times\frac{1}{4}+2\times\frac{1}{2}\right)}\_{\mathbb{E}\_{\mathbf{x} \sim p(\mathbf{x})}\left[ f(\mathbf{x}) \right]} - \underbrace{\left(-1\times\frac{1}{2}-1\times\frac{1}{2}\right)}\_{\mathbb{E}\_{\mathbf{x} \sim q(\mathbf{x};\theta)}\left[f(\mathbf{x})\right]} = 3
\end{aligned}
$$

.footnote[Credits: Francois Fleuret, [Deep Learning](https://fleuret.org/dlc/), UNIGE/EPFL.]

---

class: middle
count: false
exclude: true

Using this result, the Wasserstein GAN algorithm consists in solving the minimax problem:
$$\theta^\* = \arg \min\_\theta \max\_{\phi:||d(\cdot;\phi)||\_L \leq 1}  \mathbb{E}\_{\mathbf{x} \sim p(\mathbf{x})}\left[ d(\mathbf{x};\phi) \right] - \mathbb{E}\_{\mathbf{x} \sim q(\mathbf{x};\theta)} \left[d(\mathbf{x};\phi)\right]$$$$
Note that this formulation is very close to the original GANs, except that:
- The classifier $d:\mathcal{X} \to [0,1]$ is replaced by a critic function $d:\mathcal{X}\to \mathbb{R}$
  and its output is not interpreted through the cross-entropy loss;
- There is a strong regularization on the form of $d$.
  In practice, to ensure 1-Lipschitzness,
    - Arjovsky et al (2017) propose to clip the weights of the critic at each iteration;
    - Gulrajani et al (2017) add a regularization term to the loss.
- As a result, Wasserstein GANs benefit from:
    - a meaningful loss metric,
    - improved stability (no mode collapse is observed).

---

class: middle
count: false
exclude: true

.center.width-90[![](figures/lec11/wgan.png)]

.footnote[Credits: Arjovsky et al, [Wasserstein GAN](https://arxiv.org/abs/1701.07875), 2017.]

---

class: middle
count: false
exclude: true

.center.width-70[![](figures/lec11/wgan-gallery.png)]

.footnote[Credits: Arjovsky et al, [Wasserstein GAN](https://arxiv.org/abs/1701.07875), 2017.]

---

class: middle

# Numerics of GANs

???

Check https://mitliagkas.github.io/ift6085-2019/ift-6085-lecture-14-notes.pdf

---

class: middle

.center[
.width-45[![](figures/lec11/animation2.gif)]
.width-45[![](figures/lec11/animation1.gif)]
]

Solving for saddle points is different from gradient descent.
- Minimization of scalar functions yields *conservative* vector fields.
- Min-max saddle point problems may yield **non-conservative** vector fields.

.footnote[Credits: Ferenc Huszár, [GANs are Broken in More than One Way](https://www.inference.vc/my-notes-on-the-numerics-of-gans/), 2017.]

???

A vector field is conservative when it can be expressed as the gradient of a scalar function.

---

class: middle

Following the notations of Mescheder et al (2018), the training objective for the two players can be described by an objective function of the form
$$L(\theta,\phi) = \mathbb{E}\_{p(\mathbf{z})}\left[ f(d(g(\mathbf{z};\theta);\phi)) \right] + \mathbb{E}\_{p(\mathbf{x})}\left[f(-d(\mathbf{x};\phi))\right],$$
where the goal of the generator is to minimizes the loss, whereas the discriminator tries to maximize it.

If $f(t)=-\log(1+\exp(-t))$, then we recover the original GAN objective (assuming that $d$ outputs the logits).

???

If $f(t)=-t$ and and if we impose the Lipschitz constraint on $d$, then we recover Wassterstein GAN.

---

class: middle

Training algorithms can be described as fixed points algorithms that apply some operator $F\_h(\theta,\phi)$ to the parameters values $(\theta,\phi)$.

- For simultaneous gradient descent,
$$F\_h(\theta,\phi) = (\theta,\phi) + h v(\theta,\phi)$$
where $v(\theta,\phi)$ denotes the **gradient vector field**
$$v(\theta,\phi):= \begin{pmatrix}
-\frac{\partial L}{\partial \theta}(\theta,\phi) \\\\
\frac{\partial L}{\partial \phi}(\theta,\phi)
\end{pmatrix}$$
and $h$ is a scalar stepsize.
- Similarly, alternating gradient descent can be described by an operator $F\_h = F\_{2,h} \circ F\_{1,h}$, where $F\_{1,h}$ and $F\_{2,h}$ perform an update for the generator and discriminator, respectively.

---

class: middle

## Local convergence near an equilibrium point

Let us consider the Jacobian $J\_{F\_h}(\theta^\*,\phi^\*)$ at the equilibrium $(\theta^\*,\phi^\*)$:
- if $J\_{F\_h}(\theta^\*,\phi^\*)$ has eigenvalues with absolute value bigger than 1, the training will generally not converge to $(\theta^\*,\phi^\*)$.
- if all eigenvalues have absolute value smaller than 1, the training will converge to $(\theta^\*,\phi^\*)$.
- if all eigenvalues values are on the unit circle, training can be convergent, divergent or neither.

Mescheder et al (2017) show that all eigenvalues can be forced to remain within the unit ball if and only if the stepsize $h$ is made sufficiently small.

---

class: middle

.width-90.center[![](figures/lec11/discrete-diverge.png)]

.center[Discrete system: divergence ($h=1$, too large).]

.footnote[Credits: Mescheder et al, [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406), 2018.]

---

class: middle

.width-90.center[![](figures/lec11/discrete-converge.png)]

.center[Discrete system: convergence ($h=0.5$, small enough).]

.footnote[Credits: Mescheder et al, [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406), 2018.]

---

class: middle

For the (idealized) continuous system
$$
\begin{pmatrix}
\dot{\theta}(t) \\\\
\dot{\phi}(t)
\end{pmatrix} =
\begin{pmatrix}
-\frac{\partial L}{\partial \theta}(\theta,\phi) \\\\
\frac{\partial L}{\partial \phi}(\theta,\phi)
\end{pmatrix},$$
which corresponds to training GANs with infinitely small learning rate $h \to 0$:
- if all eigenvalues of the Jacobian $v'(\theta^\*,\phi^\*)$ at a stationary point $(\theta^\*,\phi^\*)$ have negative real-part, the continuous system converges locally to $(\theta^\*,\phi^\*)$;
- if $v'(\theta^\*,\phi^\*)$ has eigenvalues with positive real-part, the continuous system is not locally convergent.
- if all eigenvalues have zero real-part, it can be convergent, divergent or neither.

---

class: middle

.width-90.center[![](figures/lec11/continuous-diverge.png)]

.center[Continuous system: divergence.]

.footnote[Credits: Mescheder et al, [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406), 2018.]

---


class: middle

.width-90.center[![](figures/lec11/continuous-converge.png)]

.center[Continuous system: convergence.]

.footnote[Credits: Mescheder et al, [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406), 2018.]

---

class: middle

.width-90.center[![](figures/lec11/dirac-unreg.png)]

On the Dirac-GAN toy problem, eigenvalues are $\\{ -f'(0)i, +f'(0)i \\}$.
Therefore convergence of the standard GAN learning procedure is not guaranteed.

.footnote[Credits: Mescheder et al, [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406), 2018.]

---

class: middle
exclude: true

## Dirac-GAN: Wasserstein GANs

.width-90.center[![](figures/lec11/dirac-wgan.png)]

Eigenvalues are $\\{ -i, +i \\}$.
Therefore convergence is not guaranteed.

.footnote[Credits: Mescheder et al, [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406), 2018.]

---

class: middle

## Taming the vector field

.width-90.center[![](figures/lec11/dirac-reg.png)]

A penalty on the squared norm of the gradients of the discriminator results in the regularization
$$R\_1(\phi) = \frac{\gamma}{2} \mathbb{E}\_{\mathbf{x} \sim p(\mathbf{x})}\left[ || \nabla\_\mathbf{x} d(\mathbf{x};\phi)||^2 \right].$$
The resulting eigenvalues are $\\{ -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma}{4} - f'(0)^2}\\}$.
Therefore, for $\gamma>0$, all eigenvalues have negative real part, hence training is locally convergent!


.footnote[Credits: Mescheder et al, [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406), 2018.]

---

class: middle

.center.width-70[![](figures/lec11/reg1.png)]

.footnote[Credits: Mescheder et al, [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406), 2018.]

---

class: middle

.center.width-70[![](figures/lec11/reg2.png)]

.footnote[Credits: Mescheder et al, [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406), 2018.]

---

class: middle

.center.width-70[![](figures/lec11/reg3.png)]

.footnote[Credits: Mescheder et al, [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406), 2018.]

---

class: middle

# State of the art

---

class: middle

.center.width-70[![](figures/lec11/timeline.png)]

---

class: middle

## Progressive growing of GANs

.center[

Wasserstein GANs as baseline (Arjovsky et al, 2017) + <br>Gradient Penalty (Gulrajani, 2017) + (quite a few other tricks)

+

]

.center.width-100[![](figures/lec11/progressive-gan.png)]

.center[(Karras et al, 2017)]

---

class: middle

.center.width-100[![](figures/lec11/progressive-gan2.png)]

.center[(Karras et al, 2017)]


---

class: middle, center, black-slide

<iframe width="600" height="450" src="https://www.youtube.com/embed/XOxxPcy5Gr4" frameborder="0" volume="0" allowfullscreen></iframe>

(Karras et al, 2017)

---

class: middle

## BigGANs

.center[

Self-attention GANs as baseline (Zhang et al, 2018) + Hinge loss objective (Lim and Ye, 2017; Tran et al, 2017) + Class information to $g$ with class-conditional batchnorm (de Vries et al, 2017) + Class information to $d$ with projection (Miyato and Koyama, 2018) + Half the learning rate of SAGAN, 2 $d$-steps per $g$-step   + Spectral normalization for both $g$ and $d$ + Orthogonal initialization (Saxe et al, 2014) + Large minibatches (2048) + Large number of convolution filters + Shared embedding and hierarchical latent spaces + Orthogonal regularization + Truncated sampling + (quite a few other tricks)

]

<br>
.center.width-100[![](figures/lec11/biggan.png)]
.center[(Brock et al, 2018)]

---

class: middle, center, black-slide

<iframe width="600" height="450" src="https://www.youtube.com/embed/YY6LrQSxIbc" frameborder="0" allowfullscreen></iframe>

(Brock et al, 2018)

---

class: middle

## StyleGAN (v1)

.center[

Progressive GANs as baseline (Karras et al, 2017) + Non-saturating loss instead of WGAN-GP + $R\_1$ regularization (Mescheder et al, 2018) + (quite a few other tricks)

+

.width-60[![](figures/lec11/stylegan.png)]

]

---

class: middle, center, black-slide

<iframe width="600" height="450" src="https://www.youtube.com/embed/kSLJriaOumA" frameborder="0" allowfullscreen></iframe>

(Karras et al, 2018)

---

class: middle

The StyleGAN generator $g$ is so powerful that it can re-generate arbitrary faces.

.center[
.width-30[![](figures/lec11/stylegan-gilles.png)] &nbsp;
.width-30[![](figures/lec11/stylegan-damien.png)]
]

---

class: middle

.center[.width-80[![](figures/lec11/stylegan-interpolation.jpg)]]

---

class: middle

.center[
.width-30[![](figures/lec11/stylegan-damien-1.png)] &nbsp;
.width-30[![](figures/lec11/stylegan-damien-2.png)]

.width-30[![](figures/lec11/stylegan-damien-3.png)] &nbsp;
.width-30[![](figures/lec11/stylegan-damien-4.png)]
]

---

class: middle 

## StyleGAN (v2, v3)

.center[

<video controls preload="auto" height="320" width="320">
  <source src="https://nvlabs-fi-cdn.nvidia.com/_web/stylegan3/videos/video_3_afhq_interpolations.mp4#t=0.001" type="video/mp4">
</video>

<video controls preload="auto" height="320" width="320">
  <source src="https://nvlabs-fi-cdn.nvidia.com/_web/stylegan3/videos/video_4_beaches_interpolations.mp4#t=0.001" type="video/mp4">
</video>

]

.center[(Karras et al, 2019; Karras et al, 2021)]

---

class: middle

## VQGAN 

.width-100[![](figures/lec11/vqgan.png)]

.center[(Esser et al, 2021)]

???

Check https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/

---

class: middle, center
.width-45[![](figures/lec11/vqgan1.jpg)] &nbsp;
.width-45[![](figures/lec11/vqgan2.jpg)]

(Esser et al, 2021)

---

class: middle

# Applications

---

class: middle

## Image-to-image translation

.center[

.width-90[![](figures/lec11/cyclegan.jpeg)]

![](figures/lec11/horse2zebra.gif)

.center[CycleGANs (Zhu et al, 2017)]

]

---

class: middle, center, black-slide

<iframe width="600" height="450" src="https://www.youtube.com/embed/3AIpPlzM_qs" frameborder="0" volume="0" allowfullscreen></iframe>

High-resolution image synthesis (Wang et al, 2017)

---

class: middle, center, black-slide

<iframe width="600" height="450" src="https://www.youtube.com/embed/p5U4NgVGAwg" frameborder="0" allowfullscreen></iframe>

GauGAN: Changing sketches into photorealistic masterpieces (NVIDIA, 2019)

---

class: middle, center, black-slide

<iframe width="600" height="450" src="https://www.youtube.com/embed/p9MAvRpT6Cg" frameborder="0" allowfullscreen></iframe>

GauGAN2 (NVIDIA, 2021)

---

class: middle

## Living portraits / deepfakes

.center[

.width-80[![](figures/lec11/portrait.png)]

Few-Shot Adversarial Learning of Realistic Neural Talking Head Models<br> (Zakharov et al, 2019)

]

---

class: middle, center, black-slide

<iframe width="600" height="450" src="https://www.youtube.com/embed/rJb0MDrT3SE" frameborder="0" allowfullscreen></iframe>

Few-Shot Adversarial Learning of Realistic Neural Talking Head Models<br> (Zakharov et al, 2019)

---

class: middle

## Captioning

.width-100[![](figures/lec11/caption1.png)]

.width-100[![](figures/lec11/caption2.png)]

.center[(Shetty et al, 2017)]

---

class: middle

## Text-to-image synthesis

.center[

.width-100[![](figures/lec11/stackgan1.png)]

.center[(Zhang et al, 2017)]

]

---

class: middle

.center[

.width-100[![](figures/lec11/stackgan2.png)]

.center[(Zhang et al, 2017)]

]

---

class: middle

.center[

.width-100[![](figures/lec11/styleclip.png)]
.width-100[![](figures/lec11/styleclip-teaser.png)]

.center[StyleCLIP (Patashnik et al, 2021)]

]

???

See also https://stylegan-nada.github.io/ or VQGAN+CLIP.

---

class: middle

##  Music generation

.center.width-100[![](figures/lec11/musegan.png)]

.center[

<audio src="https://salu133445.github.io/musegan/audio/best_samples.mp3" type="audio/mpeg" controls="" controlslist="nodownload">Your browser does not support the audio element.</audio>

]

.center[MuseGAN (Dong et al, 2018)]

---

class: middle

## Accelerating scientific simulators

.grid[
.kol-2-3[.width-100[![](figures/lec11/calogan1.png)]]
.kol-1-3[<br>.width-100[![](figures/lec11/calogan2.png)]]
]

.center[Learning particle physics (Paganini et al, 2017)]

???

https://arxiv.org/pdf/1712.10321.pdf

---

class: middle

.center.width-70[![](figures/lec11/cosmo.png)]

.center[Learning cosmological models (Rodriguez et al, 2018)]

???

https://arxiv.org/pdf/1801.09070.pdf


---

class: end-slide, center
count: false

The end.
