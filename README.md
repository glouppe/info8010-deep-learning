# INFO8010 - Deep Learning

Lectures for INFO8010 - Deep Learning, ULiège, Spring 2019.

- Instructor: Gilles Louppe ([g.louppe@uliege.be](mailto:g.louppe@uliege.be))
- Teaching assistants:
    - Joeri Hermans ([joeri.hermans@doct.uliege.be](mailto:joeri.hermans@doct.uliege.be))
    - Matthia Sabatelli ([m.sabatelli@uliege.be](mailto:m.sabatelli@uliege.be))
    - Antoine Wehenkel ([antoine.wehenkel@uliege.be](antoine.wehenkel@uliege.be))
- When: Spring 2019, Friday 9:00AM
- Classroom: [B28/R7](https://www.campus.uliege.be/cms/c_5119631/fr/r7-montefiore)

## Slides

(Tentative and subject to change!)

- [Outline](https://glouppe.github.io/info8010-deep-learning/?p=outline.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/outline.pdf)]
- Lecture 1 (February 8): [Fundamentals of machine learning](https://glouppe.github.io/info8010-deep-learning/?p=lecture1.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec1.pdf)]
- Lecture 2 (February 15): [Neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture2.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec2.pdf)]
- Lecture 3 (February 22): [Convolutional neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture3.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec3.pdf)]
- Lecture 4 (March 1): [Training neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture4.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec4.pdf)]
- Lecture 5 (March 8): Recurrent neural networks
- Lecture 6 (March 15): Auto-encoders and generative models
- *No class on March 22.*
- Lecture 7 (March 29): Generative adversarial networks [Room change: B7a A300]
- Lecture 8 (April 5): Unsupervised Learning
- Lecture 9 (April 26): Applications
- Lecture 10 (May 3): Student presentations 1
- Lecture 11 (May 10): Student presentations 2
- Lecture 17 (May 17): Student presentations 3

## Project

The course project is an opportunity for you to apply what you have learned in class to a problem of your interest. Potential projects fall into two tracks:
- *Applications*: Pick a real-world problem and apply deep networks to solve it.
- *Models*: Build a new model or algorithm, or a variant of existing models, and apply it to a task. This track might be more challenging, but could lead to a publishable piece of work.

Be creative and ambitious!

#### Instructions

- Students can work in groups of maximum 3 students.
- Each group must write a short (1-2 pages) research project proposal. It should include a description of a minimum viable project, some nice-to-haves if time allows, and a short review of related work.
- Towards the end of the class, you will submit a project report (around 4 to 8 pages), in the format of a machine learning conference paper.
- At the end of the course everyone will present their project to the class.
- The grade will depend on the ideas, how well you present them in the report, how clearly you position your work relative to existing literature, how illuminating your experiments are, and how well-supported your conclusions are.
- Both the project proposal and the project report should follow the LaTex template [`template-report.tex`](https://glouppe.github.io/info8010-deep-learning/template-report.tex).

#### Agenda

- Project proposal, due by **March 1, 2019 at 23:59**.
- Project code and report, due by **April 26, 2019 at 23:59**.
- Project presentations on May 3, 10 and 17, 2019.

Projects should be submitted through the [Montefiore submission platform](https://submit.montefiore.ulg.ac.be).

#### Project ideas

General applications:
- Multi-task learning
- Domain transfer
- Hardware project on an NVIDIA Jetson card (contact us for more details)
- Deep learning for Science
- Recommendation systems

Languages:
- Neural translation
- Automatic image captioning
- Chat bots

Computer vision:
- Real-time object detection
- Semantic segmentation
- Pose estimation
- Image or sound synthesis
- Handwriting recognition and synthesis with recurrent networks
- Style transfer

AI & games:
- Master a video game with deep learning
- Synchronous vs. asynchronous deep RL
- Imitation learning

Theory:
- Comparison between convolutional networks and capsule networks
- Comparison of generative models
- Uncertainty in deep learning
- Theoretical investigation of deep learning
- Interpretability of deep networks

... some more inspiration:
- [Stanford CS231n Project reports (Spring 2017)](http://cs231n.stanford.edu/2017/reports.html)
- [Stanford CS231n Project reports (Winter 2016)](http://cs231n.stanford.edu/2016/reports.html)
- [Stanford CS231n Project reports (Winter 2015)](http://cs231n.stanford.edu/2015/reports.html)
- [Stanford CS229 Projects](http://cs229.stanford.edu/projects.html)

#### Evaluation guidelines

Report:
1) Clarity: quality of the abstract and introduction with a clear formulation of the research question, is there a proper related-work section and explanation of the experimental setup? Are the results of the report overall scientifically sound and replicable?  

2) Writing style: is the paper properly formatted and structured? does it not contain language errors? are the figures and tables properly captioned and interpretable?

3) Results: do the results support the original research question? why/why not? Are they clearly presented and convincing?

4) Development: are the contributions of the group to the development of the project well defined? what has been implemented with respect to the original research questions, what has been re-used from existing coding directories?

---

Credits: projects instructions adapted from [Stanford CS231](http://cs231n.stanford.edu/project.html) and [University of Toronto CSC 2541](https://www.cs.toronto.edu/~duvenaud/courses/csc2541/index.html).

## Reading assignment

Your task is to read and summarize a major scientific paper in the field of deep learning. You are free to select one among the following three papers:

> - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. arXiv:[1512.03385](https://arxiv.org/abs/1512.03385).
> - Andrychowicz, M., Denil, M., Gomez, S., Hoffman, M. W., Pfau, D., Schaul, T., ... & De Freitas, N. (2016). Learning to learn by gradient descent by gradient descent. arXiv:[1606.04474](https://arxiv.org/abs/1606.04474).
> - Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2016). Understanding deep learning requires rethinking generalization. arXiv:[1611.03530](https://arxiv.org/abs/1611.03530).

You should produce a report that summarizes the problem that is tackled by the paper and explains why it is challenging or important. The report should outline the main contributions and results with respect to the problem that is addressed. It should also include a critical discussion of the advantages and shortcomings of the contributions of the paper.

Constraints:
- You can work in groups of maximum 3 students.
- You report must be written in English.
- 2 pages (excluding references, if any).
- Formatted using the LaTeX template [`template-report.tex`](https://glouppe.github.io/info8010-deep-learning/template-report.tex).

Your report should be submitted  by **April 5, 2019 at 23:59** on the [submission platform](https://submit.montefiore.ulg.ac.be/). This is a **hard** deadline.
