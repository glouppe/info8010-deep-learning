# INFO8010 - Deep Learning

Lectures for INFO8010 - Deep Learning, ULiège, Spring 2021.

- Instructor: Gilles Louppe
- Teaching assistants: Matthia Sabatelli, Antoine Wehenkel
- When: Spring 2021, Friday 9:00 AM
- Classroom: Lectures are virtual and broadcast on [Youtube](https://www.youtube.com/channel/UCJWL9RHD2nZa85lK-k0v8lA).
- Discussion: Join us on Discord (see invitation link in emails)

## Agenda

| Date | Topic |
| --- | --- |
| February 5 | [Course syllabus](https://glouppe.github.io/info8010-deep-learning/?p=course-syllabus.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/course-syllabus.pdf)] [[video](https://www.youtube.com/watch?v=51UOdB199Nk)]<br>Lecture 0: [Introduction](https://glouppe.github.io/info8010-deep-learning/?p=lecture0.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec0.pdf)] [[video](https://www.youtube.com/watch?v=-Ee-Z311a3k)]<br>Lecture 1: [Fundamentals of machine learning](https://glouppe.github.io/info8010-deep-learning/?p=lecture1.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec1.pdf)] [[Side notes](https://glouppe.github.io/info8010-deep-learning/pdf/lec1-sidenotes.pdf)]  [[video](https://www.youtube.com/watch?v=GwpG0sHPklE)] |
| February 12 | Lecture 2: [Multi-layer perceptron](https://glouppe.github.io/info8010-deep-learning/?p=lecture2.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec2.pdf)] [[Side notes](https://glouppe.github.io/info8010-deep-learning/pdf/lec2-sidenotes.pdf)]  [[video](https://www.youtube.com/watch?v=OF6AkE9Fnjc)] |
| February 19 | Lecture 3: [Automatic differentiation](https://glouppe.github.io/info8010-deep-learning/?p=lecture3.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec3.pdf)] [[Side notes](https://glouppe.github.io/info8010-deep-learning/pdf/lec3-sidenotes.pdf)] [[video](https://youtu.be/fD047xXpSfI)] |
| February 26| Lecture 4: [Training neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture4.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec4.pdf)] [[Side notes](https://glouppe.github.io/info8010-deep-learning/pdf/lec4-sidenotes.pdf)] [[video](https://youtu.be/G7qw620V_3g)]  |
| March 1 | Deadline for [Homework 1](https://github.com/glouppe/info8010-deep-learning/raw/master/homeworks/homework1.zip) |
| March 5 | Lecture 5: [Convolutional neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture5.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec5.pdf)] [[Side notes](https://glouppe.github.io/info8010-deep-learning/pdf/lec5-sidenotes.pdf)] [[video](https://youtu.be/54WShJMWYo0)] |
| March 12 | Lecture 6: [Computer vision](https://glouppe.github.io/info8010-deep-learning/?p=lecture6.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec6.pdf)] [[Side notes](https://glouppe.github.io/info8010-deep-learning/pdf/lec6-sidenotes.pdf)] [[video](https://youtu.be/cfZGfJaLRxA)] |
| March 19 | Lecture 7: [Recurrent neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture7.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec7.pdf)] [[Side notes](https://glouppe.github.io/info8010-deep-learning/pdf/lec7-sidenotes.pdf)] [[video](https://youtu.be/qnux5dg5wZ4)] |
| March 22 | Deadline for [Homework 2](https://github.com/glouppe/info8010-deep-learning/raw/master/homeworks/homework2.ipynb) | 
| March 26<br>[Live stream](https://youtu.be/WfPNJ5MWxHI) | Lecture 8: Attention and transformers<br>Deadline for the project proposal |
| April 2 | Lecture 9: Generative models (part 1) |
| April 23 | Lecture 10: Generative models (part 2) |
| April 30 | Lecture 11: Uncertainty<br>Deadline for the reading assignment |
| May 7 | Lecture 12: Deep reinforcement learning |
| May 14 | Deadline for the project (code and report) |

## Homeworks

The goal of these two assignments is to get you familiar with the PyTorch library.
Each homework should be done in groups of 2 or 3 (the same as for the project) and must be submitted on the [Montefiore submission platform](https://submit.montefiore.ulg.ac.be/) before 23:59 on the due date.

- [Homework 1](https://github.com/glouppe/info8010-deep-learning/raw/master/homeworks/homework1.zip): Tensor operations, `autograd` and `nn`. Due by March 1.
- [Homework 2](https://github.com/glouppe/info8010-deep-learning/raw/master/homeworks/homework2.ipynb): Dataset, Dataloader, running on GPU, training a convolutional neural network. Due by March 22.

Homeworks are optional. If submitted, each homework will account for 5% of the final grade.

## Project

See instructions in [`project.md`](https://github.com/glouppe/info8010-deep-learning/blob/master/project.md).

## Reading assignment

Your task is to read and summarize a major scientific paper in the field of deep learning. You are free to select one among the following three papers:

> - T. Karras al, "Analyzing and improving the image quality of StyleGAN", 2020. [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf)]
> - E. Nalisnick, "Do Deep Generative Models Know What They Don't Know?", 2019 [[pdf](https://arxiv.org/pdf/1810.09136)]
> - M. Cranmer et al, "Discovering Symbolic Models from Deep Learning with Inductive Biases", 2020. [[pdf](https://arxiv.org/pdf/2006.11287.pdf)]

You should produce a report that summarizes the problem that is tackled by the paper and explains why it is challenging or important, from the perspective of the wider research context. The report should outline the main contributions and results with respect to the problem that is addressed. It should also include a critical discussion of the advantages and shortcomings of the contributions of the paper.
Further guidelines for writing a good summary can be found [here](https://web.stanford.edu/class/cs224n/project/project-proposal-instructions.pdf) (Section 2, "The summary").

Constraints:
- You can work in groups of maximum 3 students.
- You report must be written in English.
- 2 pages (excluding references, if any).
- Formatted using the LaTeX template [`template-report.tex`](https://glouppe.github.io/info8010-deep-learning/template-report.tex).

Your report should be submitted  by **April 30, 2021 at 23:59** on the [submission platform](https://submit.montefiore.ulg.ac.be/). This is a **hard** deadline.

## Archives

- [2019-2020](https://github.com/glouppe/info8010-deep-learning/tree/v3-info8010-2020)
- [2018-2019](https://github.com/glouppe/info8010-deep-learning/tree/v2-info8010-2019)
