# INFO8010 - Deep Learning

Lectures for INFO8010 - Deep Learning, ULiège, Spring 2022.

- Instructor: Gilles Louppe
- Teaching assistants: Arnaud Delaunoy, François Rozet, Gaspard Lambrechts, Yann Claes, Antoine Wehenkel
- When: Spring 2022, Friday 8:30 AM
- Classroom: R3 / B28

## Agenda

| Date | Topic |
| --- | --- |
| February 11 | [Course syllabus](https://glouppe.github.io/info8010-deep-learning/?p=course-syllabus.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/course-syllabus.pdf)] [[video](https://www.youtube.com/watch?v=51UOdB199Nk)]<br>Lecture 0: [Introduction](https://glouppe.github.io/info8010-deep-learning/?p=lecture0.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec0.pdf)] [[video](https://www.youtube.com/watch?v=-Ee-Z311a3k)]<br>Lecture 1: [Fundamentals of machine learning](https://glouppe.github.io/info8010-deep-learning/?p=lecture1.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec1.pdf)]  [[video](https://www.youtube.com/watch?v=GwpG0sHPklE)] |
| February 18 | Lecture 2: [Multi-layer perceptron](https://glouppe.github.io/info8010-deep-learning/?p=lecture2.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec2.pdf)] [[video](https://www.youtube.com/watch?v=OF6AkE9Fnjc)] |
| February 25 | Lecture 3: [Automatic differentiation](https://glouppe.github.io/info8010-deep-learning/?p=lecture3.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec3.pdf)] [[video](https://youtu.be/fD047xXpSfI)] |
| March 4 | Lecture 4: [Training neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture4.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec4.pdf)] [[video](https://youtu.be/G7qw620V_3g)]  |
| March 7 | Deadline for homework 1 |
| March 11 | Lecture 5: [Convolutional neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture5.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec5.pdf)] [[video](https://youtu.be/54WShJMWYo0)] |
| March 14 | Deadline for project proposal |
| March 18 | Lecture 6: [Computer vision](https://glouppe.github.io/info8010-deep-learning/?p=lecture6.md) [PDF] [[video](https://youtu.be/cfZGfJaLRxA)] |
| March 21 | Deadline for homework 2 |
| March 25 | Lecture 7: [Recurrent neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture7.md) [PDF] [[video](https://youtu.be/qnux5dg5wZ4)] |
| April 1 | Lecture 8: [Attention and transformers](https://glouppe.github.io/info8010-deep-learning/?p=lecture8.md) [PDF] [[video](https://youtu.be/cwFE1pLld-g)] |
| April 22 | Lecture 9: [Graph neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture9.md)  [PDF] |
| April 29 | Lecture 10: [Auto-encoders and variational auto-encoders](https://glouppe.github.io/info8010-deep-learning/?p=lecture10.md) [[video](https://youtu.be/6gWeyUZtHWs)] |
| May 6 | Lecture 11: [Generative adversarial networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture11.md) [[video](https://youtu.be/cM6m1eHY5FI)] |
| May 8 | Deadline for the reading assignment |
| May 8 | Deadline for the project |
| May 13 | Lecture 12: [Uncertainty](https://glouppe.github.io/info8010-deep-learning/?p=lecture12.md) [PDF]  [[video](https://youtu.be/AxJBY9eRTL4)] |

## Homeworks

The goal of these two assignments is to get you familiar with the PyTorch library. You can find the installation instructions in the [Homeworks](./homeworks) folder.
Each homework should be done in groups of 2 or 3 (the same as for the project) and must be submitted **before 23:59 on the due date**.
Homeworks should be submitted on the [Montefiore submission platform](https://submit.montefiore.ulg.ac.be).

- [Homework 1](https://github.com/glouppe/info8010-deep-learning/raw/master/homeworks/homework1.ipynb): Tensor operations, `autograd` and `nn`. Due by **March 7th, 2022**.
- [Homework 2](https://github.com/glouppe/info8010-deep-learning/raw/master/homeworks/homework2.ipynb): Dataset, Dataloader, running on GPU, training a convolutional neural network. Due by **March 21st, 2022**.

Homeworks are optional. If submitted, each homework will account for 5% of the final grade.

## Project

See instructions in [`project.md`](https://github.com/glouppe/info8010-deep-learning/blob/master/project.md).

## Reading assignment

Your task is to read and summarize a major scientific paper in the field of deep learning. You are free to select one among the following three papers:

- Mildenhall et al, "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", 2020. [[Paper](https://arxiv.org/abs/2003.08934)]
- Ho et al, "Denoising Diffusion Probabilistic Models", 2020. [[Paper](https://arxiv.org/abs/2006.11239)]
- Jumper et al, "Highly accurate protein structure prediction with AlphaFold", 2021. [[Paper](https://www.nature.com/articles/s41586-021-03819-2)]

You should produce a report that summarizes the problem that is tackled by the paper and explains why it is challenging or important, from the perspective of the wider research context. The report should outline the main contributions and results with respect to the problem that is addressed. It should also include a critical discussion of the advantages and shortcomings of the contributions of the paper.
Further guidelines for writing a good summary can be found [here](https://web.stanford.edu/class/cs224n/project/project-proposal-instructions.pdf) (Section 2, "The summary").

Constraints:
- You can work in groups of maximum 3 students (the same as for the project).
- You report must be written in English.
- 2 pages (excluding references, if any).
- Formatted using the LaTeX template [`template-report.tex`](https://glouppe.github.io/info8010-deep-learning/template-report.tex).

Your report should be submitted by May 8. This is a **hard** deadline.

## Archives

- [2020-2021](https://github.com/glouppe/info8010-deep-learning/tree/v4-info8010-2021)
- [2019-2020](https://github.com/glouppe/info8010-deep-learning/tree/v3-info8010-2020)
- [2018-2019](https://github.com/glouppe/info8010-deep-learning/tree/v2-info8010-2019)