# INFO8010 - Deep Learning

Lectures for INFO8010 - Deep Learning, ULiège, Spring 2023.

- Instructor: Gilles Louppe
- Teaching assistants: Arnaud Delaunoy, François Rozet, Yann Claes, Victor Dachet
- When: Spring 2023, Friday 8:30 AM
- Classroom: R3 / B28

## Agenda

| Date | Topic |
| --- | --- |
| February 10 | [Course syllabus](https://glouppe.github.io/info8010-deep-learning/?p=course-syllabus.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/course-syllabus.pdf)] [[video](https://www.youtube.com/watch?v=51UOdB199Nk)]<br>Lecture 0: [Introduction](https://glouppe.github.io/info8010-deep-learning/?p=lecture0.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec0.pdf)] [[video](https://www.youtube.com/watch?v=-Ee-Z311a3k)]<br>Lecture 1: [Fundamentals of machine learning](https://glouppe.github.io/info8010-deep-learning/?p=lecture1.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec1.pdf)]  [[video](https://www.youtube.com/watch?v=GwpG0sHPklE)] |
| February 17 | Lecture 2: [Multi-layer perceptron](https://glouppe.github.io/info8010-deep-learning/?p=lecture2.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec2.pdf)] [[video](https://www.youtube.com/watch?v=OF6AkE9Fnjc)] [[code 1](https://github.com/glouppe/info8010-deep-learning/blob/master/code/lec2-space-stretching.ipynb), [code 2](https://github.com/glouppe/info8010-deep-learning/blob/master/code/lec2-spiral-classification.ipynb)] |
| February 24 | Lecture 3: [Automatic differentiation](https://glouppe.github.io/info8010-deep-learning/?p=lecture3.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec3.pdf)] [[video](https://youtu.be/fD047xXpSfI)] [[code](https://github.com/glouppe/info8010-deep-learning/blob/master/code/lec3-autodiff.ipynb)] |
| March 3 | Lecture 4: [Training neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture4.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec4.pdf)] [[video](https://youtu.be/G7qw620V_3g)]  |
| March  6 | Deadline for Homework 1 | 
| March 10 | Lecture 5: [Convolutional neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture5.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec5.pdf)] [[video](https://youtu.be/54WShJMWYo0)] [[code](https://github.com/glouppe/info8010-deep-learning/blob/master/code/lec5-convnet.ipynb)] |
| March 17 | Lecture 6: [Computer vision](https://glouppe.github.io/info8010-deep-learning/?p=lecture6.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec6.pdf)] [[video](https://youtu.be/cfZGfJaLRxA)] |
| March 20 | Deadline for Homework 2 |
| March 20 | Deadline for the project proposal |
| March 24 | Lecture 7: [Attention and transformers](https://glouppe.github.io/info8010-deep-learning/?p=lecture7.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec7.pdf)] [[video](https://youtu.be/cwFE1pLld-g)] |
| March 31 | Lecture 8: [GPT](https://glouppe.github.io/info8010-deep-learning/?p=lecture8.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec8.pdf)] [[code](https://github.com/glouppe/info8010-deep-learning/blob/master/code/gpt/)] |
| April 21 | Lecture 9: [Graph neural networks](https://glouppe.github.io/info8010-deep-learning/?p=lecture9.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec9.pdf)] |
| April 28 | Lecture 10: [Uncertainty](https://glouppe.github.io/info8010-deep-learning/?p=lecture10.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec10.pdf)] [[video](https://youtu.be/AxJBY9eRTL4)]<br>Tutorial: Weight and Biases (Thomas Capelle, ML engineer at `wandb.ai`) |
| May 5 | Lecture 11: [Auto-encoders and variational auto-encoders](https://glouppe.github.io/info8010-deep-learning/?p=lecture11.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec11.pdf)] [[video](https://youtu.be/6gWeyUZtHWs)] [[code](https://github.com/glouppe/info8010-deep-learning/blob/master/code/lec11-vae.ipynb)] |
| May 12 | Lecture 12: [Diffusion models](https://glouppe.github.io/info8010-deep-learning/?p=lecture12.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/lec12.pdf)] |
| May 19 | Deadline for the reading assignment |
| May 19 | Deadline for the project |

## Homeworks

The goal of these two assignments is to get you familiar with the PyTorch library. You can find the installation instructions in the [Homeworks](./homeworks) folder.
Each homework should be done in groups of 2 or 3 (the same as for the project) and must be submitted **before 23:59 on the due date**.
Homeworks should be submitted on Gradescope.

- [Homework 1](https://github.com/glouppe/info8010-deep-learning/raw/master/homeworks/homework1.ipynb): Tensor operations, `autograd` and `nn`. Due by **March 6, 2023**.
- [Homework 2](https://github.com/glouppe/info8010-deep-learning/raw/master/homeworks/homework2.ipynb): Dataset, Dataloader, running on GPU, training a convolutional neural network. Due by **March 20, 2023**.

Homeworks are optional. If submitted, each homework will account for 5% of the final grade.

## Project

See instructions in [`project.md`](https://github.com/glouppe/info8010-deep-learning/blob/master/project.md).

## Reading assignment

Your task is to read and summarize a major scientific paper in the field of deep learning. You are free to select one among the following three papers:

- Rombach et al, "High-Resolution Image Synthesis with Latent Diffusion Models", 2022. [[Paper](https://arxiv.org/abs/2112.10752)]
- Chen et al, "Evaluating Large Language Models Trained on Code", 2021 [[Paper](https://arxiv.org/abs/2107.03374)]
- Jumper et al, "Highly accurate protein structure prediction with AlphaFold", 2021. [[Paper](https://www.nature.com/articles/s41586-021-03819-2)]

You should produce a report that summarizes the problem that is tackled by the paper and explains why it is challenging or important, from the perspective of the wider research context. The report should outline the main contributions and results with respect to the problem that is addressed. It should also include a critical discussion of the advantages and shortcomings of the contributions of the paper.
Further guidelines for writing a good summary can be found [here](https://web.stanford.edu/class/cs224n/project/project-proposal-instructions.pdf) (Section 2, "The summary").

Constraints:
- You can work in groups of maximum 3 students (the same as for the project).
- You report must be written in English.
- 2 pages (excluding references, if any).
- Formatted using the LaTeX template [`template-report.tex`](https://glouppe.github.io/info8010-deep-learning/template-report.tex).

Your report should be submitted by **May 19** on Gradescope. This is a **hard** deadline.

## Archives

### Previous editions

- [2021-2022](https://github.com/glouppe/info8010-deep-learning/tree/v5-info8010-2022)
- [2020-2021](https://github.com/glouppe/info8010-deep-learning/tree/v4-info8010-2021)
- [2019-2020](https://github.com/glouppe/info8010-deep-learning/tree/v3-info8010-2020)
- [2018-2019](https://github.com/glouppe/info8010-deep-learning/tree/v2-info8010-2019)

### Archived lectures

Due to progress in the field, some of the lectures have become less relevant. However, they are still available for those who are interested.

| Topic |
| --- |
| [Recurrent neural networks](https://glouppe.github.io/info8010-deep-learning/?p=archives-lecture-rnn.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/archives-lec-rnn.pdf)] [[video](https://youtu.be/qnux5dg5wZ4)] |
| [Generative adversarial networks](https://glouppe.github.io/info8010-deep-learning/?p=archives-lecture-gan.md) [[PDF](https://glouppe.github.io/info8010-deep-learning/pdf/archives-lec-gan.pdf)] [[video](https://youtu.be/cM6m1eHY5FI)] |
