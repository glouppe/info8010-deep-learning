class: middle, center, title-slide

# Deep Learning

Lecture 8: GPT

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

???

https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html

R: implement a GPT model from scratch (nanoGPT)
R: https://jaykmody.com/blog/gpt-from-scratch/ (picoGPT)

Scalability https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html
Emergent properties https://docs.google.com/presentation/d/1yzbmYB5E7G8lY2-KzhmArmPYwwl7o7CUST1xRZDUu1Y/edit?resourcekey=0-6_TnUMoKWCk_FN2BiPxmbw#slide=id.g16197112905_0_0
Emergent https://twitter.com/AndrewLampinen/status/1629534694617370625
https://t.co/WBw53cb3cz
chain of thoughts

https://twitter.com/peteflorence/status/1634256569335713799

Composition of models (langchain)

---

---

class: middle

.width-100[![](figures/lec7/gpt.png)]

.center[GPT, Radford et al. (2018)]

.footnote[Credits: Radford et al., [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), 2018.]

---

class: middle

Increasing the training data and the model size leads to significant improvement of transformer language models. These models are now .bold[the largest in deep learning].

.center.width-80[![](figures/lec7/plot-size.png)]

---

class: middle

GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text. The diversity of the dataset causes this simple goal to contain naturally occurring demonstrations of many tasks across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than 10X the amount of data.

.pull-right[Radford et al. (2019)]

.footnote[Credits: Francois Fleuret, [Deep Learning](https://fleuret.org/dlc/), UNIGE/EPFL.]

---

class: middle

The GPT-3 model has 175B parameters and was trained on 300B tokens from
various sources.

.pull-right[Brown et al. (2020)]

.footnote[Credits: Francois Fleuret, [Deep Learning](https://fleuret.org/dlc/), UNIGE/EPFL.]

---

# Disclaimer

Large, general language models could have significant **societal impacts**, and also have many near-term applications, including:
- AI writing assistants
- more capable dialogue agents
- unsupervised translation between languages
- better speech recognition systems.

.footnote[Credits: OpenAI [Better Language Models and Their Implications](https://openai.com/blog/better-language-models/).]

---

class: middle

However, we can also imagine the application of these models for **malicious purposes**, including the following:
- generate misleading news articles
- impersonate others online
- automate the production of abusive or faked content to post on social media
- automate the production of spam/phising content.

.alert[The public at large will need to become more skeptical of text they find online, just as the "deep fakes" phenomenon calls for more skepticism about images.]

.footnote[Credits: OpenAI [Better Language Models and Their Implications](https://openai.com/blog/better-language-models/).]