# Entropic Algorithms

This project started as a summer internship, under the supervision of Prof. Riccardo Zecchina, and later developed into the experimental part of my Bachelor thesis. The code under `tensorflow implementation` is what I used during my internship. The code under `pytorch implementation` is a reformulation in Pytorch, which I found more flexible and better suited to carry out further experiments for my thesis.

In order to fully understand the motivation and context for this project, it is best to have a glance at the structure of my thesis, of which chapter 4 corresponds to the experiments with the code of this repository. I paste here its brief introduction, the interested reader can delve deeper into the experimental setup and the wider theoretical picture in `Mattia Scardecchia Undergraduate Thesis.pdf`.

## Thesis Introduction

During the past decade, deep learning has taken the world by storm, with stunning ap-
plications to fields as diverse as computer vision, natural language processing, automatic
speech recognition, reinforcement learning, and bioinformatics [1]. Recently, the public
debate around the potentialities and risks of AI has exploded, inspired by a series of im-
pressive breakthroughs in computer vision and NLP which were open-sourced or released
to the public as an interactive demo [2] [3].

However, despite its spectacular successes, the theoretical understanding of deep learning
is seriously lagging behind. In fact, most recent breakthroughs have been guided almost
exclusively by intuition and experimentation, and they are often the result of clever en-
gineering and scaling up, without being grounded in a solid theoretical understanding.
One of the main reasons for the existing gap between theory and applications in deep
learning is the overwhelming complexity of these devices, which makes it extremely hard
to investigate their behaviour analytically. That being said, there do exist methods that,
in largely simplified settings, allow to study these systems and gain insights that can turn
out to be important for more complex architectures as well. In this direction, one of the
most promising lines of research employs tools originally developed in the framework of
statistical physics and complex systems to study the geometrical and statistical proper-
ties of the large-scale optimization problems involved in the training of neural networks.
In turn, the understanding of such properties can promote algorithmic advances in the
training of these devices.

In the first chapter of my thesis, I will start by providing a very brief overview of some of
the main ideas from statistical physics that are useful for the analysis of neural networks.
Then, I will present some results from the literature that used these techniques to find
evidence of the existence of large and connected dense clusters of solutions in the binary
perceptron learning problem, which are both rare and accessible to algorithms.

In the second chapter, I will show how insight from the analysis of the geometry of the
landscape of the binary perceptron has been used to derive very general and effective
algorithmic schemes, that turn out to be useful not only for shallow networks, but also
for deep learning. These are called entropic algorithms.

After that, in the third chapter, I will present some analytical results of my own, obtained
in the study of a recurrent network of neurons with random couplings. Using techniques
from statistical physics and spin glass theory, I computed the number of fixed points of
such a network under a zero-temperature metropolis dynamics that includes in the en-
ergy a self-coupling term, and studied the dependence of the results on the strength of
the latter.

Finally, in the fourth chapter, I will present the results of some numerical experiments
that I carried out with shallow and deep autoencoders in the random feature model. The
aim was twofold. First, I assessed the ability of such devices to estimate the latent di-
mensionality of data. Then, in this unsupervised setting, I compared the performance of
an entropic algorithm called replicated Adam with that of standard gradient-based meth-
ods like vanilla Adam, considering flatness in weight space, ability to denoise corrupted
patterns and reconstruction error on unseen patterns.

## Experimental results

Through extensive experimentation, I found that the solutions obtained with Replicated Adam are located in flatter regions of the loss landscape, and enjoy better generalization (reconstruction of unseen patterns from the same distribution) and denoising capabilities compared to those found by Adam. For a broader discussion of the experimental findings and their implications, I refer to my thesis.

![]()
![]()