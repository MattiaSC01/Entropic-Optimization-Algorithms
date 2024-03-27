# Project Goal

The goal of this project was twofold. First, investigating the relationship between the flatness of the loss landscape around minima and the ability to generalize in deep unsupervised learning; second, evaluating the effectiveness of entropic optimization algorithms in this setting, compared to standard gradient-based methods.

For a detailed description of the background, experimental setting and results I refer to the manuscript of my undergraduate thesis, available as `Mattia Scardecchia Undergraduate Thesis.pdf`.

# Installation

To install the project, from the root directory do:

```
pip install -r requirements.txt
pip install -e .
```

This will install the most recent version of the codebase, written in pytorch and available under `src/ae`. Scripts and notebooks inside the `scripts` and `notebooks` directories assume you have performed these steps.
There are also two older versions of the code, one of which is in tensorflow, under `src/older code`.

# Motivation

It was shown analytically, in the supervised setting, that in the loss landscape of shallow neural networks there exist large, connected and dense clusters of good configurations, which turn out to be both rare and accessible to search algorithms.
These configurations turn out to generalize better to unseen patterns, and to better tolerate noisy inputs compared to typical solutions. These findings were confirmed empirically in deep networks, which are intractable by the current methods of spin glass theory.

Inspired by these results, general algorithmic schemes that explicitly target these rare clusters in an effort to improve generalization were devised. These are sometimes called entropic algorithms (e.g. replicated SGD, entropy SGD, SAM, SWA), and have been shown empirically to improve generalization in classification tasks.

In this project, I considered the important setting of unupervised learning, which is increasingly relevant in applications given the striking success of generative models, and I investigated whether similar phenomena to the supervised case can be observed in this scenario.

# Experimental Results

I carried out extensive expriments with shallow and deep autoencoders using a variety of architectures and datasets, comparing entropic algorithms such as SAM and replicated Adam with standard gradient-based optimizers. I found that entropic algorithms tend to find solutions lying in flatter regions of the loss landscape, which incur a smaller reconstruction error on unseen patterns and are better at denoising corrupted patterns.

![](https://github.com/MattiaSC01/ReplicatedSGD/blob/main/figures/Screenshot%202023-10-15%20at%2021.41.08.png)
![](https://github.com/MattiaSC01/ReplicatedSGD/blob/main/figures/Screenshot%202023-10-15%20at%2021.41.48.png)
