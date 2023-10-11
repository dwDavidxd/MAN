<div align="center">  

# Modeling Adversarial Noise for Adversarial Training
[![Paper](https://img.shields.io/badge/paper-ICML-green)](https://proceedings.mlr.press/v162/zhou22k/zhou22k.pdf)

</div>

The implementation of [Modeling Adversarial Noise for Adversarial Training](https://proceedings.mlr.press/v162/zhou22k/zhou22k.pdf) (ICML 2022).

Deep neural networks have been demonstrated to be vulnerable to adversarial noise, promoting the development of defense against adversarial attacks. Motivated by the fact that adversarial noise contains well-generalizing features and that the relationship between adversarial data and natural data can help infer natural data and make reliable predictions, in this paper, we study to model adversarial noise by learning the transition relationship between adversarial labels (i.e. the flipped labels used to generate adversarial data) and natural labels (i.e. the ground truth labels of the natural data). Specifically, we introduce an instance-dependent transition matrix to relate adversarial labels and natural labels, which can be seamlessly embedded with the target model (enabling us to model stronger adaptive adversarial noise). Empirical evaluations demonstrate that our method could effectively improve adversarial accuracy.


<p float="left" align="center">
<img src="arch.png" width="800" /> 
<figcaption align="center">
The illustration of our proposed Modeling Adversarial Noise-based method (MAN). $\boldsymbol{\hat{y}^{\prime}}$
</figcaption>
</p>
