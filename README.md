# T-Prior Variational Autoencoder (VAE)

This repository contains an implementation of the **T-Prior Variational Autoencoder (VAE)**, a model designed to mitigate the overregularization issue observed in traditional VAEs. The method leverages the unique properties of the t-distribution to improve generalization performance, particularly for outlier pattern identification.

---

## Introduction

Variational Autoencoders (VAEs) often suffer from overregularization due to the strong influence of the KL divergence term between the Gaussian prior and the encoder’s Gaussian distribution. This results in reduced generalization performance.

To address this, we propose a **T-Prior VAE**, which uses a t-distributed latent space. The t-distribution, with its slower tail decay compared to the Gaussian, results in a lower KL divergence. This helps balance the trade-off between reconstruction accuracy and generalization.

---

## Overregularization of traditional VAEs
![image](https://github.com/user-attachments/assets/274c5c73-3812-49a1-a594-f0ccca597a89)

Caused by the strong influence of the KL divergence term between the gaussian prior and the gaussian encoder's distribution

Recap variational inference: $log(p(\textbf{x}))=E_{q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})}[log(p_{\boldsymbol{\theta}}(\textbf{x}|\textbf{z}))]-KL(q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})||p(\textbf{z}))+KL(q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})||p_{\boldsymbol{\theta}}(\textbf{z}|\textbf{x}))$

Define loss function as: $Loss=-E_{q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})}[log(p_{\boldsymbol{\theta}}(\textbf{x}|\textbf{z}))]+KL(q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})||p(\textbf{z})):= Loss_{RE} + Loss_{KL}$

---

## t-distribution
Note that t-distribution belongs to the location-scale family: $Y=\mu+\sigma X, \ f_Y(y)=\frac{1}{\sigma}f_X(\frac{x-\mu}{\sigma}) $

probability density function(pdf)s of gaussian and t distribution: 
$pdf_{t(\mu,\sigma^2,\nu)}(x)=\frac{\Gamma(\frac{\nu+1}{2})}{\sigma\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})}(1+\frac{1}{\nu}(\frac{x-\mu}{\sigma})^2)^{-\frac{\nu+1}{2}}$
$pdf_{N(\mu,\sigma^2)}(x)=\frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{1}{2}(\frac{x-\mu}{\sigma})^2)$

The tail decay of the t-distribution is significantly slower compared to the Gaussian distribution.

therefore, KL divergence between a t-distribution is lower then that of a gaussian distribution

here's why in detail: $KL(q(x)||p(x))=\int q(x)log\frac{q(x)}{p(x)}$

In distributions with lighter tails, such as the Gaussian, tail differences significantly influence the KL divergence. The rapid exponential decay of these tails magnifies the relative values of the probability density function

Therefore, the KL divergence between the t-distribution is relatively smaller compared to the Gaussian distribution

**Based on the above reasoning, we expect that a VAE with t-distributed latent space would improve generalization performance, particularly in identifying outlier patterns**




---


## loss derivation for t-prior VAE

## backprop derivation for t-prior VAE



---

## Implementation Details

- **Backpropagation**: Derivations account for the t-distribution's characteristics.
- **Decoder**: Remains consistent with standard VAEs.
- **KL Divergence**: Explicitly derived for the t-prior.

---

### Prerequisites
- Python 3.x
- TensorFlow/PyTorch (specify your choice)
- Libraries: NumPy, SciPy
