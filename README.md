# T-Prior Variational Autoencoder (VAE)

This repository contains an implementation of the **T-Prior Variational Autoencoder (VAE)**, a model designed to mitigate the overregularization issue observed in traditional VAEs. The method leverages the unique properties of the t-distribution to improve generalization performance, particularly for outlier pattern identification.

---

## Introduction

Variational Autoencoders (VAEs) often suffer from overregularization due to the strong influence of the KL divergence term between the Gaussian prior and the encoder’s Gaussian distribution. This results in reduced generalization performance.

To address this, we propose a **T-Prior VAE**, which uses a t-distributed latent space. The t-distribution, with its slower tail decay compared to the Gaussian, results in a lower KL divergence. This helps balance the trade-off between reconstruction accuracy and generalization.

---

## Overregularization of traditional VAEs
![image](https://github.com/user-attachments/assets/274c5c73-3812-49a1-a594-f0ccca597a89, alt="LaTeX Equation" width="300")

Caused by the strong influence of the KL divergence term between the gaussian prior and the gaussian encoder's distribution

Recap variational inference: 

$$log(p(\textbf{x}))=E_{q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})}[log(p_{\boldsymbol{\theta}}(\textbf{x}|\textbf{z}))]-KL(q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})||p(\textbf{z}))+KL(q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})||p_{\boldsymbol{\theta}}(\textbf{z}|\textbf{x}))$$

Define loss function as: 

$$Loss=-E_{q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})}[log(p_{\boldsymbol{\theta}}(\textbf{x}|\textbf{z}))]+KL(q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})||p(\textbf{z})):= Loss_{RE} + Loss_{KL}$$

---

## t-distribution
Note that t-distribution belongs to the location-scale family: $Y=\mu+\sigma X, \ f_Y(y)=\frac{1}{\sigma}f_X(\frac{x-\mu}{\sigma}) $

probability density function(pdf)s of gaussian and t distribution: 

$$pdf_{N(\mu,\sigma^2)}(x)=\frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{1}{2}(\frac{x-\mu}{\sigma})^2)$$

$$pdf_{t(\mu,\sigma^2,\nu)}(x)=\frac{\Gamma(\frac{\nu+1}{2})}{\sigma\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})}(1+\frac{1}{\nu}(\frac{x-\mu}{\sigma})^2)^{-\frac{\nu+1}{2}}$$

The tail decay of the t-distribution is significantly slower compared to the Gaussian distribution.

therefore, KL divergence between a t-distribution is lower then that of a gaussian distribution

***here's why in detail:*** 

$$KL(q(x)||p(x))=\int q(x)log\frac{q(x)}{p(x)}$$

In distributions with lighter tails, such as the Gaussian, tail differences significantly influence the KL divergence. The rapid exponential decay of these tails magnifies the relative values of the probability density function

Therefore, the KL divergence between the t-distribution is relatively smaller compared to the Gaussian distribution

**Based on the above reasoning, we expect that a VAE with t-distributed latent space would improve generalization performance, particularly in identifying outlier patterns**


---


## t-prior VAE
Gaussian VAE:
![image](https://github.com/user-attachments/assets/beacf0ce-b669-412b-9f3c-d7a6225b75dd)

To mitigate the aforementioned problem, we propose a new model, the t-prior VAE, defined as follows:
![image](https://github.com/user-attachments/assets/d5183570-bf8e-4827-a62b-9e55178ec543)


---


## loss derivation for t-prior VAE
Given that the decoder assumption remains unchanged, $Loss_{RE}\cong-\frac{1}{L}\sum^L_{i=1}log(p_{\boldsymbol{\theta}}(\textbf{x}^{(i)}|\textbf{z}^{(i)}))$ by Monte Carlo approximation.
we now derive the KL divergence between the t-distributed encoder and the prior

$$Loss_{KL}=KL(q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})||p(\textbf{z}))=\int q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})log\frac{q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})}{p(\textbf{z})}=E_{q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})}[log(\frac{q_{\boldsymbol{\phi}}(\textbf{z}|\textbf{x})}{p(\textbf{z})})]$$

for univariate case: $p(z)\sim t(0,1,\nu), \ q(z|\textbf{x}) \sim t(\mu_q,\sigma_q,\nu)$ 

$$E_{q(z|x)}[log(\frac{q(z|x)}{p(z)})]=-log(\sigma_q)+E_{q_(z|x)}[-\frac{\nu+1}{2}log(1+\frac{1}{\nu}(\frac{z-\mu_q}{\sigma_q})^2)+\frac{\nu+1}{2}log(1+\frac{1}{\nu}z^2)]$$

Applying the Monte Carlo approximation,

$$E_{q(z|x)}[log(\frac{q(z|x)}{p(z)})] \cong-log(\sigma_q)+ \frac{1}{L}\sum^L_{i=1}[-\frac{\nu+1}{2}log(1+\frac{1}{\nu}(\frac{z^{(i)}-\mu_q}{\sigma_q})^2)+\frac{\nu+1}{2}log(1+\frac{1}{\nu}z^{(i)^2})]$$

Thus, the loss function for the univariate distribution becomes as

$$Loss = Loss_{RE} + Loss_{KL} \cong-\frac{1}{L}\sum^L_{i=1}log(p(x^{(i)}|z^{(i)}))-log(\sigma_q)+\frac{1}{L}\sum^L_{i=1}[-\frac{\nu+1}{2}log(1+\frac{1}{\nu}(\frac{z^{(i)}-\mu_q}{\sigma_q})^2)+\frac{\nu+1}{2}log(1+\frac{1}{\nu}z^{(i)^2})]$$

setting L=1,

$$Loss=\frac{1}{N}\sum^N_{i=1}[\sum^d_{j=1}\{x_{i,j}log(p_{i,j})+(1-x_{i,j})log(1-p_{i,j})\}+\sum^m_{j=1}\{-log(\sigma_{i,j})-\frac{\nu+1}{2}log(1+\frac{1}{\nu}(\frac{z_{i,j}-\mu_{i,j}}{\sigma_{i,j}})^2)+\frac{\nu+1}{2}log(1+\frac{1}{\nu}z_{i,j}^2)]\}$$

---


## backprop derivation for t-prior VAE
![image](https://github.com/user-attachments/assets/56d6b149-d0f9-432b-8f2b-233809a7f851)

the only change for calculating backprop calculation from traditional VAE model is the gradient of latent parameters($\mu$ and $log(\sigma^2)$).

---

## Implementation Details

- **Decoder**: Remains consistent with standard VAEs.(bernoulli decoder)
- **KL Divergence**: since we can get KL divergenc between t-distributions in closed form, approximated by monte calro approximations.
- **Reparameterization trick**: it is available for t-distribution since it belongs to location-scale family.
- **hyperparameter**: here the degree of freedom of t-distributed prior and encoder is treated as hyperparameter. If $\nu is too large, the model behaves like a Gaussian VAE, potentially leading to overregularization. Conversely, if \nu is too small, the model may suffer from poor learning due to sampling from a heavy-tailed distribution.
- **generating representatives of t-distribution**: 


---

### Prerequisites
- Python 3.x
- PyTorch
- Libraries: NumPy
