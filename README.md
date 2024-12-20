# T-Prior Variational Autoencoder (VAE)

This repository contains an implementation of the **T-Prior Variational Autoencoder (VAE)**, a model designed to mitigate the overregularization issue observed in traditional VAEs. The method leverages the unique properties of the t-distribution to improve generalization performance, particularly for outlier pattern identification.

---

## Introduction

Variational Autoencoders (VAEs) often suffer from overregularization due to the strong influence of the KL divergence term between the Gaussian prior and the encoder’s Gaussian distribution. This results in reduced generalization performance.

To address this, we propose a **T-Prior VAE**, which uses a t-distributed latent space. The t-distribution, with its slower tail decay compared to the Gaussian, results in a lower KL divergence. This helps balance the trade-off between reconstruction accuracy and generalization.

---

## Overregularization of traditional VAEs
![image](https://github.com/user-attachments/assets/274c5c73-3812-49a1-a594-f0ccca597a89)


---

## t-distribution

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
