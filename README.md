# Fine-Tuning a Variational Autoencoder (VAE) on MNIST  
**TensorFlow + Hugging Face Diffusers**

This repository demonstrates how to **fine-tune a pretrained Variational Autoencoder (VAE)** from Hugging Face Diffusers on the MNIST dataset using **TensorFlow**.

Unlike inference-only demos, this project walks through the **entire VAE training pipeline**.

---

## 🚀 What This Project Covers

- Loading MNIST with Hugging Face `datasets`
- Image preprocessing for diffusion-style VAEs
- Building an efficient `tf.data` pipeline
- Loading a pretrained Stable Diffusion VAE
- Implementing:
  - Encoder → latent distribution
  - Reparameterization sampling
  - Decoder reconstruction
  - Reconstruction loss (MSE)
  - KL divergence regularization
- Backpropagation & optimizer updates
- Visual comparison of originals vs reconstructions

---

## 🧠 Why Fine-Tune a Pretrained VAE?

Fine-tuning helps you:
- Adapt VAEs to **custom datasets**
- Understand **VAE loss decomposition**
- Learn how Stable Diffusion models are trained internally
- Bridge theory → production-grade models

---

## 📦 Requirements

pip install tensorflow datasets diffusers matplotlib

## 🏗️ Training Architecture

Loss Function

Total Loss = Reconstruction Loss + KL Weight × KL Divergence

Reconstruction Loss: Mean Squared Error (MSE)

KL Divergence: Latent regularization

KL Weight: 0.001 (kept small for stability)

## 🔍 Key Concepts Explained in Code

Variational inference

Latent probability distributions

KL divergence

TensorFlow GradientTape

Diffusion-style VAEs

Fine-tuning pretrained models safely

## 🖼️ Visualization

After training, the script shows:

Top row: Original MNIST images

Bottom row: Reconstructed images after fine-tuning

## 🧑‍🎓 Who Should Use This Repo?

Learners studying VAEs beyond theory

Engineers preparing for diffusion model training

TensorFlow users wanting Hugging Face examples

Researchers exploring latent-space learning

## ⚠️ Important Notes

MNIST is resized to 256×256 RGB

VAE weights are trainable

Training epochs are kept small for demo purposes

Increase batch size & epochs for better results

## 📜 License

MIT License

## ⭐ Support

If this repo helped you understand VAEs or Diffusers:

### ⭐ Star the repo

### 🧠 Share it with other ML learners
