"""
VAE_MNIST_FT.py - Fine-tuning Variational Autoencoder on MNIST (TensorFlow + Hugging Face Diffusers)

This script demonstrates how to fine-tune a pretrained KL Autoencoder (from Stability AI via Hugging Face Diffusers)
on the MNIST dataset. It covers the full pipeline from dataset preparation to training and visualization.

Workflow:
- Load MNIST training split using Hugging Face `datasets`.
- Preprocess images:
    * Convert grayscale → RGB (3 channels)
    * Resize to 256x256
    * Normalize pixel values to [-1, 1]
- Build a TensorFlow `tf.data` pipeline for batching and shuffling.
- Load a pretrained VAE (`stabilityai/sd-vae-ft-mse`) with trainable parameters.
- Define a training step:
    * Encode images into latent distributions
    * Sample latents using the reparameterization trick
    * Decode latents back into reconstructed images
    * Compute reconstruction loss (MSE) + KL divergence penalty
    * Backpropagate and update weights
- Run a small training loop with progress logging.
- Visualize original vs reconstructed images side by side.

Notes:
- This script is intended as a teaching/demo example; adjust `batch_size`, `epochs`, and `learning_rate`
  for more serious training.
- The KL weight is set small (0.001) to balance reconstruction fidelity with latent regularization.
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# `datasets` (Hugging Face) is used to load the MNIST dataset conveniently.
from datasets import load_dataset

# `TFAutoencoderKL` is a TF implementation of an Autoencoder with a latent
# distribution; this is what the Stable Diffusion family uses for latents.
from diffusers import TFAutoencoderKL

# ---------------------------------------------------------------------
# 1) Dataset loading and preprocessing
# ---------------------------------------------------------------------

# Load the MNIST training split using the Hugging Face `datasets` library.
# Each example is a dict with an "image" PIL image object and a "label".
dataset = load_dataset("mnist", split="train")


def preprocess(example):
    """
    Convert MNIST example to a float32 NumPy array, scale to [-1, 1],
    convert single-channel grayscale to 3-channel RGB, and resize to 256x256.

    Input:
      example: dict with key "image" (PIL.Image)
    Returns:
      dict with key "image" containing a float32 image in range [-1, 1]
           and shape (256, 256, 3).
    """
    # Convert PIL Image to NumPy float32 in [0, 255], then normalize to [0,1].
    image = np.array(example["image"]).astype("float32") / 255.0

    # MNIST images are grayscale (H, W). Many VAE implementations expect 3 channels,
    # so stack the single channel into 3 identical channels (RGB-like).
    image = np.stack([image, image, image], axis=-1)  # shape: (28, 28, 3)

    # Resize to 256x256 to match the VAE input size used by many Stable Diffusion VAEs.
    # `tf.image.resize` returns a Tensor; we keep it as a TensorFlow tensor here.
    image = tf.image.resize(image, (256, 256))

    # Rescale pixel values from [0, 1] → [-1, 1], which is commonly used by
    # diffusion/autoencoder models.
    image = image * 2.0 - 1.0

    return {"image": image}


# Apply preprocessing to the whole dataset, so each item becomes a dict with a
# preprocessed `image` Tensor.
dataset = dataset.map(preprocess)

# Instruct the dataset object to present the "image" column as TensorFlow tensors.
dataset.set_format(type="tensorflow", columns=["image"])

# ---------------------------------------------------------------------
# 2) Build the tf.data pipeline
# ---------------------------------------------------------------------

batch_size = 4

# Create a tf.data.Dataset from the array of images in the Hugging Face dataset.
# The object `dataset["image"]` is an array/tensor with shape (N, 256, 256, 3).
train_ds = tf.data.Dataset.from_tensor_slices(dataset["image"])

# Shuffle, batch, and prefetch for better performance during training.
train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------------------
# 3) Load the pretrained VAE
# ---------------------------------------------------------------------

# Load a pretrained TensorFlow VAE. The checkpoint "stabilityai/sd-vae-ft-mse"
# is an example fine-tuned VAE used in Stable Diffusion ecosystems.
# `from_pt=False` instructs the loader that the checkpoint is already in TF format.
vae = TFAutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    from_pt=False
)

# Ensure the VAE's variables are trainable so gradients will flow and updates occur.
vae.trainable = True

# Optimizer: small learning rate because this is a fine-tuning example.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

# ---------------------------------------------------------------------
# 4) Training step function
# ---------------------------------------------------------------------

@tf.function
def train_step(images):
    """
    A single training step on a batch of `images`:
      1) Encode images to a posterior distribution over latents.
      2) Sample latents from the posterior.
      3) Decode sampled latents to reconstruct images.
      4) Compute reconstruction MSE loss and a KL penalty from the posterior.
      5) Backpropagate and apply gradients.

    Returns:
      recon_loss: mean squared reconstruction error (scalar)
      kl_loss: mean KL divergence (scalar)
      total_loss: recon_loss + kl_weight * kl_loss
    """
    with tf.GradientTape() as tape:
        # Encode the images: produces an object (posterior) that contains a
        # `latent_dist` with sampling/kl utilities.
        posterior = vae.encode(images)

        # Sample from the posterior latent distribution (reparameterized if supported).
        # Note: `latent_dist.sample()` will typically return a Tensor of shape
        # (batch_size, latent_channels, latent_h, latent_w) or similar.
        z = posterior.latent_dist.sample()

        # Decode the sampled latents. `vae.decode(z)` typically returns an object
        # with `.sample` containing the reconstructed image Tensor.
        reconstructed = vae.decode(z).sample

        # Reconstruction loss: mean squared error between inputs and reconstructions.
        # We use `tf.reduce_mean` so the returned value is a scalar.
        recon_loss = tf.reduce_mean(tf.square(images - reconstructed))

        # KL loss: use the KL divergence computed by the posterior latent distribution.
        # The method/attribute used here should match what the `encode` returns.
        # We average it to get a scalar representing mean KL divergence over batch.
        kl_loss = tf.reduce_mean(posterior.latent_dist.kl())

        # Total loss includes a small weight on KL — matches the original script.
        kl_weight = 0.001
        total_loss = recon_loss + kl_weight * kl_loss

    # Compute gradients w.r.t. all trainable variables of the VAE.
    grads = tape.gradient(total_loss, vae.trainable_variables)

    # Apply gradients using the optimizer.
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))

    return recon_loss, kl_loss, total_loss


# ---------------------------------------------------------------------
# 5) Training loop
# ---------------------------------------------------------------------

epochs = 3  # small number for demonstration; increase for real training

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # Iterate over the tf.data training dataset
    for step, images in enumerate(train_ds):
        # Run a training step and get loss components
        recon_loss, kl_loss, total_loss = train_step(images)

        # Periodically print training progress
        if step % 100 == 0:
            print(
                f"Step {step} | "
                f"Recon: {recon_loss:.4f} | "
                f"KL: {kl_loss:.4f} | "
                f"Total: {total_loss:.4f}"
            )

# ---------------------------------------------------------------------
# 6) Visualize a few reconstructions
# ---------------------------------------------------------------------

# Take one batch from the dataset for visualization.
images = next(iter(train_ds))

# Encode → sample → decode to produce reconstructions (same pipeline as training).
posterior = vae.encode(images)
z = posterior.latent_dist.sample()
reconstructed = vae.decode(z).sample

# The model uses images in [-1, 1]; convert back to [0, 1] for display with matplotlib.
images_vis = (images + 1.0) / 2.0
recon_vis = (reconstructed + 1.0) / 2.0

# Plot original (top row) and reconstructed (bottom row) for the first 4 images.
plt.figure(figsize=(8, 4))
for i in range(4):
    # Original images
    plt.subplot(2, 4, i + 1)
    plt.imshow(images_vis[i])
    plt.axis("off")
    if i == 0:
        plt.title("Original")

    # Reconstructed images
    plt.subplot(2, 4, i + 5)
    plt.imshow(recon_vis[i])
    plt.axis("off")
    if i == 0:
        plt.title("Reconstructed")

plt.show()