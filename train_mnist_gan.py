
import pprint
from functools import partial
from typing import Any

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training import common_utils, train_state
from jax import random
from PIL import Image

AUTOTUNE = tf.data.AUTOTUNE


class TrainState(train_state.TrainState):
    batch_stats: Any

def preprocess(image, label):
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, label

def get_dataset():
    train_data, info = tfds.load(
        'mnist', split='train', shuffle_files=True,
        as_supervised=True, with_info=True)

    train_data = train_data.map(preprocess, num_parallel_calls=AUTOTUNE) \
        .cache() \
        .shuffle(info.splits['train'].num_examples) \
        .batch(32, drop_remainder=True) \
        .prefetch(AUTOTUNE)

    return train_data

class Discriminator(nn.Module):
    layers_per_scale: int
    base_channels: int
    channel_multiplier: float = 2.

    @nn.compact
    def __call__(self, x, training=True):
        channels = self.base_channels
        for _ in range(self.layers_per_scale): # 28x28
            x = nn.Conv(channels, (3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        channels = int(channels*self.channel_multiplier)
        for i in range(self.layers_per_scale): # 14x14
            strides = (2, 2) if i == 0 else (1, 1)
            x = nn.Conv(channels, (3, 3), padding='SAME', strides=strides, use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        channels = int(channels*self.channel_multiplier)
        for i in range(self.layers_per_scale): # 7x7
            strides = (2, 2) if i == 0 else (1, 1)
            x = nn.Conv(channels, (3, 3), padding='SAME', strides=strides, use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = jnp.mean(x, (1, 2))
        x = nn.Dense(1, use_bias=False)(x)
        return x


class Generator(nn.Module):
    layers_per_scale: int
    latent_dim: int
    base_channels: int
    channels_multiplier: float=2.

    @nn.compact
    def __call__(self, x, training=True) -> Any:
        channels = int(self.base_channels * self.channels_multiplier ** 2)
        x = nn.Dense(channels*7*7, use_bias=False)(x)
        x = jnp.reshape(x, (-1, 7, 7, channels))
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        for _ in range(self.layers_per_scale-1): # 7*7
            x = nn.Conv(channels, (3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        channels = int(channels / self.channels_multiplier)
        x = nn.ConvTranspose(channels, (4, 4), (2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        for _ in range(self.layers_per_scale-1): # 14*14
            x = nn.Conv(channels, (3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        channels = int(channels / self.channels_multiplier)
        x = nn.ConvTranspose(channels, (4, 4), (2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        for _ in range(self.layers_per_scale-1): # 28*28
            x = nn.Conv(channels, (3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(1, (3, 3), padding='SAME', use_bias=False)(x)
        x = jax.nn.tanh(x)
        return x


def get_models(latent_dim, layers_per_scale=2, base_channels=32):
    D = Discriminator(layers_per_scale, base_channels)
    G = Generator(layers_per_scale, latent_dim, base_channels)
    return D, G

def get_optimizer(lr=0.0002, beta1=0.5, beta2=0.999):
    return optax.adam(learning_rate=lr, b1=beta1, b2=beta2)

@jax.vmap
def bce_w_logits(logits, labels):
    return jnp.maximum(logits, 0) - logits * labels + jnp.log(1 + jnp.exp(-jnp.abs(logits)))

def real_loss(logits):
    ones = jnp.ones(logits.shape[0], dtype=logits.dtype)
    loss = bce_w_logits(logits, ones)
    return jnp.mean(loss)

def fake_loss(logits):
    zeros = jnp.zeros(logits.shape[0], dtype=logits.dtype)
    loss = bce_w_logits(logits, zeros)
    return jnp.mean(loss)

def d_loss(real_prob, fake_prob):
    r_loss = real_loss(real_prob)
    f_loss = fake_loss(fake_prob)
    return r_loss + f_loss

def g_loss(fake_prob):
    return real_loss(fake_prob)


def d_step(d_state, g_state, real, z):
    def loss_fn(params):
        fake, new_g_params = g_state.apply_fn(
            {'params': g_state.params, 'batch_stats': g_state.batch_stats}, z, mutable=['batch_stats'])
        real_prob, new_d_params = d_state.apply_fn(
            {'params': params, 'batch_stats': d_state.batch_stats}, real, mutable=['batch_stats'])
        fake_prob, new_d_params = d_state.apply_fn(
            {'params': params, 'batch_stats': new_d_params['batch_stats']}, fake, mutable=['batch_stats'])
        loss = d_loss(real_prob, fake_prob)
        return loss, (new_d_params, new_g_params, fake)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (new_d_params, new_g_params, fake)), grads = grad_fn(d_state.params)
    d_state = d_state.apply_gradients(grads=grads, batch_stats=new_d_params['batch_stats'])
    g_state = g_state.replace(batch_stats=new_g_params['batch_stats'])
    return d_state, g_state, fake

def g_step(d_state, g_state, z):
    def loss_fn(params):
        fake, new_g_params = g_state.apply_fn(
            {'params': params, 'batch_stats': g_state.batch_stats}, z, mutable=['batch_stats'])
        fake_prob, new_d_params = d_state.apply_fn(
            {'params': d_state.params, 'batch_stats': d_state.batch_stats}, fake, mutable=['batch_stats'])
        loss = g_loss(fake_prob)
        return loss, (new_g_params, new_d_params)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (new_g_params, new_d_params)), grads = grad_fn(g_state.params)
    g_state = g_state.apply_gradients(grads=grads, batch_stats=new_g_params['batch_stats'])
    d_state = d_state.replace(batch_stats=new_d_params['batch_stats'])
    return g_state, d_state

@partial(jax.jit, static_argnums=4)
def step(d_state, g_state, real, rng, latent_dim):
    z_key, nextrng = random.split(rng)

    z = random.normal(z_key, (real.shape[0], latent_dim))
    d_state, g_state, fake = d_step(d_state, g_state, real, z)
    g_state, d_state = g_step(d_state, g_state, z)

    return d_state, g_state, fake, nextrng


def save_image(image, filename):
    if not isinstance(image, np.ndarray):
        image = jax.device_get(image)
    image = (image + 1) * 127.5 # [0, 255]
    image = image.astype(np.uint8)
    nrow = int(image.shape[0]**0.5)
    size = image.shape[1]
    image_grid = np.zeros((nrow*size, nrow*size, 1), dtype=np.uint8)
    y = 0
    for i in range(nrow**2):
        sample = image[i]
        x = i - y * nrow
        image_grid[size*y:size*(y+1), size*x:size*(x+1)] = sample
        y = (i + 1) // nrow
    pil_image = Image.fromarray(image_grid.repeat(3, -1))
    pil_image.save(filename)

def run():
    epochs = 100
    latent_dim = 128

    traindata = get_dataset()
    rng = random.PRNGKey(0)
    d_key, g_key, rng = random.split(rng, 3)
    d_input, g_input = jnp.ones((1, 28, 28, 1)), jnp.ones((1, latent_dim))
    D, G = get_models(latent_dim)
    d_params = D.init(d_key, d_input)
    g_params = G.init(g_key, g_input)
    d_optim = get_optimizer()
    g_optim = get_optimizer()
    d_state = TrainState.create(
        apply_fn=D.apply, params=d_params['params'], batch_stats=d_params['batch_stats'],
        tx=d_optim)
    g_state = TrainState.create(
        apply_fn=G.apply, params=g_params['params'], batch_stats=g_params['batch_stats'],
        tx=g_optim)

    for epoch in range(epochs):
        for images, _ in traindata:
            d_state, g_state, fake, rng = step(d_state, g_state, images._numpy(), rng, latent_dim)
        save_image(fake, f'./images/mnist_epoch[{epoch:03}].jpg')
        print(epoch, 'done')

if __name__=='__main__':
    tf.config.experimental.set_visible_devices([], "GPU")
    run()
