
import pprint
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training import common_utils, train_state
from jax import random

AUTOTUNE = tf.data.AUTOTUNE


class TrainState(train_state.TrainState):
  batch_stats: Any


def preprocess(image, label):
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, label

def get_datasets():
    (train_data, test_data), info = tfds.load(
        'mnist', split=['train', 'test'], shuffle_files=True,
        as_supervised=True, with_info=True)

    train_data = train_data.map(preprocess, num_parallel_calls=AUTOTUNE) \
        .cache() \
        .shuffle(info.splits['train'].num_examples) \
        .batch(32, drop_remainder=True) \
        .prefetch(AUTOTUNE)
    test_data = test_data.map(preprocess, num_parallel_calls=AUTOTUNE) \
        .cache() \
        .batch(32) \
        .prefetch(AUTOTUNE)

    return train_data, test_data

class ConvNet(nn.Module):
    layers_per_scale: int
    base_channels: int
    channel_multiplier: float = 2.
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, training=True):
        channels = self.base_channels
        for _ in range(self.layers_per_scale): # 28x28
            x = nn.Conv(channels, (3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
        channels = int(channels*self.channel_multiplier)
        for i in range(self.layers_per_scale): # 14x14
            strides = (2, 2) if i == 0 else (1, 1)
            x = nn.Conv(channels, (3, 3), padding='SAME', strides=strides, use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
        channels = int(channels*self.channel_multiplier)
        for i in range(self.layers_per_scale): # 7x7
            strides = (2, 2) if i == 0 else (1, 1)
            x = nn.Conv(channels, (3, 3), padding='SAME', strides=strides, use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
        x = jnp.mean(x, (1, 2))
        x = nn.Dense(self.num_classes, use_bias=False)(x)
        return x


def get_model(layers_per_scale: int=3, base_channels: int=64, channel_multiplier: float=2.):
    return ConvNet(layers_per_scale, base_channels, channel_multiplier)


def get_optimizer(lr=0.0001, momentum=0.9):
    return optax.sgd(learning_rate=lr, momentum=momentum)


def cross_entropy_loss(logits, labels):
    labels = common_utils.onehot(labels, num_classes=10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(loss)


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        output, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats}, images, mutable=['batch_stats'])
        loss = cross_entropy_loss(output, labels)
        return loss, (output, new_model_state)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (logits, new_model_state)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    metrics = compute_metrics(logits, labels)
    return state, metrics


@jax.jit
def eval_step(state, images, labels):
    model_params = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(model_params, images, training=False, mutable=False)
    return compute_metrics(logits, labels)


def train_epoch(state, traindata, epoch):
    """Train for a single epoch."""

    batch_metrics = []
    for images, labels in traindata:
        state, metrics = train_step(state, images._numpy(), labels._numpy())
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}

    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    return state


def eval_model(state, testdata):
    test_metrics = []
    for images, labels in testdata:
        metrics = eval_step(state, images._numpy(), labels._numpy())
        test_metrics.append(metrics)

    batch_metrics_np = jax.device_get(test_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}

    print('TEST: loss: %.4f, accuracy: %.2f' % (
        epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

def run(epochs=10):
    rng = random.PRNGKey(0)

    traindata, testdata = get_datasets()
    model = get_model()
    modelkey, rng = random.split(rng)
    dummy = jnp.ones((1, 28, 28, 1))
    params = model.init(modelkey, dummy)
    optimizer = get_optimizer()
    state = TrainState.create(
        apply_fn=model.apply, params=params['params'], batch_stats=params['batch_stats'],
        tx=optimizer)

    for epoch in range(epochs):
        state = train_epoch(state, traindata, epoch)
        eval_model(state, testdata)


def check_env():
    device_counts = jax.device_count()
    print('Number of devices :', device_counts)
    devices = jax.devices()
    print('Devices           :', pprint.pformat(devices))


def main():
    check_env()
    run()

if __name__=='__main__':
    # see: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    tf.config.experimental.set_visible_devices([], "GPU")
    main()
