
from __future__ import annotations

import argparse
import os
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

from prng_sequence import PRNGKeySequence
from train_mnist_gan import AUTOTUNE

AUTOTUNE = tf.data.AUTOTUNE


def download_dataset(data_dir='./cifar10/'):
    if not os.path.exists(data_dir):
        tfds.load('cifar10', download=True, data_dir=data_dir)


def preprocess(image, label):
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, label

def get_datasets(data_dir='./cifar10/'):
    (traindata, testdata), info = tfds.load(
        'cifar10', split=['train', 'test'], shuffle_files=True, data_dir=data_dir,
        as_supervised=True, with_info=True)

    traindata = traindata.map(preprocess, num_parallel_calls=AUTOTUNE) \
        .cache() \
        .shuffle(info.splits['train'].num_examples) \
        .batch(32, drop_remainder=True) \
        .prefetch(AUTOTUNE)
    testdata = testdata.map(preprocess, num_parallel_calls=AUTOTUNE) \
        .cache() \
        .batch(32) \
        .prefetch(AUTOTUNE)

    return traindata, testdata



class BasicBlock(nn.Module):
    channels: int
    stride: tuple[int]

    @nn.compact
    def __call__(self, x, training=True):
        skip = x
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = jax.nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), self.stride, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = jax.nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), use_bias=False)(x)

        if skip.shape != x.shape:
            skip = nn.Conv(self.channels, (1, 1), self.stride, use_bias=False)(skip)

        x = x + skip
        return x

class ResNet(nn.Module):
    num_blocks: tuple[int]
    base_channels: int
    channel_mutiplier: float=2.
    num_classes: int=10

    @nn.compact
    def __call__(self, x, training=True):
        channels = self.base_channels
        x = nn.Conv(channels, (3, 3), use_bias=False)(x) # we do no pooling in stem
        for i, blocks in enumerate(self.num_blocks):
            for j in range(blocks):
                stride = (2, 2) if j == 0 and i != 0 else (1, 1)
                x = BasicBlock(channels, stride)(x, training)
            channels = int(channels * self.channel_mutiplier)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = jax.nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, use_bias=False)(x)
        return x

def get_model(num_blocks=(2, 2, 4, 2), base_channels=32):
    return ResNet(num_blocks, base_channels)

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

class TrainState(train_state.TrainState):
    batch_stats: Any

def run(args):
    traindata, testdata = get_datasets(args.data_dir)
    keyseq = PRNGKeySequence(0)

    model = get_model(tuple(args.num_blocks), args.base_channels)
    dummy = jnp.ones((1, 32, 32, 3))
    params = model.init(next(keyseq), dummy)
    optimizer = get_optimizer(args.lr, args.momentum)
    state = TrainState.create(
        apply_fn=model.apply, params=params['params'], batch_stats=params['batch_stats'],
        tx=optimizer)

    for epoch in range(args.epochs):
        state = train_epoch(state, traindata, epoch)
        eval_model(state, testdata)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',      default='./cifar10/')
    parser.add_argument('--num-blocks',    default=[2, 2, 4, 2], type=int, nargs='+')
    parser.add_argument('--base-channels', default=32, type=int)
    parser.add_argument('--epochs',        default=100, type=int)
    parser.add_argument('--lr',            default=0.0001, type=float)
    parser.add_argument('--momentum',      default=0.9, type=float)
    return parser.parse_args()

def main():
    args = get_args()
    download_dataset(args.data_dir)
    run(args)

if __name__=='__main__':
    tf.config.experimental.set_visible_devices([], "GPU")
    main()
