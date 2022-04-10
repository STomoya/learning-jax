
import pprint

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training import common_utils, train_state
from jax import random


def preprocess(image, label):
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    image = tf.reshape(image, (-1, ))
    return image, label

def get_datasets():
    (train_data, test_data), info = tfds.load(
        'mnist', split=['train', 'test'], shuffle_files=True,
        as_supervised=True, with_info=True)

    train_data = train_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .shuffle(info.splits['train'].num_examples) \
        .batch(128, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)
    test_data = test_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(128) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE)

    return train_data, test_data


class MLP(nn.Module):
    num_layers: int
    hidden_features: int
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_features)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

def get_model(num_layers=10, hid_features=256):
    return MLP(num_layers, hid_features)

def get_optimizer(params, lr=0.0001, momentum=0.9):
    optimizer = optax.sgd(learning_rate=lr, momentum=momentum)
    optimizer.init(params)
    return optimizer

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
        output = state.apply_fn(params, images)
        loss = cross_entropy_loss(output, labels)
        return loss, output
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, labels)
    return state, metrics


@jax.jit
def eval_step(state, images, labels):
    logits = state.apply_fn(state.params, images)
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

def run():
    rng = random.PRNGKey(0)

    traindata, testdata = get_datasets()
    model = get_model()
    key1, rng = random.split(rng)
    dummy = jnp.ones((1, 784))
    params = model.init(key1, dummy)

    optimizer = get_optimizer(params)

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)

    for epoch in range(100):
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
    main()
