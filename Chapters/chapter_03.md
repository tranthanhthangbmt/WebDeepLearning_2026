<!-- tabs:start -->

#### **Tiếng Anh (English)**

# Chapter 3: Introduction to TensorFlow, PyTorch, JAX, and Keras

This chapter covers

* A closer look at all major deep learning frameworks and their relationships
* An overview of how core deep learning concepts translate to code across
  all frameworks

This chapter is meant to give you everything you need to start doing deep
learning in practice. First, you’ll get familiar with three popular deep learning
frameworks that can be used with Keras:

* TensorFlow (<https://tensorflow.org>)
* PyTorch (<https://pytorch.org/>)
* JAX (<https://jax.readthedocs.io/>)

Then, building on top of the first contact you’ve had with Keras in chapter 2,
we’ll review the core components of neural networks and how they translate
to Keras APIs.

By the end of this chapter, you’ll be ready to move on to practical, real-world
applications — which will start with chapter 4.

## A brief history of deep learning frameworks

In the real world, you’re not going to be writing low-level code from scratch like
we did at the end of chapter 2. Instead, you’re going to use a framework.
Besides Keras, the main deep learning frameworks today are JAX, TensorFlow, and PyTorch.
This book will teach you about all four.

If you’re just getting started with deep learning,
it may seem like all these frameworks have been here forever.
In reality, they’re all quite recent, with Keras being the oldest among the four (launched in March 2015).
The ideas behind these frameworks, however, have a long history —
the first paper about automatic differentiation was published in 1964[[1]](#footnote-1)

All these frameworks combine three key features:

* A way to compute gradients for arbitrary differentiable functions (automatic differentiation)
* A way to run tensor computations on CPUs and GPUs (and possibly even on other specialized deep learning hardware)
* A way to distribute computation across multiple devices or multiple computers, such as multiple GPUs on one computer, or
  even multiple GPUs across multiple separate computers

Together, these three simple features unlock all modern deep learning.

It took a long time for the field to develop robust solutions for all three problems
and package those solutions in a reusable form. Since its inception in the 1960s and until the 2000s,
autodifferentiation had no practical applications in machine learning — folks who worked with neural networks
simply wrote their own gradient logic by hand, usually in a language like C++. Meanwhile, GPU programming was all but impossible.

Things started to slowly change in the late 2000s. First,
Python and its ecosystem were slowly rising in popularity in the scientific community, gaining traction
over MATLAB and C++. Second, NVIDIA released CUDA in 2006, unlocking the possibility of building
neural networks that could run on consumer GPUs. The initial focus on CUDA was on physics simulation
rather than machine learning, but that didn’t stop machine learning researchers from starting to implement
CUDA-based neural networks from 2009 onward. They were typically one-off implementations that ran on a single GPU
without any autodifferentiation.

The first framework to enable autodifferentiation and GPU computation to train deep learning models
was Theano, circa 2009. Theano is the conceptual ancestor of all modern deep learning tools.
It started getting good traction in the machine learning research community in 2013–2014,
after the results of the ImageNet 2012 competition ignited the world’s interest in deep learning.
Around the same time, a few other GPU-enabled deep learning libraries started gaining popularity in the computer vision world —
in particular, Torch 7 (Lua-based) and Caffe (C++-based). Keras launched in early 2015 as a higher-level,
easier-to-use deep learning library powered by Theano, and it quickly gained traction with the few thousands of people who were into deep learning
at the time.

Then in late 2015, Google launched TensorFlow, which took many of the key ideas from Theano and added support for large-scale
distributed computation. The release of TensorFlow was a watershed moment that precipitated deep learning in
the mainstream developer zeitgeist. Keras immediately added support for TensorFlow. By mid-2016, over half of all TensorFlow users
were using it through Keras.

In response to TensorFlow, Meta (named Facebook at the time) launched PyTorch about one year later, taking
ideas from Chainer (a niche but innovative framework launched in mid-2015, now long dead) and NumPy-Autograd, a CPU-only autodifferentiation
library for NumPy released by Maclaurin et al. in 2014. Meanwhile, Google released TPUs as an alternative to GPUs,
alongside XLA, a high-performance compiler developed to enable TensorFlow to run on TPUs.

A few years later, at Google, Matthew Johnson — one of the developers who worked on NumPy-Autograd — released JAX as an alternative
way to use autodifferentiation with XLA. JAX quickly gained traction with researchers thanks to its minimalistic API and
high scalability. Today, Keras, TensorFlow, PyTorch, and JAX are the top frameworks in the deep learning world.

Looking back on this chaotic history, we can ask, What’s next? Will a new framework arise tomorrow?
Will we switch to a new programming language or a new hardware platform?

If you ask me, three things today are certain:

* Python has won. Its machine learning and data science ecosystem simply has too much momentum at this point.
  There won’t be a brand new language to replace it — at least not in the next 15 years.
* We’re in a multiframework world — all four frameworks are well established and are unlikely to go anywhere in the next few years.
  It’s a good idea for you to learn a little bit about each one.
  However, it’s highly possible that *new* frameworks will gain popularity in the future,
  in addition to them; Apple’s recently released MLX could be one such example.
  In this context, using Keras is a considerable advantage: you should be able to run your existing Keras models
  on any new up-and-coming framework via a new Keras backend. Keras will keep providing future-proof stability
  to machine learning developers in the future, like it has since 2015 — back when neither TensorFlow nor PyTorch nor JAX existed.
* New chips may certainly arise in the future, alongside NVIDIA’s GPUs and Google’s TPUs.
  For instance, AMD’s GPU line likely has bright days ahead.
  But any new such chip will have to work with the existing frameworks to gain traction.
  New hardware is unlikely to disrupt your workflows.

## How these frameworks relate to each other

Keras, TensorFlow, PyTorch, and JAX don’t all have the same feature set and aren’t interchangeable.
They have some overlap, but to a large extent, they serve different roles for different use cases.
The biggest difference is between Keras and the three others. Keras is a high-level framework,
while the others are lower level. Imagine building a house. Keras is like a prefabricated building kit:
it provides a streamlined interface for setting up and training neural networks.
In contrast, TensorFlow, PyTorch, and JAX are like the raw materials used in construction.

As you saw in the previous chapters, training a neural network revolves
around the following concepts:

* *First, low-level tensor manipulation* — The infrastructure that underlies
  all modern machine learning. This translates to low-level APIs found in TensorFlow,
  PyTorch[[2]](#footnote-2), and JAX:
  + *Tensors*, including special tensors that store the network’s state (*variables*)
  + *Tensor operations* such as addition, `relu`, or `matmul`
  + *Backpropagation*, a way to compute the gradient of mathematical expressions
* *Second, high-level deep learning concepts* — This translates to Keras APIs:
  + *Layers*, which are combined into a *model*
  + A *loss function*, which defines the feedback signal used for learning
  + An *optimizer*, which determines how learning proceeds
  + *Metrics* to evaluate model performance, such as accuracy
  + A *training loop* that performs mini-batch stochastic gradient descent

Further, Keras is unique in that it isn’t a fully standalone framework. It needs a *backend engine* to run, (see figure 3.4),
much like a prefabricated house-building kit needs to source building materials from somewhere.
TensorFlow, PyTorch, and JAX can all be used as Keras backends.
In addition, Keras can run on NumPy, but since NumPy does not provide an API for gradients,
Keras workflows on NumPy are restricted to making predictions from a model — training is impossible.

Now that you have a clearer understanding of how all these frameworks came to be and how they relate
to each other, let’s dive into what it’s like to work with them. We’ll cover them in chronological order:
TensorFlow first, then PyTorch, and finally JAX.

## Introduction to TensorFlow

TensorFlow is a Python-based open source machine learning framework
developed primarily by Google. Its initial release was in November 2015,
followed by a v1 release in February 2017, and a v2 release in October 2019.
TensorFlow is heavily used in production-grade machine learning applications across the industry.

It’s important to keep in mind that TensorFlow is more than a single library.
It’s really a platform, home to a vast ecosystem of components, some
developed by Google, some developed by third parties. For instance, there’s
TFX for industry-strength machine learning workflow management,
TF-Serving for production deployment,
the TF Optimization Toolkit for model quantization and pruning,
and TFLite and MediaPipe for mobile application deployment.

Together, these components cover a very wide range of use cases,
from cutting-edge research to large-scale production applications.

### First steps with TensorFlow

Over the next paragraphs, you’ll get familiar with all the basics of TensorFlow. We’ll cover the following
key concepts:

* Tensors and variables
* Numerical operations in TensorFlow
* Computing gradients with a `GradientTape`
* Making TensorFlow functions fast by using just-in-time compilation

We’ll then conclude the introduction with an end-to-end example: a pure-TensorFlow
implementation of linear regression.

Let’s get those tensors flowing.

#### Tensors and variables in TensorFlow

To do anything in TensorFlow, we’re going to need some tensors. There are a few
different ways you can create them.

##### Constant tensors

Tensors need to be created with some initial value, so common ways to create
tensors are via `tf.ones` (equivalent to `np.ones`) and `tf.zeros` (equivalent
to `np.zeros`). You can also create a tensor from Python or NumPy values using
`tf.constant`.

```python
>>> import tensorflow as tf
>>> # Equivalent to np.ones(shape=(2, 1))
>>> tf.ones(shape=(2, 1))
tf.Tensor([[1.], [1.]], shape=(2, 1), dtype=float32)
>>> # Equivalent to np.zeros(shape=(2, 1))
>>> tf.zeros(shape=(2, 1))
tf.Tensor([[0.], [0.]], shape=(2, 1), dtype=float32)
>>> # Equivalent to np.array([1, 2, 3], dtype="float32")
>>> tf.constant([1, 2, 3], dtype="float32")
tf.Tensor([1., 2., 3.], shape=(3,), dtype=float32)
```

[Listing 3.1](#listing-3-1): All-ones or all-zeros tensors

##### Random tensors

You can also create tensors filled with random values via
one of the methods of the `tf.random` submodule (equivalent to
the `np.random` submodule).

```python
>>> # Tensor of random values drawn from a normal distribution with
>>> # mean 0 and standard deviation 1. Equivalent to
>>> # np.random.normal(size=(3, 1), loc=0., scale=1.).
>>> x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
>>> print(x)
tf.Tensor(
[[-0.14208166]
 [-0.95319825]
 [ 1.1096532 ]], shape=(3, 1), dtype=float32)
>>> # Tensor of random values drawn from a uniform distribution between
>>> # 0 and 1. Equivalent to np.random.uniform(size=(3, 1), low=0.,
>>> # high=1.).
>>> x = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
>>> print(x)
tf.Tensor(
[[0.33779848]
 [0.06692922]
 [0.7749394 ]], shape=(3, 1), dtype=float32)
```

[Listing 3.2](#listing-3-2): Random tensors

##### Tensor assignment and the Variable class

A significant difference between NumPy arrays and TensorFlow tensors is that
TensorFlow tensors aren’t assignable: they’re constant. For instance, in
NumPy, you can do the following.

```python
import numpy as np

x = np.ones(shape=(2, 2))
x[0, 0] = 0.0
```

[Listing 3.3](#listing-3-3): NumPy arrays are assignable

Try to do the same thing in TensorFlow: you will get an error,
`EagerTensor object does not support item assignment`.

```python
x = tf.ones(shape=(2, 2))
# This will fail, as a tensor isn't assignable.
x[0, 0] = 0.0
```

[Listing 3.4](#listing-3-4): TensorFlow tensors are not assignable

To train a model, we’ll need to update its state, which is a set of tensors.
If tensors aren’t assignable, how do we do it, then?
That’s where variables come in. `tf.Variable` is the
class meant to manage modifiable state in TensorFlow.

To create a variable, you need to provide some initial value, such as a random
tensor.

```python
>>> v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
>>> print(v)
array([[-0.75133973],
       [-0.4872893 ],
       [ 1.6626885 ]], dtype=float32)>
```

[Listing 3.5](#listing-3-5): Creating a `tf.Variable`

The state of a variable can be modified via its `assign` method.

```python
>>> v.assign(tf.ones((3, 1)))
array([[1.],
       [1.],
       [1.]], dtype=float32)>
```

[Listing 3.6](#listing-3-6): Assigning a value to a `Variable`

Assignment also works for a subset of the coefficients.

```python
>>> v[0, 0].assign(3.)
array([[3.],
       [1.],
       [1.]], dtype=float32)>
```

[Listing 3.7](#listing-3-7): Assigning a value to a subset of a `Variable`

Similarly, `assign_add` and `assign_sub` are efficient equivalents of
`+=` and `-=`.

```python
>>> v.assign_add(tf.ones((3, 1)))
array([[2.],
       [2.],
       [2.]], dtype=float32)>
```

[Listing 3.8](#listing-3-8): Using `assign_add`

#### Tensor operations: Doing math in TensorFlow

Just like NumPy, TensorFlow offers a large collection of tensor operations
to express mathematical formulas. Here are a few examples.

```python
a = tf.ones((2, 2))
# Takes the square, same as np.square
b = tf.square(a)
# Takes the square root, same as np.sqrt
c = tf.sqrt(a)
# Adds two tensors (element-wise)
d = b + c
# Takes the product of two tensors (see chapter 2), same as np.matmul
e = tf.matmul(a, b)
# Concatenates a and b along axis 0, same as np.concatenate
f = tf.concat((a, b), axis=0)
```

[Listing 3.9](#listing-3-9): A few basic math operations in TensorFlow

Here’s an equivalent of the `Dense` layer we saw in chapter 2:

```python
def dense(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
```

#### Gradients in TensorFlow: A second look at the GradientTape API

So far, TensorFlow seems to look a lot like NumPy. But here’s something NumPy
can’t do: retrieve the gradient of any differentiable expression with respect
to any of its inputs. Just open a `GradientTape` scope, apply some computation
to one or several input tensors, and retrieve the gradient of the result with
respect to the inputs.

```python
input_var = tf.Variable(initial_value=3.0)
with tf.GradientTape() as tape:
    result = tf.square(input_var)
gradient = tape.gradient(result, input_var)
```

[Listing 3.10](#listing-3-10): Using the `GradientTape`

This is most commonly used to retrieve the gradients of the loss of a model
with respect to its weights: `gradients = tape.gradient(loss, weights)`.

In chapter 2, you saw how the `GradientTape` works on either a single
input or a list of inputs and how inputs could be either scalars
or high-dimensional tensors.

So far, you’ve only seen the case where the input tensors in `tape.gradient()`
were TensorFlow variables. It’s actually possible for these inputs
to be any arbitrary tensor. However, only *trainable variables* are being tracked
by default.
With a constant tensor, you’d have to manually mark it as being tracked,
by calling `tape.watch()` on it.

```python
input_const = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(input_const)
    result = tf.square(input_const)
gradient = tape.gradient(result, input_const)
```

[Listing 3.11](#listing-3-11): Using the `GradientTape` with constant tensor inputs

Why? Because it would be too expensive to preemptively store
the information required to compute the gradient of anything with respect
to anything. To avoid wasting resources, the tape needs to know what to watch.
Trainable variables are watched by default because computing the gradient
of a loss with regard to a list of trainable variables is the most common use
case of the gradient tape.

The gradient tape is a powerful utility, even capable of computing
*second-order gradients* — that is, the gradient of a gradient.
For instance, the gradient of the position of an object with
regard to time is the speed of that object, and the second-order gradient
is its acceleration.

If you measure the position of a falling apple along a vertical axis over time,
and find that it verifies `position(time) = 4.9 * time ** 2`,
what is its acceleration? Let’s use two nested gradient tapes to find out.

```python
time = tf.Variable(0.0)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time**2
    speed = inner_tape.gradient(position, time)
# We use the outer tape to compute the gradient of the gradient from
# the inner tape. Naturally, the answer is 4.9 * 2 = 9.8.
acceleration = outer_tape.gradient(speed, time)
```

[Listing 3.12](#listing-3-12): Using nested gradient tapes to compute second-order gradients

#### Making TensorFlow functions fast using compilation

All the TensorFlow code you’ve written so far has been executing “eagerly.”
This means operations are executed one after the other in the Python runtime,
much like any Python code or NumPy code. Eager execution is great for debugging,
but it is typically quite slow. It can often be beneficial
to parallelize some computation, or “fuse” operations — replacing two consecutive operations,
like `matmul` followed by `relu`, with a single, more efficient operation that does the same
thing without materializing the intermediate output.

This can be achieved via *compilation*. The general idea of compilation is to take
certain functions you’ve written in Python, lift them out of Python, automatically rewrite
them into a faster and more efficient “compiled program,” and then call that program from the Python
runtime.

The main benefit of compilation is improved performance. There’s a drawback too: the code you write is no longer
the code that gets executed, which can make the debugging experience painful. Only turn on compilation after you’ve already debugged your code
in the Python runtime.

You can apply compilation to any TensorFlow function by wrapping it in a `tf.function` decorator, like this:

```python
@tf.function
def dense(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
```

When you do this, any call to `dense()` is replaced with a call to a compiled program that implements
a more optimized version of the function. The first call to the function will take a bit longer, because TensorFlow
will be compiling your code. This only happens once — all subsequent calls to the same function will be fast.

TensorFlow has two compilation modes:

* First, the default one, which we refer to as “graph mode.” Any function
  decorated with `@tf.function` runs in graph mode.
* Second, compilation with XLA, a high-performance compiler for ML (it’s short
  for Accelerated Linear Algebra). You can turn it on by specifying
  `jit_compile=True`, like this:

```python
@tf.function(jit_compile=True)
def dense(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
```

It is often the case that compiling a function with XLA will make it run faster than graph mode — though it takes more time to execute the function
the first time, since the compiler has more work to do.

### An end-to-end example: A linear classifier in pure TensorFlow

You know about tensors, variables, and tensor operations, and you know how to
compute gradients. That’s enough to build any TensorFlow-based machine learning model based
on gradient descent. Let’s walk through an end-to-end example to make sure everything is
crystal clear.

In a machine learning job interview, you may be asked to implement a linear
classifier from scratch: a very simple task
that serves as a filter between candidates who have some minimal machine
learning background, and those who don’t. Let’s get you past that filter,
and use your newfound knowledge of TensorFlow to implement such
a linear classifier.

First, let’s come up with some nicely linearly separable
synthetic data to work with: two classes of points in a 2D plane.

```python
import numpy as np

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    # Generates the first class of points: 1,000 random 2D points with
    # specified "mean" and "covariance matrix." Intuitively, the
    # "covariance matrix" describes the shape of the point cloud, and
    # the "mean" describes its position in the plane. `cov=[[1,
    # 0.5],[0.5, 1]]` corresponds to "an oval-like point cloud oriented
    # from bottom left to top right."
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    # Generates the other class of points with a different mean and the
    # same covariance matrix (point cloud with a different position and
    # the same shape)
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)
```

[Listing 3.13](#listing-3-13): Generating two classes of random points in a 2D plane

`negative_samples` and `positive_samples` are both arrays with shape `(1000, 2)`.
Let’s stack them into a single array with shape `(2000, 2)`.

```python
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
```

[Listing 3.14](#listing-3-14): Stacking the two classes into an array with shape `(2000, 2)`

Let’s generate the corresponding target labels, an array of 0s and 1s of
shape `(2000, 1)`, where `targets[i, 0]` is 0 if `inputs[i]` belongs to class 0
(and inversely).

```python
targets = np.vstack(
    (
        np.zeros((num_samples_per_class, 1), dtype="float32"),
        np.ones((num_samples_per_class, 1), dtype="float32"),
    )
)
```

[Listing 3.15](#listing-3-15): Generating the corresponding targets (0 and 1)

Let’s plot our data with Matplotlib, a well-known Python data visualization
library (it comes preinstalled in Colab, so no need for you to install it
yourself), as shown in figure 3.1.

```python
import matplotlib.pyplot as plt

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()
```

[Listing 3.16](#listing-3-16): Plotting the two point classes


![](../images/ch03/linear_model_inputs.282fc3b6.png)


[Figure 3.1](#figure-3-1): Our synthetic data: two classes of random points in the 2D plane

Now, let’s create a linear classifier that can learn to separate these two blobs.
A linear classifier is an affine transformation (`prediction = matmul(input, W) + b`)
trained to minimize the square of the difference between predictions
and the targets.

As you’ll see, it’s actually a much simpler example
than the end-to-end example of a toy two-layer neural network from
the end of chapter 2. However, this time,
you should be able to understand everything about the code, line by line.

Let’s create our variables `W` and `b`, initialized with
random values and with zeros, respectively.

```python
# The inputs will be 2D points.
input_dim = 2
# The output predictions will be a single score per sample (close to 0
# if the sample is predicted to be in class 0, and close to 1 if the
# sample is predicted to be in class 1).
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
```

[Listing 3.17](#listing-3-17): Creating the linear classifier variables

Here’s our forward pass function.

```python
def model(inputs, W, b):
    return tf.matmul(inputs, W) + b
```

[Listing 3.18](#listing-3-18): The forward pass function

Because our linear classifier operates on 2D inputs, `W` is really just two
scalar coefficients: `W = [[w1], [w2]]`.
Meanwhile, `b` is a single scalar coefficient. As such, for given input point
`[x, y]`, its prediction value is
`prediction = [[w1], [w2]] • [x, y] + b = w1 * x + w2 * y + b`.

Here’s our loss function.

```python
def mean_squared_error(targets, predictions):
    # per_sample_losses will be a tensor with the same shape as targets
    # and predictions, containing per-sample loss scores.
    per_sample_losses = tf.square(targets - predictions)
    # We need to average these per-sample loss scores into a single
    # scalar loss value: reduce_mean does this.
    return tf.reduce_mean(per_sample_losses)
```

[Listing 3.19](#listing-3-19): The mean squared error loss function

Now, we move to the training step, which receives some training data and updates the
weights `W` and `b` to minimize the loss on the data.

```python
learning_rate = 0.1

# Wraps the function in a tf.function decorator to speed it up
@tf.function(jit_compile=True)
def training_step(inputs, targets, W, b):
    # Forward pass, inside of a gradient tape scope
    with tf.GradientTape() as tape:
        predictions = model(inputs, W, b)
        loss = mean_squared_error(predictions, targets)
    # Retrieves the gradient of the loss with regard to weights
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    # Updates the weights
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss
```

[Listing 3.20](#listing-3-20): The training-step function

For simplicity, we’ll do *batch training* instead of *mini-batch training*:
we’ll run each training step (gradient computation and weight update) on the
entire data, rather than iterate over the data in small batches. On one hand,
this means that each training step will take much longer to run, since we
compute the forward pass and the gradients for 2,000 samples at once.
On the other hand, each gradient update will be much more effective at reducing
the loss on the training data, since it will encompass information from all
training samples instead of, say, only 128 random samples.
As a result, we will need many fewer steps of training, and we should use
a larger learning rate than what we would typically use for mini-batch training
(we’ll use `learning_rate = 0.1`, as previously defined).

```python
for step in range(40):
    loss = training_step(inputs, targets, W, b)
    print(f"Loss at step {step}: {loss:.4f}")
```

[Listing 3.21](#listing-3-21): The batch training loop

After 40 steps, the training loss seems to have stabilized around 0.025.
Let’s plot how our linear model classifies the training data points, as shown in figure 3.2.
Because our targets are 0s and 1s, a given input point
will be classified as “0” if its prediction value is below 0.5,
and as “1” if it is above 0.5:

```python
predictions = model(inputs, W, b)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
```


![](../images/ch03/linear_model_predictions.3e5424ac.png)


[Figure 3.2](#figure-3-2): Our model’s predictions on the training inputs: pretty similar to the training targets

Recall that the prediction value for a given point `[x, y]` is simply
`prediction == [[w1], [w2]] • [x, y] + b == w1 * x + w2 * y + b`.
Thus, class “0” is defined as
`w1 * x + w2 * y + b < 0.5` and class “1” is defined as
`w1 * x + w2 * y + b > 0.5`. You’ll notice that what you’re looking at is
really the equation of a line in the 2D plane: `w1 * x + w2 * y + b = 0.5`.
Class 1 is above the line; class 0 is below the line.
You may be used to seeing line equations in the format `y = a * x + b`; in the same
format, our line becomes `y = - w1 / w2 * x + (0.5 - b) / w2`.

Let’s plot this line, as shown in figure 3.3:

```python
# Generates 100 regularly spaced numbers between -1 and 4, which we
# will use to plot our line
x = np.linspace(-1, 4, 100)
# This is our line's equation.
y = -W[0] / W[1] * x + (0.5 - b) / W[1]
# Plots our line (`"-r"` means "plot it as a red line")
plt.plot(x, y, "-r")
# Plots our model's predictions on the same plot
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
```


![](../images/ch03/linear_model_with_plotted_line.fd88e7bc.png)


[Figure 3.3](#figure-3-3): Our model, visualized as a line

This is really what a linear classifier is all about: finding the parameters
of a line (or, in higher-dimensional spaces, a hyperplane) neatly separating
two classes of data.

### What makes the TensorFlow approach unique

You’re now familiar with all the basic APIs that underlie TensorFlow-based workflows,
and you’re about to dive into more frameworks — in particular, PyTorch and JAX. What makes
working with TensorFlow different from working with any other framework? When should you use TensorFlow,
and when could you use something else?

If you ask us, here are the main benefits of TensorFlow:

* Thanks to graph mode and XLA compilation, it’s fast. It’s usually significantly faster than PyTorch and NumPy, though JAX is often even faster.
* It is extremely feature complete. Unique among all frameworks, it has support for string tensors as well as “ragged tensors” (tensors where different entries
  may have different dimensions — very useful for handling sequences without requiring to pad them to a shared length). It also has outstanding support for data
  preprocessing, via the highly performant `tf.data` API. `tf.data` is so good that even JAX recommends it for data preprocessing.
  Whatever you need to do, TensorFlow has a solution for it.
* Its ecosystem for production deployment is the most mature among all frameworks, especially when it comes to deploying on mobile or in the browser.

However, TensorFlow also has some noticeable flaws:

* It has a sprawling API — the flipside of being very feature complete. TensorFlow includes thousands of different operations.
* Its numerical API is occasionally inconsistent with the NumPy API, making it a bit harder to approach if you’re already familiar with NumPy.
* The popular pretrained model-sharing platform Hugging Face has less support for TensorFlow, which means that
  the latest generative AI models may not always be available in TensorFlow.

Now, let’s move on to PyTorch.

## Introduction to PyTorch

PyTorch is a Python-based open source machine learning framework developed primarily by Meta (formerly Facebook)
It was originally released in September 2016 (as a response to the release of TensorFlow),
with its 1.0 version launched in 2018, and its 2.0 version launched in 2023.
PyTorch inherits its programming style from the now-defunct Chainer framework, which was itself inspired by NumPy-Autograd.
PyTorch is used extensively in the machine learning research community.

Like TensorFlow, PyTorch is at the center of a large ecosystem of related packages, such as `torchvision`, `torchaudio`,
or the popular model-sharing platform Hugging Face.

The PyTorch API is higher level than that of TensorFlow and JAX: it includes layers and optimizers, like Keras.
These layers and optimizers are compatible with Keras workflows when you use Keras with the PyTorch backend.

### First steps with PyTorch

Over the next paragraphs, you’ll get familiar with all the basics of PyTorch. We’ll cover the following
key concepts:

* Tensors and parameters
* Numerical operations in PyTorch
* Computing gradients with the `backward()` method
* Packaging computation with the `Module` class
* Speeding up PyTorch by using compilation

We’ll conclude the introduction by reimplementing our linear regression end-to-end example in pure PyTorch.

#### Tensors and parameters in PyTorch

A first gotcha about PyTorch is that the package isn’t named `pytorch`. It’s actually named `torch`.
You’d install it via `pip install torch` and you’d import it via `import torch`.

Like in NumPy and TensorFlow, the object at the heart of the framework is the tensor. First, let’s get our hands on some PyTorch tensors.

##### Constant tensors

Here are some constant tensors.

```python
>>> import torch
>>> # Unlike in other frameworks, the shape argument is named "size"
>>> # rather than "shape."
>>> torch.ones(size=(2, 1))
tensor([[1.], [1.]])
>>> torch.zeros(size=(2, 1))
tensor([[0.], [0.]])
>>> # Unlike in other frameworks, you cannot pass dtype="float32" as a
>>> # string. The dtype argument must be a torch dtype instance.
>>> torch.tensor([1, 2, 3], dtype=torch.float32)
tensor([1., 2., 3.])
```

[Listing 3.22](#listing-3-22): All-ones or all-zeros tensors

##### Random tensors

Random tensor creation is similar to NumPy and TensorFlow, but with divergent syntax.
Consider the function `normal`: it doesn’t take a shape argument. Instead,
the mean and standard deviation should be provided as PyTorch tensors with the expected output shape.

```python
>>> # Equivalent to tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
>>> torch.normal(
... mean=torch.zeros(size=(3, 1)),
... std=torch.ones(size=(3, 1)))
tensor([[-0.9613],
        [-2.0169],
        [ 0.2088]])
```

[Listing 3.23](#listing-3-23): Random tensors

As for creating a random uniform tensor, you’d do that via `torch.rand`. Unlike `np.random.uniform` or `tf.random.uniform`,
the output shape should be provided as independent arguments for each dimension, like this:

```python
>>> # Equivalent to tf.random.uniform(shape=(3, 1), minval=0.,
>>> # maxval=1.)
>>> torch.rand(3, 1)
```

##### Tensor assignment and the Parameter class

Like NumPy arrays, but unlike TensorFlow tensors, PyTorch tensors are assignable. You can do operations like this:

```python
>>> x = torch.zeros(size=(2, 1))
>>> x[0, 0] = 1.
>>> x
tensor([[1.],
        [0.]])
```

While you can just use a regular `torch.Tensor` to store the trainable state of a model,
PyTorch does provide a specialized tensor subclass for that purpose, the `torch.nn.parameter.Parameter` class.
Compared to a regular tensor, it provides semantic clarity — if you see a `Parameter`, you’ll know it’s a piece of trainable state, whereas a `Tensor`
could be anything. As a result, it enables PyTorch to automatically track and retrieve the `Parameters` you assign
to PyTorch models — similar to what Keras does with Keras `Variable` instances.

Here’s a `Parameter`.

```python
>>> x = torch.zeros(size=(2, 1))
>>> # A Parameter can only be created using a torch.Tensor value — no
>>> # NumPy arrays allowed.
>>> p = torch.nn.parameter.Parameter(data=x)
```

[Listing 3.24](#listing-3-24): Creating a PyTorch parameter

#### Tensor operations: Doing math in PyTorch

Math in PyTorch works just the same as math in NumPy or TensorFlow, although much like TensorFlow,
the PyTorch API often diverges in subtle ways from the NumPy API.

```python
a = torch.ones((2, 2))
# Takes the square, same as np.square
b = torch.square(a)
# Takes the square root, same as np.sqrt
c = torch.sqrt(a)
# Adds two tensors (element-wise)
d = b + c
# Takes the product of two tensors (see chapter 2), same as np.matmul
e = torch.matmul(a, b)
# Concatenates a and b along axis 0, same as np.concatenate
f = torch.cat((a, b), dim=0)
```

[Listing 3.25](#listing-3-25): A few basic math operations in PyTorch

Here’s a dense layer:

```python
def dense(inputs, W, b):
    return torch.nn.relu(torch.matmul(inputs, W) + b)
```

#### Computing gradients with PyTorch

There’s no explicit “gradient tape” in PyTorch. A similar mechanism does
exist: when you run any computation in PyTorch, the framework creates a one-time
computation graph (a “tape”) that records what just happened.
However, that tape is hidden from the user. The public API for using it
is at the level of tensors themselves: you can call
`tensor.backward()` to run backpropagation through all operations previously executed
that led to that tensor. Doing this will populate the `.grad` attribute of
all tensors that are tracking gradients.

```python
>>> # To compute gradients with respect to a tensor, it must be created
>>> # with requires_grad=True.
>>> input_var = torch.tensor(3.0, requires_grad=True)
>>> result = torch.square(input_var)
>>> # Calling backward() populates the "grad" attribute on all tensors
>>> # create with requires_grad=True.
>>> result.backward()
>>> gradient = input_var.grad
>>> gradient
tensor(6.)
```

[Listing 3.26](#listing-3-26): Computing a gradient with `.backward()`

If you call `backward()` multiple times in a row, the `.grad` attribute will “accumulate” gradients: each
new call will sum the new gradient with the preexisting one. For instance, in the following code,
`input_var.grad` is not the gradient of `square(input_var)` with respect to `input_var`; rather, it is the sum
of that gradient and the previously computed gradient — its value has doubled since our last code snippet:

```python
>>> result = torch.square(input_var)
>>> result.backward()
>>> # .grad will sum all gradient values from each time backward() is
>>> # called.
>>> input_var.grad
tensor(12.)
```

To reset gradients, you can just set `.grad` to `None`:

```python
>>> input_var.grad = None
```

Now let’s put this into practice!

### An end-to-end example: A linear classifier in pure PyTorch

You now know enough to rewrite our linear classifier in PyTorch. It will stay very similar to the TensorFlow one — the only major difference is how we compute the gradients.

Let’s start by creating our model variables. Don’t forget to pass `requires_grad=True` so we can compute gradients with respect to them:

```python
input_dim = 2
output_dim = 1

W = torch.rand(input_dim, output_dim, requires_grad=True)
b = torch.zeros(output_dim, requires_grad=True)
```

This is our model — no difference so far. We just went from `tf.matmul` to `torch.matmul`:

```python
def model(inputs, W, b):
    return torch.matmul(inputs, W) + b
```

This is our loss function. We just switch from `tf.square` to `torch.square` and from `tf.reduce_mean` to `torch.mean`:

```python
def mean_squared_error(targets, predictions):
    per_sample_losses = torch.square(targets - predictions)
    return torch.mean(per_sample_losses)
```

Now for the training step. Here’s how it works:

1. `loss.backward()` runs backpropagation starting from the `loss` output node and populates
   the `tensor.grad` attribute on all tensors that were involved in the computation of `loss`.
   `tensor.grad` represents the gradient of the loss with regard to that tensor.
2. We use the `.grad` attribute to recover the gradients of the loss with regard to `W` and `b`.
3. We update `W` and `b` using those gradients. Because these updates are not intended to be
   part of the backward pass, we do them inside a `torch.no_grad()` scope, which skips gradient
   computation for everything inside it.
4. We reset the contents of the `.grad` property of our `W` and `b` parameters, by setting it to `None`.
   If we didn’t do this, gradient values would accumulate across multiple calls to `training_step()`,
   resulting in invalid values:

```python
learning_rate = 0.1

def training_step(inputs, targets, W, b):
    # Forward pass
    predictions = model(inputs)
    loss = mean_squared_error(targets, predictions)
    # Computes gradients
    loss.backward()
    # Retrieves gradients
    grad_loss_wrt_W, grad_loss_wrt_b = W.grad, b.grad
    with torch.no_grad():
        # Updates weights inside a no_grad scope
        W -= grad_loss_wrt_W * learning_rate
        b -= grad_loss_wrt_b * learning_rate
    # Resets gradients
    W.grad = None
    b.grad = None
    return loss
```

This could be made even simpler — let’s see how.

#### Packaging state and computation with the Module class

PyTorch also has a higher-level, object-oriented API for performing backpropagation, which requires
relying on two new classes: the `torch.nn.Module` class and an optimizer class from
the `torch.optim` module, such as `torch.optim.SGD` (the equivalent of `keras.optimizers.SGD`).

The general idea is to define a subclass of `torch.nn.Module`, which will

* Hold some `Parameters`, to store state variables. Those are defined in the `__init__()` method.
* Implement the forward pass computation in the `forward()` method.

It should look just like the following.

```python
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.rand(input_dim, output_dim))
        self.b = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        return torch.matmul(inputs, self.W) + self.b
```

[Listing 3.27](#listing-3-27): Defining a `torch.nn.Module`

We can now instantiate our `LinearModel`:

```python
model = LinearModel()
```

When using an instance of `torch.nn.Module`, rather than calling the `forward()`
method directly, you’d use `__call__()` (i.e., directly call the model class on
inputs), which redirects to `forward()` but adds a few framework hooks to it:

```python
torch_inputs = torch.tensor(inputs)
output = model(torch_inputs)
```

Now, let’s get our hands on a PyTorch optimizer. To instantiate it, you will
need to provide the list of parameters that the optimizer is intended to update.
You can retrieve it from our `Module` instance via `.parameters()`:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Using our `Module` instance and the PyTorch `SGD` optimizer, we can run a simplified training step:

```python
def training_step(inputs, targets):
    predictions = model(inputs)
    loss = mean_squared_error(targets, predictions)
    loss.backward()
    optimizer.step()
    model.zero_grad()
    return loss
```

Previously, updating the model parameters looked like this:

```python
with torch.no_grad():
    W -= grad_loss_wrt_W * learning_rate
    b -= grad_loss_wrt_b * learning_rate
```

Now we can just do `optimizer.step()`.

Similarly, previously we needed to reset parameter gradients by hand by doing `tensor.grad = None` on each one.
Now we can just do `model.zero_grad()`.

Overall, this may feel a bit confusing — somehow the loss tensor, the optimizer, and the `Module` instance
all seem to be aware of each other through some hidden background mechanism.
They’re all interacting with one another via spooky action at a distance. Don’t worry though — you
can just treat this sequence of steps (`loss.backward()` - `optimizer.step()` - `model.zero_grad()`)
as a magic incantation to be recited any time you need to write a training step function. Just make sure not to forget
`model.zero_grad()`. That would be a major bug (and it is unfortunately quite common)!

#### Making PyTorch modules fast using compilation

One last thing. Similarly to how TensorFlow lets you compile functions for better performance, PyTorch lets
you compile functions or even `Module` instances via the `torch.compile()` utility.
This API uses PyTorch’s very own compiler, named Dynamo.

Let’s try it on our linear regression `Module`:

```python
compiled_model = torch.compile(model)
```

The resulting object is intended to work identically to the original — except the forward and backward pass should run faster.

You can also use `torch.compile()` as a function decorator:

```python
@torch.compile
def dense(inputs, W, b):
    return torch.nn.relu(torch.matmul(inputs, W) + b)
```

In practice, most PyTorch code out there does not use compilation and simply runs eagerly,
as the compiler may not always work with all models and may not always result in a speedup when it does work. Unlike in TensorFlow and
Jax where compilation was built in from the inception of the library, PyTorch’s compiler is a relatively recent addition.

### What makes the PyTorch approach unique

Compared to TensorFlow and JAX, which we will cover next, what makes PyTorch
stand out? Why should you use it or not use it?

Here are PyTorch’s two key strengths:

* PyTorch code executes eagerly by default, making it easy to debug.
  Note that this is also the case for TensorFlow code and JAX code, but a big difference is that PyTorch is generally intended to be
  run eagerly at all times, whereas any serious TensorFlow or JAX project will inevitably need compilation at some point, which can significantly hurt the debugging experience.
* The popular pretrained model-sharing platform Hugging Face has first-class support for PyTorch, which
  means that any model you’d like to use is likely available in PyTorch.
  This is the primary drive behind PyTorch adoption today.

Meanwhile, there are also some downsides to using PyTorch:

* Like with TensorFlow, the PyTorch API is inconsistent with NumPy. Further, it’s also internally inconsistent. For instance, the commonly used keyword `axis` is occasionally named `dim` instead, depending on the function.
  Some pseudo-random number generation operations take a `seed` argument; others don’t. And so on.
  This can make PyTorch frustrating to learn, especially when coming from NumPy.
* Due to its focus on eager execution, PyTorch is quite slow — it’s the slowest
  of all the major frameworks by a large margin. For most models, you may see a 20% or 30% speedup with JAX.
  For some models — especially large ones — you may even see a 3× or a 5× speedup with JAX, even after using `torch.compile()`.
* While it is possible to make PyTorch code faster via `torch.compile()`, the PyTorch Dynamo compiler
  remains at this time (in 2025) quite ineffective and full of trapdoors. As a result, only a very small percentage of the
  PyTorch user base uses compilation. Perhaps this will be improved in future versions!

## Introduction to JAX

JAX is an open source library for differentiable computation, primarily developed by Google.
After its release in 2018, JAX quickly gained traction in the research community, particularly for its ability to use Google’s TPUs at scale.
Today, JAX is in use by most of the top players in the generative AI space — companies like DeepMind, Apple, Midjourney, Anthropic, Cohere, and so on.

JAX embraces a *stateless* approach to computation, meaning that functions in JAX do not maintain any persistent state. This contrasts with traditional imperative programming, where variables can hold values between function calls.

The stateless nature of JAX functions has several advantages. In particular, it enables effective automatic parallelization and distributed computation, as functions can be executed independently without the need for synchronization. The extreme scalability of JAX is essential for handling the very large-scale machine learning problems faced by companies like Google and DeepMind.

### First steps with JAX

We’ll go over the following key concepts:

* The `array` class
* Random operations in JAX
* Numerical operations in JAX
* Computing gradients via `jax.grad` and `jax.value_and_grad`
* Making JAX functions fast by leveraging just-in-time compilation

Let’s get started.

### Tensors in JAX

One of the best features of JAX is that it doesn’t try to implement its own independent, similar-to-NumPy-but-slightly-divergent
numerical API. Instead, it just implements the NumPy API, as is. It is available as the `jax.numpy` namespace, and you
will often see it imported as `jnp` for short.

Here are some JAX arrays.

```python
>>> from jax import numpy as jnp
>>> jnp.ones(shape=(2, 1))
Array([[1.],
       [1.]], dtype=float32)
>>> jnp.zeros(shape=(2, 1))
Array([[0.],
       [0.]], dtype=float32)
>>> jnp.array([1, 2, 3], dtype="float32")
Array([1., 2., 3.], dtype=float32)
```

[Listing 3.28](#listing-3-28): All-ones or all-zeros tensors

There are, however, two minor differences between `jax.numpy` and the actual NumPy API: random number generation and array assignment. Let’s take a look.

### Random number generation in JAX

The first difference between JAX and NumPy has to do with the way JAX handles random operations — what is known as “PRNG” (Pseudo-Random Number Generation) operations.
We said earlier that JAX is *stateless*, which implies that JAX code can’t rely on any hidden global state. Consider the following NumPy code.

```python
>>> np.random.normal(size=(3,))
array([-1.68856166,  0.16489586,  0.67707523])
>>> np.random.normal(size=(3,))
array([-0.73671259,  0.3053194 ,  0.84124895])
```

[Listing 3.29](#listing-3-29): Random tensors

How did the second call to `np.random.normal()` know to return a different value from the first call? That’s right — it’s a hidden piece of global state.
You can actually retrieve that global state via `np.random.get_state()` and set it via `np.random.seed(seed)`.

In a stateless framework, we can’t have any such global state. The same API call must always return the same value. As a result, in a stateless version of NumPy, you would have to rely on passing different seed arguments to your `np.random` calls to get different values.

Now, it’s often the case that your PRNG calls are going to be in functions that get called multiple times and that are intended to use different random values each time. If you don’t want to rely on any global state, this requires you to manage your seed state outside of the target function, like this:

```python
def apply_noise(x, seed):
    np.random.seed(seed)
    x = x * np.random.normal((3,))
    return x

seed = 1337
y = apply_noise(x, seed)
seed += 1
z = apply_noise(x, seed)
```

It’s basically the same in JAX. However, JAX doesn’t use integer seeds. It uses
special array structures called *keys*. You can create one from an integer value, like this:

```python
import jax

seed_key = jax.random.key(1337)
```

To force you to always provide a seed “key” to PRNG calls, all JAX PRNG-using operations take `key` (the random seed) as their first positional argument. Here’s how to use `random.normal()`:

```python
>>> seed_key = jax.random.key(0)
>>> jax.random.normal(seed_key, shape=(3,))
Array([ 1.8160863 , -0.48262316,  0.33988908], dtype=float32)
```

Two calls to `random.normal()` that receive the same seed key will always return the same value.

```python
>>> seed_key = jax.random.key(123)
>>> jax.random.normal(seed_key, shape=(3,))
Array([-0.1470326,  0.5524756,  1.648498 ], dtype=float32)
>>> jax.random.normal(seed_key, shape=(3,))
Array([-0.1470326,  0.5524756,  1.648498 ], dtype=float32)
```

[Listing 3.30](#listing-3-30): Using a random seed in Jax

If you need a new seed key, you can simply create a new one from an existing one using the `jax.random.split()` function. It is deterministic, so the same sequence of splits will always result in the same final seed key:

```python
>>> seed_key = jax.random.key(123)
>>> jax.random.normal(seed_key, shape=(3,))
Array([-0.1470326,  0.5524756,  1.648498 ], dtype=float32)
>>> # You could even split your key into multiple new keys at once!
>>> new_seed_key = jax.random.split(seed_key, num=1)[0]
>>> jax.random.normal(new_seed_key, shape=(3,))
Array([ 0.5362355, -1.1920372,  2.450225 ], dtype=float32)
```

This is definitely more work than `np.random`! But the benefits of statelessness far outweigh the costs: it makes your code *vectorizable* (i.e., the JAX compiler can automatically turn it into highly parallel code) while maintaining determinism (i.e., you can run the same code twice with the same results). That is impossible to achieve with a global PRNG state.

#### Tensor assignment

The second difference between JAX and NumPy is tensor assignment.
Like in TensorFlow, JAX arrays are not assignable in place. That’s because any sort of in-place modification would go against JAX’s stateless design.
Instead, if you need to update a tensor, you must create a new tensor with the desired value. JAX makes this easy by providing
the `at()`/`set()` API. These methods allow you to create a new tensor with an updated element at a specific index. Here’s an example of how you would update the first element of a JAX array to a new value.

```python
>>> x = jnp.array([1, 2, 3], dtype="float32")
>>> new_x = x.at[0].set(10)
```

[Listing 3.31](#listing-3-31): Modifying values in a JAX array

Simple enough!

#### Tensor operations: Doing math in JAX

Doing math in JAX looks exactly the same as it does in NumPy. No need to learn anything new this time!

```python
a = jnp.ones((2, 2))
# Takes the square
b = jnp.square(a)
# Takes the square root
c = jnp.sqrt(a)
# Adds two tensors (element-wise)
d = b + c
# Takes the product of two tensors (see chapter 2)
e = jnp.matmul(a, b)
# Multiplies two tensors (element-wise)
e *= d
```

[Listing 3.32](#listing-3-32): A few basic math operations in JAX

Here’s a dense layer:

```python
def dense(inputs, W, b):
    return jax.nn.relu(jnp.matmul(inputs, W) + b)
```

#### Computing gradients with JAX

Unlike TensorFlow and PyTorch, JAX takes a *metaprogramming* approach
to gradient computation. Metaprogramming refers to the idea of having *functions that return functions*
— you could call them “meta-functions.” In practice, JAX lets you *turn a loss-computation function into a gradient-computation function*.
So computing gradients in JAX is a three-step process:

1. Define a loss function, `compute_loss()`.
2. Call `grad_fn = jax.grad(compute_loss)` to retrieve a gradient-computation function.
3. Call `grad_fn` to retrieve the gradient values.

The loss function should verify the following properties:

* It should return a scalar loss value.
* Its first argument (which, in the following example, is also the only argument) should contain the state arrays we need gradients for.
  This argument is usually named `state`. For instance, this first argument could be a single array, a list of arrays, or a dict of arrays.

Let’s take a look at a simple example. Here’s a loss-computation function that takes a single scalar, `input_var` and returns a scalar loss
value — just the square of the input:

```python
def compute_loss(input_var):
    return jnp.square(input_var)
```

We can now call the JAX utility `jax.grad()` on this loss function.
It returns a gradient-computation function — a function that takes the
same arguments as the original loss function and returns the gradient of the loss with respect to `input_var`:

```python
grad_fn = jax.grad(compute_loss)
```

Once you’ve obtained `grad_fn()`, you can call it with the same arguments as `compute_loss()`, and it will return gradients arrays
corresponding to the first argument of `compute_loss()`. In our case, our first argument was a single array, so `grad_fn()` directly
returns the gradient of the loss with respect to that one array:

```python
input_var = jnp.array(3.0)
grad_of_loss_wrt_input_var = grad_fn(input_var)
```

#### JAX gradient-computation best practices

So far so good! Metaprogramming is a big word, but it turns out to be quite simple.
Now, in real-world use cases, there are a few more things you’ll need to take into account. Let’s take a look.

##### Returning the loss value

It’s usually the case that you don’t just need the gradient array; you also need the loss
value. It would be quite inefficient to recompute it independently outside of `grad_fn()`, so instead, you
can just configure your `grad_fn()` to also return the loss value. This is done by using the JAX utility
`jax.value_and_grad()` instead of `jax.grad()`. It works identically, but it returns a tuple of values,
where the first entry is the loss value, and the second entry is the gradient(s):

```python
grad_fn = jax.value_and_grad(compute_loss)
output, grad_of_loss_wrt_input_var = grad_fn(input_var)
```

##### Getting gradients for a complex function

Now, what if you need gradients for more than a single variable?
And what if your `compute_loss()` function has more than one input?

Let’s say your state contains three variables, `a`, `b`, and `c`, and your loss function has two inputs, `x` and `y`.
You would simply structure it like this:

```python
# state contains a, b, and c. It must be the first argument.
def compute_loss(state, x, y):
    ...
    return loss

grad_fn = jax.value_and_grad(compute_loss)
state = (a, b, c)
# grads_of_loss_wrt_state has the same structure as state.
loss, grads_of_loss_wrt_state = grad_fn(state, x, y)
```

Note that `state` doesn’t have to be a tuple — it could be a dict, a list, or any nested structure of tuples, dicts, and lists. In
JAX parlance, such a nested structure is called a *tree*.

##### Returning auxiliary outputs

Finally, what if your `compute_loss()` function needs to return more than just the loss?
Let’s say you want to return an additional value `output` that’s computed as a by-product of the loss computation.
How to get it out?

You would use the `has_aux` argument:

1. Edit the loss function to return a tuple where the first entry is the loss, and the second entry is your extra output.
2. Pass the argument `has_aux=True` to `value_and_grad()`. This tells `value_and_grad()` to return not just the gradient
   but also the “auxiliary” output(s) of `compute_loss()`, like this:

```python
def compute_loss(state, x, y):
    ...
    # Returns a tuple
    return loss, output

# Passes has_aux=True here
grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
# Gets back a nested tuple
loss, (grads_of_loss_wrt_state, output) = grad_fn(state, x, y)
```

Admittedly, things are starting to be pretty convoluted at this point.
Don’t worry, though; this is about as hard as JAX gets! Almost everything else is simpler by comparison.

#### Making JAX functions fast with @jax.jit

One more thing. As a JAX user, you will frequently use the `@jax.jit` decorator, which behaves
identically to the `@tf.function(jit_compile=True)` decorator. It turns any
stateless JAX function into an XLA-compiled piece of code, typically delivering a considerable execution speedup:

```python
@jax.jit
def dense(inputs, W, b):
    return jax.nn.relu(jnp.matmul(inputs, W) + b)
```

Be mindful that you can only decorate a stateless function — any tensors that get updated by the
function should be part of its return values.

### An end-to-end example: A linear classifier in pure JAX

Now you know enough JAX to write the JAX version of our linear classifier example. There are two major differences
from the TensorFlow and PyTorch versions you’ve already seen:

* All functions we will create will be *stateless*. That means the state (the arrays `W` and `b`) will be provided
  as function arguments, and if they get modified by the function, their new value will be returned by the function.
* Gradients are computed using the JAX `value_and_grad()` utility.

Let’s get started.
The model function and the mean squared error function should look familiar:

```python
def model(inputs, W, b):
    return jnp.matmul(inputs, W) + b

def mean_squared_error(targets, predictions):
    per_sample_losses = jnp.square(targets - predictions)
    return jnp.mean(per_sample_losses)
```

To compute gradients, we need to package loss computation in
a single `compute_loss()` function. It returns the total loss as a scalar,
and it takes `state` as its first argument — a tuple of all
the tensors we need gradients for:

```python
def compute_loss(state, inputs, targets):
    W, b = state
    predictions = model(inputs, W, b)
    loss = mean_squared_error(targets, predictions)
    return loss
```

Calling `jax.value_and_grad()` on this function gives us a new
function, with the same argument as `compute_loss`, which returns
both the loss and the gradients of the loss with regard to the elements
of `state`:

```python
grad_fn = jax.value_and_grad(compute_loss)
```

Next, we can set up our training step function. It looks straightforward.
Be mindful that, unlike its TensorFlow and PyTorch equivalents, it needs
to be stateless, and so it must return the updated values of the `W` and `b`
tensors:

```python
learning_rate = 0.1

# We use the jax.jit decorator to take advantage of XLA compilation.
@jax.jit
def training_step(inputs, targets, W, b):
    # Computes the forward pass and backward pass in one go
    loss, grads = grad_fn((W, b), inputs, targets)
    grad_wrt_W, grad_wrt_b = grads
    # Updates W and b
    W = W - grad_wrt_W * learning_rate
    b = b - grad_wrt_b * learning_rate
    # Make sure to return the new values of W and b in addition to the
    # loss!
    return loss, W, b
```

Because we won’t change the `learning_rate` during our example, we can
consider it part of the function itself and not our model’s state. If we
wanted to modify our learning rate during training, we’d need to pass it through
as well.

Finally, we’re ready to run the full training loop. We initialize `W` and `b`,
and we repeatedly update them via stateless calls to `training_step()`:

```python
input_dim = 2
output_dim = 1

W = jax.numpy.array(np.random.uniform(size=(input_dim, output_dim)))
b = jax.numpy.array(np.zeros(shape=(output_dim,)))
state = (W, b)
for step in range(40):
    loss, W, b = training_step(inputs, targets, W, b)
    print(f"Loss at step {step}: {loss:.4f}")
```

That’s it! You’re now able to write a custom training loop in JAX.

### What makes the JAX approach unique

The main thing that makes JAX unique among modern machine learning frameworks is its functional, stateless philosophy. While it may seem to cause friction at first,
it is what unlocks the power of JAX — its ability to compile to extremely fast code and to scale to arbitrarily large models and arbitrarily many devices.

There’s a lot to like about JAX:

* It’s fast. For most models, it is the fastest of all frameworks you’ve seen so far.
* Its numerical API is fully consistent with NumPy, making it pleasant to learn.
* It’s the best fit for training models on TPUs, as it was developed from the ground up for XLA and TPUs.

Using JAX can also come with some amount of developer friction:

* Its use of metaprogramming and compilation can make it significantly harder to debug compared to pure eager execution.
* Low-level training loops tend to be more verbose and more difficult to write than in TensorFlow or PyTorch.

At this point, you know the basics of TensorFlow, PyTorch, and JAX, and you can use these frameworks
to implement a basic linear classifier from scratch. That’s a solid foundation to
build upon. It’s now time to move on to a more productive path to deep learning: the Keras API.

## Introduction to Keras

Keras is a deep learning API for Python that provides a convenient way to define and train
any kind of deep learning model. It was released in March 2015, with its v2 in 2017 and its v3 in 2023.

Keras users range from academic researchers, engineers, and data scientists
at both startups and large companies to graduate students and hobbyists.
Keras is used at Google, Netflix, Uber, YouTube, CERN, NASA, Yelp, Instacart,
Square, Waymo, YouTube, and thousands of smaller organizations
working on a wide range of problems across every industry.
Your YouTube recommendations originate from Keras models.
The Waymo self-driving cars rely on Keras models for processing sensor data.
Keras is also a popular framework on Kaggle, the machine learning competition website.

Because Keras has a diverse user base, it doesn’t force you to follow
a single “true” way of building and training models. Rather, it enables
a wide range of different workflows,
from the very high-level to the very low-level,
corresponding to different user profiles. For instance, you have an array
of ways to build models and an array of ways to train them,
each representing a certain tradeoff between usability and flexibility.
In chapter 7, we’ll review
in detail a good fraction of this spectrum of workflows.

### First steps with Keras

Before we get to writing Keras code, there are a few things to consider when
setting up the library before it’s imported.

#### Picking a backend framework

Keras can be used together with JAX, TensorFlow, or PyTorch. They’re the
“backend frameworks” of Keras. Through these backend frameworks,
Keras can run on top of different types of hardware
(see figure 3.4) — GPU, TPU, or plain CPU —
can be seamlessly scaled to thousands of machines, and can be deployed to a variety of platforms.

![](../images/ch03/keras_and_backends.7fcf768f.png)


[Figure 3.4](#figure-3-4): Keras and its backends. A backend is a low-level tensor-computing platform; Keras is a high-level deep learning API.

Backend frameworks are pluggable: you can switch to a different backend framework
*after* you’ve written some Keras code. You aren’t locked into a single framework and a single
ecosystem — you can move your models from JAX to TensorFlow to PyTorch depending on your current needs.
For instance, when you develop a Keras model, you could debug it with PyTorch,
train it on TPU with JAX for maximum efficiency, and finally run inference
with the excellent tooling from the TensorFlow ecosystem.

The default backend for Keras right now is TensorFlow, so if you run `import keras` in a fresh
environment, without having configured anything, you will be running on top of TensorFlow.
There are two ways to pick a different backend:

* Set the environment variable `KERAS_BACKEND`. Before you start your `python` repl, you can
  run the following shell command to use JAX as your Keras backend: `export KERAS_BACKEND=jax`.
  Alternatively, you can add the following code snippet at the top of your Python file or notebook
  (note that it must imperatively go before the first `import keras`):

```python
import os

# Sets the environment variable from within the Python runtime
os.environ["KERAS_BACKEND"] = "jax"

# Only then should you import Keras.
import keras
```

* Edit your local Keras configuration file at `~/.keras/keras.json`. If you have already imported
  Keras once, this file has already been created with default settings.
  You can use any text editor to open and modify it — it’s a human-readable JSON file. It should look like this:

```python
{
    # Default floating-point precision. It should typically not be
    # changed.
    "floatx": "float32",
    # Default numerical fuzzing factor. It should typically not be
    # changed.
    "epsilon": 1e-07,
    # Change "tensorflow" to "jax" or "torch."
    "backend": "tensorflow",
    # This is the default image layout. We'll talk about this in
    # chapter 8.
    "image_data_format": "channels_last",
}
```



When configuring the Keras backend, you should use the string `"torch"` to refer
to the PyTorch backend, rather than the string `"pytorch"`, which would be invalid.
This is because the PyTorch package name is `torch` (as in `import torch` or `pip install torch`).

Now, you may ask, which backend should I be picking? It’s really your own choice:
all Keras code examples in the rest of the book will be compatible with all three backends.
If the need for backend-specific code arises (as in chapter 7, for instance),
I will show you all three versions — TensorFlow, PyTorch, JAX.
If you have no particular backend preference,
my personal recommendation is JAX. It’s usually the most performant backend.

Once your backend is configured, you can start actually building and training Keras models. Let’s take a look.

### Layers: The building blocks of deep learning

The fundamental data structure in
neural networks is the *layer*, to which you were introduced in chapter 2. A
layer is a data processing module that takes as input one or more tensors and
that outputs one or more tensors. Some layers are stateless, but more
frequently layers have a state: the layer’s *weights*, one or
several tensors learned with stochastic gradient descent, which together contain
the network’s *knowledge*.

Different types of layers are appropriate for different tensor formats
and different types of data processing.
For instance, simple vector data, stored in 2D
tensors of shape `(samples, features)`, is often processed by
*densely connected* layers, also called *fully connected*
or *dense* layers (the `Dense` class in Keras). Sequence data, stored in 3D
tensors of shape `(samples, timesteps, features)`, is typically processed by
*recurrent* layers, such as an `LSTM` layer, or 1D convolution layers (`Conv1D`).
Image data, stored in rank-4 tensors, is usually processed by 2D convolution
layers (`Conv2D`).

You can think of layers as the LEGO bricks of deep learning, a metaphor that is
made explicit by Keras. Building deep learning models in Keras
is done by clipping together compatible layers to form useful
data transformation pipelines.

#### The base `Layer` class in Keras

A simple API should have a single abstraction around which everything is centered.
In Keras, that’s the `Layer` class. Everything in Keras is either a `Layer` or
something that closely interacts with a `Layer`.

A `Layer` is an object that encapsulates some state (weights) and some computation
(a forward pass). The weights are typically defined in a `build()` (although they
could also be created in the constructor `__init__()`), and the computation is
defined in the `call()` method.

In the previous chapter, we implemented a `NaiveDense` class that contained
two weights `W` and `b` and applied the computation
`output = activation(matmul(input, W) + b)`. The following is what the same layer would
look like in Keras.

```python
import keras

# All Keras layers inherit from the base Layer class.
class SimpleDense(keras.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    # Weight creation takes place in the build() method.
    def build(self, input_shape):
        batch_dim, input_dim = input_shape
        # add_weight is a shortcut method for creating weights. It's
        # also possible to create standalone variables and assign them
        # as layer attributes, like self.W = keras.Variable(shape=...,
        # initializer=...).
        self.W = self.add_weight(
            shape=(input_dim, self.units), initializer="random_normal"
        )
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")

    # We define the forward pass computation in the call() method.
    def call(self, inputs):
        y = keras.ops.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y
```

[Listing 3.33](#listing-3-33): A simple dense layer from scratch in Keras

In the next section, we’ll cover in detail the purpose of these `build()` and
`call()` methods. Don’t worry if you don’t understand everything just yet!

Once instantiated, a layer like this can be used just like a function, taking as
input a tensor:

```python
>>> # Instantiates our layer, defined previously
>>> my_dense = SimpleDense(units=32, activation=keras.ops.relu)
>>> # Creates some test inputs
>>> input_tensor = keras.ops.ones(shape=(2, 784))
>>> # Calls the layer on the inputs, just like a function
>>> output_tensor = my_dense(input_tensor)
>>> print(output_tensor.shape)
(2, 32)
```

Now, you’re probably wondering, why did we have to implement `call()`
and `build()`, since we ended up using our layer by plainly calling it, that is
to say, by using its `__call__` method? It’s because we want to be able to
create the state just in time. Let’s see how that works.

#### Automatic shape inference: Building layers on the fly

Just like with LEGO bricks, you can only “clip” together layers
that are *compatible*. The notion of *layer compatibility* here
refers specifically to the fact that every layer will
only accept input tensors of a certain shape and will return output tensors of
a certain shape. Consider the following example:

```python
from keras import layers

# A dense layer with 32 output units
layer = layers.Dense(32, activation="relu")
```

This layer will return a tensor whose non-batch dimension is 32. It can only be
connected to a downstream layer that expects 32-dimensional vectors as its
input.

When using Keras, you don’t have to worry about size compatibility
most of the time because the layers you add to your models are dynamically
built to match the shape of the incoming inputs. For instance, suppose you
write the following:

```python
from keras import models
from keras import layers

model = models.Sequential(
    [
        layers.Dense(32, activation="relu"),
        layers.Dense(32),
    ]
)
```

The layers didn’t receive any information about the shape of their inputs.
Instead, they automatically inferred their input shape as being the
shape of the first inputs they see.

In the toy version of a `Dense` layer that we’ve implemented in chapter 2,
we had to pass the layer’s input size explicitly
to the constructor in order to be able to create its weights.
That’s not ideal, because it would lead to
models that look like this, where each new layer needs to be made aware
of the shape of the layer before it:

```python
model = NaiveSequential(
    [
        NaiveDense(input_size=784, output_size=32, activation="relu"),
        NaiveDense(input_size=32, output_size=64, activation="relu"),
        NaiveDense(input_size=64, output_size=32, activation="relu"),
        NaiveDense(input_size=32, output_size=10, activation="softmax"),
    ]
)
```

It would be even worse when the rules used by a layer to produce its output
shape are complex. For instance, what if our layer returned outputs of shape
`(batch, input_size * 2 if input_size % 2 == 0 else input_size * 3)`?

If we were to reimplement our `NaiveDense` layer as a Keras layer capable of
automatic shape inference, it would look like the `SimpleDense` layer,
with its `build()` and `call()` methods.

In the Keras `SimpleDense`, we no longer create weights
in the constructor like in the previous example. Instead,
we create them in a dedicated state-creation method `build()`,
which receives as argument the first input shape seen by the layer.
The `build()` method is called automatically the first time the layer is called
(via its `__call__()` method). In fact, that’s why we defined the computation
in a separate `call()` method rather than in the `__call__()` method directly!
The `__call__()` method of the base layer schematically looks like this:

```python
def __call__(self, inputs):
    if not self.built:
        self.build(inputs.shape)
        self.built = True
    return self.call(inputs)
```

With automatic shape inference, our previous example becomes simple and neat:

```python
model = keras.Sequential(
    [
        SimpleDense(32, activation="relu"),
        SimpleDense(64, activation="relu"),
        SimpleDense(32, activation="relu"),
        SimpleDense(10, activation="softmax"),
    ]
)
```

Note that automatic shape inference is not the only thing that the `Layer`
class’s `__call__()` method handles. It takes care of many more things,
in particular routing between *eager* and *graph* execution,
and input masking (which we cover in chapter 14).
For now, just remember: when implementing your own layers,
put the forward pass in the `call()` method.

### From layers to models

A deep learning model is a graph of layers.
In Keras, that’s the `Model` class.
For now, you’ve only seen `Sequential` models (a subclass of `Model`),
which are simple stacks of layers, mapping a single input to a single output.
But as you move forward, you’ll be exposed to a much broader variety of network
topologies. Some common ones are

* Two-branch networks
* Multihead networks
* Residual connections

Network topology can get quite involved. For instance, figure 3.5 shows topology
of the graph of layers of a Transformer, a common architecture designed to
process text data.

![](../images/ch03/transformer.cb3f137f.png)


[Figure 3.5](#figure-3-5): The Transformer architecture. There’s a lot going on here. Throughout the next few chapters, you’ll climb your way up to understanding it (in chapter 15).

There are generally two ways of building such models in Keras: you
can directly subclass the `Model` class, or you can use the Functional API,
which lets you do more with less code. We’ll cover both approaches in chapter 7.

The topology of a model defines a *hypothesis space*. You may remember that
in chapter 1, we described machine learning as “searching for useful
representations of some input data, within a predefined
*space of possibilities*, using guidance from a feedback signal.”
By choosing a network topology, you constrain your space of possibilities
(hypothesis space) to a specific series of tensor operations, mapping
input data to output data. What you’ll then be searching for is a good set
of values for the weight tensors involved in these tensor operations.

To learn from data, you have to make assumptions about it. These assumptions
define what can be learned. As such, the structure of your hypothesis space —
the architecture of your model — is extremely important.
It encodes the assumptions you make about your problem,
the prior knowledge that the model starts with. For instance,
if you’re working on a two-class classification problem with a model made
of a single `Dense` layer with no activation (a pure affine transformation),
you are assuming that your two classes are linearly separable.

Picking the right network architecture is more an art than a science, and
although there are some best practices and principles you can rely on, only
practice can help you become a proper neural network architect. The next few
chapters will both teach you explicit principles for building neural networks
and help you develop intuition as to what works or doesn’t work for specific
problems. You’ll build a solid intuition about what type of
model architectures work for different kinds of problems, how to build
these networks in practice, how to pick the
right learning configuration, and how to tweak a model until it yields the
results you want to see.

### The “compile” step: Configuring the learning process

Once the model architecture is defined, you still have to
choose three more things:

* *Loss function (objective function)*  — The quantity that will
  be minimized during training. It represents a measure of success for
  the task at hand.

* *Optimizer*  — Determines how the network will be updated based on the loss
  function. It implements a specific variant of stochastic gradient descent (SGD).

* *Metrics* — The measures of success you want to monitor during training and
  validation, such as classification accuracy.
  Unlike the loss, training will not optimize directly for these metrics.
  As such, metrics don’t need to be differentiable.

Once you’ve picked your loss, optimizer, and metrics, you can use the
built-in `compile()` and `fit()` methods to start training your model.
Alternatively, you can write your own custom training loops —
we cover how to do this in chapter 7. It’s a lot more work!
For now, let’s take a look at `compile()` and `fit()`.

The `compile()` method configures the training process — you’ve already been
introduced to it in your very first neural network example in chapter 2.
It takes the arguments `optimizer`, `loss`, and `metrics` (a list):

```python
# Defines a linear classifier
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(
    # Specifies the optimizer by name: RMSprop (it's case-insensitive)
    optimizer="rmsprop",
    # Specifies the loss by name: mean squared error
    loss="mean_squared_error",
    # Specifies a list of metrics: in this case, only accuracy
    metrics=["accuracy"],
)
```

In the previous call to `compile()`, we passed the optimizer, loss, and metrics
as strings (such as `"rmsprop"`). These strings are actually
shortcuts that get converted to Python objects. For instance, `"rmsprop"` becomes
`keras.optimizers.RMSprop()`. Importantly, it’s also possible to specify these
arguments as object instances, like this:

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.BinaryAccuracy()],
)
```

This is useful if you want to pass your own custom losses or metrics or if
you want to further configure the objects you’re using — for instance, by
passing a `learning_rate` argument to the optimizer:

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
    loss=my_custom_loss,
    metrics=[my_custom_metric_1, my_custom_metric_2],
)
```

In chapter 7, we cover how to create custom losses and metrics. In general,
you won’t have to create your own losses, metrics,
or optimizers from scratch because Keras offers a wide range of built-in
options that is likely to include what you need:

* *Optimizers*
  + `SGD()` (with or without momentum)
  + `RMSprop()`
  + `Adam()`
  + Etc.
* *Losses*
  + `CategoricalCrossentropy()`
  + `SparseCategoricalCrossentropy()`
  + `BinaryCrossentropy()`
  + `MeanSquaredError()`
  + `KLDivergence()`
  + `CosineSimilarity()`
  + Etc.
* *Metrics*
  + `CategoricalAccuracy()`
  + `SparseCategoricalAccuracy()`
  + `BinaryAccuracy()`
  + `AUC()`
  + `Precision()`
  + `Recall()`
  + Etc.

Throughout this book, you’ll see concrete applications of many of these options.

### Picking a loss function

Choosing the right loss function for the right problem is extremely
important: your network will take any shortcut it can to minimize the loss.
So if the objective doesn’t fully correlate with success for the task at hand,
your network will end up doing things you may not have wanted. Imagine a
stupid, omnipotent AI trained via SGD, with this poorly chosen objective
function: “Maximize the average well-being of all humans alive.” To make its
job easier, this AI might choose to kill all humans except a few and focus on
the well-being of the remaining ones because average well-being isn’t affected
by how many humans are left. That might not be what you intended! Just
remember that all neural networks you build will be just as ruthless in
lowering their loss function, so choose the objective wisely, or you’ll have to
face unintended side effects.

Fortunately, when it comes to common problems such as classification,
regression, and sequence prediction, there are simple guidelines you can follow
to choose the correct loss. For instance, you’ll use binary crossentropy for a
two-class classification problem, categorical crossentropy for a many-class
classification problem, and so on. Only when you’re working on truly
new research problems will you have to develop your own loss functions.
In the next few chapters, we’ll detail explicitly which loss functions to
choose for a wide range of common tasks.

### Understanding the fit method

After `compile()` comes `fit()`. The `fit` method implements the training loop
itself. Its key arguments are

* The *data* (inputs and targets) to train on. It will typically be passed
  either in the form of NumPy arrays or a TensorFlow `Dataset` object. You’ll
  learn more about the `Dataset` API in the next chapters.
* The number of *epochs* to train for: how many times the training loop
  should iterate over the data passed.
* The batch size to use within each epoch of mini-batch gradient descent:
  the number of training examples considered to compute the gradients for
  one weight update step.

```python
history = model.fit(
    # The input examples, as a NumPy array
    inputs,
    # The corresponding training targets, as a NumPy array
    targets,
    # The training loop will iterate over the data 5 times.
    epochs=5,
    # The training loop will iterate over the data in batches of 128
    # examples.
    batch_size=128,
)
```

[Listing 3.34](#listing-3-34): Calling `fit` with NumPy data

The call to `fit` returns a `History` object. This object contains
a `history` field, which is a dict mapping key, such as `"loss"` or specific
metric names to the list of their per-epoch values:

```python
>>> history.history
{"binary_accuracy": [0.855, 0.9565, 0.9555, 0.95, 0.951],
 "loss": [0.6573270302042366,
  0.07434618508815766,
  0.07687718723714351,
  0.07412414988875389,
  0.07617757616937161]}
```

### Monitoring loss and metrics on validation data

The goal of machine learning is not to obtain models that perform well on the
training data, which is easy — all you have to do is follow the gradient.
The goal is to obtain models that perform well in
general, particularly on data points that the model has never encountered
before. Just because a model performs well on its training data doesn’t mean
it will perform well on data it has never seen! For
instance, it’s possible that your model could end up merely *memorizing* a
mapping between your training samples and their targets, which would be
useless for the task of predicting targets for data the model has never seen
before. We’ll go over this point in much more detail in the chapter 5.

To keep an eye on how the model does on new data, it’s standard practice
to reserve a subset of the training data as “validation data”: you won’t
be training the model on this data, but you will use it to compute a loss value
and metrics value. You do this by using the `validation_data` argument in `fit()`.
Like the training data, the validation data could be passed as NumPy arrays
or as a TensorFlow `Dataset` object.

```python
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# To avoid having samples from only one class in the validation data,
# shuffles the inputs and targets using a random indices permutation
indices_permutation = np.random.permutation(len(inputs))
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]

# Reserves 30% of the training inputs and targets for "validation."
# (We'll exclude these samples from training and reserve them to
# compute the "validation loss" and metrics).
num_validation_samples = int(0.3 * len(inputs))
val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples:]
training_targets = shuffled_targets[num_validation_samples:]
model.fit(
    # Training data, used to update the weights of the model
    training_inputs,
    training_targets,
    epochs=5,
    batch_size=16,
    # Validation data, used only to monitor the "validation loss" and
    # metrics
    validation_data=(val_inputs, val_targets),
)
```

[Listing 3.35](#listing-3-35): Using the validation data argument

The value of the loss on the validation data is called the
*validation loss*, to distinguish it from the *training loss*. Note that
it’s essential to keep the training data and validation data strictly separate:
the purpose of validation is to monitor whether what the model is learning is
actually useful on new data. If any of the validation data has been seen
by the model during training, your validation loss and metrics will be flawed.

If you want to compute the validation loss and metrics after training
is complete, you can call the `evaluate` method:

`loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size=128)`

`evaluate()` will iterate in batches (of size `batch_size`) over the data passed
and return a list of scalars, where the first entry is the validation loss
and the following entries are the validation metrics. If the model has no
metrics, only the validation loss is returned (rather than a list).

### Inference: Using a model after training

Once you’ve trained your model, you’re going to want to use it to make predictions
on new data. This is called *inference*.
To do this, a naive approach would simply be to `__call__` the model:

```python
# Takes a NumPy array or a tensor for your current backend and returns
# a tensor for your current backend
predictions = model(new_inputs)
```

However, this will process all inputs in `new_inputs` at once, which may
not be feasible if you’re looking at a lot of data (in particular, it may
require more memory than your GPU has).

A better way to do inference is to use the `predict()` method. It will iterate
over the data in small batches and return a NumPy array of predictions.
And unlike `__call__`, it can also process TensorFlow `Dataset` objects:

```python
# Takes a NumPy array or a Dataset and returns a NumPy array
predictions = model.predict(new_inputs, batch_size=128)
```

For instance, if we use `predict()` on some of our validation data with the linear
model we trained earlier, we get scalar scores that correspond to the model’s
prediction for each input sample:

```python
>>> predictions = model.predict(val_inputs, batch_size=128)
>>> print(predictions[:10])
[[0.3590725 ]
 [0.82706255]
 [0.74428225]
 [0.682058  ]
 [0.7312616 ]
 [0.6059811 ]
 [0.78046083]
 [0.025846  ]
 [0.16594526]
 [0.72068727]]
```

For now, this is all you need to know about Keras models. At this point, you
are ready to move on to solving real-world machine problems with Keras,
in the next chapter.

## Summary

* TensorFlow, PyTorch, and JAX are three popular low-level
  frameworks for numerical computation and autodifferentiation.
  They all have their own way of doing things and their own strengths and weaknesses.
* Keras is a high-level API for building and training neural networks. It can be used with either
  TensorFlow, PyTorch, or JAX — just pick the backend you like best.
* The central class of Keras is the `Layer`. A layer encapsulates some weights
  and some computation. Layers are assembled into models.
* Before you start training a model, you need to pick an optimizer, a loss,
  and some metrics, which you specify via the `model.compile()` method.
* To train a model, you can use the `fit()` method, which runs mini-batch gradient
  descent for you. You can also use it to monitor your loss and metrics on
  validation data, a set of inputs that the model doesn’t see during training.
* Once your model is trained, you can use the `model.predict()` method to generate
  predictions on new inputs.

#### **Tiếng Việt (Vietnamese)**

# Chương 3: Giới thiệu về TensorFlow, PyTorch, JAX và Keras

Chương này bao gồm

* Xem xét kỹ hơn tất cả các nền tảng học sâu chính và mối quan hệ của chúng
* Tổng quan về cách chuyển các khái niệm deep learning cốt lõi sang mã trên toàn bộ
tất cả các khuôn khổ

Chương này nhằm cung cấp cho bạn mọi thứ bạn cần để bắt đầu học sâu trong thực tế. Trước tiên, bạn sẽ làm quen với ba framework deep learning phổ biến có thể sử dụng với Keras:

* TensorFlow (<https://tensorflow.org>)
* PyTorch (<https://pytorch.org/>)
* JAX (<https://jax.readthedocs.io/>)

Sau đó, dựa trên mối liên hệ đầu tiên mà bạn có với Keras trong chương 2, chúng ta sẽ xem xét các thành phần cốt lõi của mạng thần kinh và cách chúng chuyển sang API Keras.

Đến cuối chương này, bạn sẽ sẵn sàng chuyển sang các ứng dụng thực tế, thực tế - sẽ bắt đầu từ chương 4.

## Sơ lược về lịch sử của các framework deep learning

Trong thế giới thực, bạn sẽ không viết mã cấp thấp từ đầu như chúng ta đã làm ở cuối chương 2. Thay vào đó, bạn sẽ sử dụng một framework. Ngoài Keras, các framework deep learning chính hiện nay là JAX, TensorFlow và PyTorch. Cuốn sách này sẽ dạy cho bạn về cả bốn điều đó.

Nếu bạn mới bắt đầu học sâu, có vẻ như tất cả các khuôn khổ này đã tồn tại từ lâu. Trên thực tế, tất cả chúng đều khá mới, trong đó Keras là sản phẩm lâu đời nhất trong số 4 sản phẩm (ra mắt vào tháng 3 năm 2015). Tuy nhiên, những ý tưởng đằng sau những khuôn khổ này có lịch sử lâu dài - bài báo đầu tiên về vi phân tự động được xuất bản vào năm 1964[[1]](#footnote-1)

Tất cả các khung này kết hợp ba tính năng chính:

* Một cách tính toán độ dốc cho các hàm khả vi tùy ý (phân biệt tự động)
* Một cách để chạy tính toán tensor trên CPU và GPU (và thậm chí có thể trên phần cứng học sâu chuyên dụng khác)
* Cách phân phối tính toán trên nhiều thiết bị hoặc nhiều máy tính, chẳng hạn như nhiều GPU trên một máy tính hoặc
thậm chí nhiều GPU trên nhiều máy tính riêng biệt

Cùng với nhau, ba tính năng đơn giản này mở ra tất cả các phương pháp học sâu hiện đại.

Phải mất một thời gian dài, lĩnh vực này mới phát triển được các giải pháp mạnh mẽ cho cả ba vấn đề và đóng gói các giải pháp đó ở dạng có thể tái sử dụng. Kể từ khi ra đời vào những năm 1960 và cho đến những năm 2000, khả năng tự vi phân không có ứng dụng thực tế nào trong học máy - những người làm việc với mạng thần kinh chỉ đơn giản là viết logic gradient của riêng họ bằng tay, thường bằng ngôn ngữ như C++. Trong khi đó, việc lập trình GPU gần như là không thể.

Mọi thứ bắt đầu thay đổi dần dần vào cuối những năm 2000. Đầu tiên, Python và hệ sinh thái của nó đang dần trở nên phổ biến trong cộng đồng khoa học, thu hút được sự chú ý hơn MATLAB và C++. Thứ hai, NVIDIA phát hành CUDA vào năm 2006, mở ra khả năng xây dựng mạng lưới thần kinh có thể chạy trên GPU tiêu dùng. Trọng tâm ban đầu của CUDA là mô phỏng vật lý thay vì học máy, nhưng điều đó không ngăn cản các nhà nghiên cứu học máy bắt đầu triển khai mạng thần kinh dựa trên CUDA từ năm 2009 trở đi. Chúng thường là những triển khai một lần chạy trên một GPU duy nhất mà không có bất kỳ sự phân biệt tự động nào.

Khung đầu tiên cho phép tự động phân biệt và tính toán GPU để đào tạo các mô hình học sâu là Theano, vào khoảng năm 2009. Theano là tổ tiên về mặt khái niệm của tất cả các công cụ học sâu hiện đại. Nó bắt đầu nhận được sự chú ý tốt trong cộng đồng nghiên cứu máy học vào năm 2013–2014, sau khi kết quả của cuộc thi ImageNet 2012 đã khơi dậy sự quan tâm của thế giới đối với việc học sâu. Cùng lúc đó, một số thư viện deep learning hỗ trợ GPU khác bắt đầu trở nên phổ biến trong thế giới thị giác máy tính — đặc biệt là Torch 7 (dựa trên Lua) và Caffe (dựa trên C++). Keras ra mắt vào đầu năm 2015 dưới dạng thư viện deep learning cấp cao hơn, dễ sử dụng hơn do Theano cung cấp và nó nhanh chóng thu hút được sự chú ý của hàng nghìn người đang học sâu vào thời điểm đó.

Sau đó, vào cuối năm 2015, Google ra mắt TensorFlow, lấy nhiều ý tưởng chính từ Theano và bổ sung hỗ trợ cho tính toán phân tán quy mô lớn. Việc phát hành TensorFlow là một bước ngoặt thúc đẩy việc học sâu trong hệ tư tưởng của nhà phát triển chính thống. Keras ngay lập tức bổ sung hỗ trợ cho TensorFlow. Vào giữa năm 2016, hơn một nửa số người dùng TensorFlow đã sử dụng nó thông qua Keras.

Để đáp lại TensorFlow, Meta (lúc đó được đặt tên là Facebook) đã ra mắt PyTorch khoảng một năm sau, lấy ý tưởng từ Chainer (một framework thích hợp nhưng sáng tạo được ra mắt vào giữa năm 2015, hiện đã ngừng hoạt động từ lâu) và NumPy-Autograd, một thư viện tự động phân biệt chỉ dành cho CPU cho NumPy do Maclaurin và cộng sự phát hành. vào năm 2014. Trong khi đó, Google đã phát hành TPU như một giải pháp thay thế cho GPU, cùng với XLA, một trình biên dịch hiệu suất cao được phát triển để cho phép TensorFlow chạy trên TPU.

Vài năm sau, tại Google, Matthew Johnson — một trong những nhà phát triển từng làm việc trên NumPy-Autograd — đã phát hành JAX như một cách thay thế để sử dụng tính năng tự phân biệt với XLA. JAX nhanh chóng thu hút được sự chú ý của các nhà nghiên cứu nhờ API tối giản và khả năng mở rộng cao. Ngày nay, Keras, TensorFlow, PyTorch và JAX là những framework hàng đầu trong thế giới deep learning.

Nhìn lại lịch sử hỗn loạn này, chúng ta có thể hỏi: Điều gì tiếp theo? Liệu một khuôn khổ mới sẽ xuất hiện vào ngày mai? Chúng ta sẽ chuyển sang ngôn ngữ lập trình mới hay nền tảng phần cứng mới?

Nếu bạn hỏi tôi, hôm nay có ba điều chắc chắn:

* Python đã thắng. Hệ sinh thái khoa học dữ liệu và học máy của nó đơn giản là có quá nhiều động lực vào thời điểm này.
Sẽ không có một ngôn ngữ hoàn toàn mới nào có thể thay thế nó - ít nhất là trong 15 năm tới.
* Chúng ta đang ở trong một thế giới đa khung - cả bốn khung đều được thiết lập tốt và khó có thể thay đổi trong vài năm tới.
Bạn nên tìm hiểu một chút về từng cái một.
Tuy nhiên, rất có thể các khung công tác *mới* sẽ trở nên phổ biến trong tương lai,
ngoài họ; MLX được phát hành gần đây của Apple có thể là một ví dụ như vậy.
Trong bối cảnh này, sử dụng Keras là một lợi thế đáng kể: bạn sẽ có thể chạy các mô hình Keras hiện có của mình
trên bất kỳ khung công tác mới sắp ra mắt nào thông qua chương trình phụ trợ Keras mới. Keras sẽ tiếp tục cung cấp sự ổn định trong tương lai
cho các nhà phát triển máy học trong tương lai, giống như đã từng làm kể từ năm 2015 - thời kỳ mà cả TensorFlow, PyTorch và JAX đều chưa tồn tại.
* Các chip mới chắc chắn có thể xuất hiện trong tương lai, cùng với GPU của NVIDIA và TPU của Google.
Ví dụ: dòng GPU của AMD có thể sẽ có những ngày tươi sáng phía trước.
Nhưng bất kỳ con chip mới nào như vậy sẽ phải hoạt động với các khuôn khổ hiện có để đạt được lực kéo.
Phần cứng mới không có khả năng làm gián đoạn quy trình làm việc của bạn.

## Các khung này liên quan với nhau như thế nào

Keras, TensorFlow, PyTorch và JAX đều không có cùng bộ tính năng và không thể thay thế cho nhau. Chúng có một số điểm trùng lặp, nhưng ở mức độ lớn, chúng phục vụ các vai trò khác nhau cho các trường hợp sử dụng khác nhau. Sự khác biệt lớn nhất là giữa Keras và ba người khác. Keras là một framework cấp cao, trong khi các framework khác ở cấp độ thấp hơn. Hãy tưởng tượng bạn đang xây một ngôi nhà. Keras giống như một bộ công cụ xây dựng sẵn: nó cung cấp giao diện hợp lý để thiết lập và đào tạo mạng lưới thần kinh. Ngược lại, TensorFlow, PyTorch và JAX giống như những nguyên liệu thô được sử dụng trong xây dựng.

Như bạn đã thấy trong các chương trước, việc huấn luyện mạng nơ-ron xoay quanh các khái niệm sau:

* *Đầu tiên, thao tác tensor cấp thấp* — Cơ sở hạ tầng làm nền tảng cho
tất cả các máy học hiện đại. Điều này chuyển thành các API cấp thấp được tìm thấy trong TensorFlow,
PyTorch[[2]](#footnote-2) và JAX:
+ *Tensor*, bao gồm các tensor đặc biệt lưu trữ trạng thái của mạng (*biến*)
+ *Các phép toán tensor* chẳng hạn như phép cộng, `relu` hoặc `matmul`
+ *lan truyền ngược*, một cách tính gradient của các biểu thức toán học
* *Khái niệm học sâu thứ hai, cấp cao* — Điều này chuyển sang API Keras:
+ *Các lớp*, được kết hợp thành một *mô hình*
+ Một *hàm mất*, xác định tín hiệu phản hồi được sử dụng cho việc học
+ Một *trình tối ưu hóa*, xác định quá trình học tập diễn ra
+ *Số liệu* để đánh giá hiệu suất của mô hình, chẳng hạn như độ chính xác
+ Một *vòng huấn luyện* thực hiện giảm độ dốc ngẫu nhiên theo lô nhỏ

Hơn nữa, Keras còn độc đáo ở chỗ nó không phải là một framework hoàn toàn độc lập. Nó cần một *công cụ phụ trợ* để chạy, (xem hình 3.4), giống như một bộ công cụ xây dựng nhà tiền chế cần tìm nguồn vật liệu xây dựng từ đâu đó. TensorFlow, PyTorch và JAX đều có thể được sử dụng làm phần phụ trợ của Keras. Ngoài ra, Keras có thể chạy trên NumPy, nhưng vì NumPy không cung cấp API cho gradient nên quy trình công việc của Keras trên NumPy bị hạn chế trong việc đưa ra dự đoán từ mô hình – việc đào tạo là không thể.

Bây giờ bạn đã hiểu rõ hơn về cách tất cả các framework này hình thành và chúng liên quan với nhau như thế nào, hãy cùng tìm hiểu xem làm việc với chúng sẽ như thế nào. Chúng tôi sẽ trình bày chúng theo thứ tự thời gian: TensorFlow trước, sau đó là PyTorch và cuối cùng là JAX.

## Giới thiệu về TensorFlow

TensorFlow là một khung học máy nguồn mở dựa trên Python được phát triển chủ yếu bởi Google. Bản phát hành đầu tiên của nó là vào tháng 11 năm 2015, tiếp theo là bản phát hành v1 vào tháng 2 năm 2017 và bản phát hành v2 vào tháng 10 năm 2019. TensorFlow được sử dụng nhiều trong các ứng dụng học máy cấp sản xuất trong toàn ngành.

Điều quan trọng cần lưu ý là TensorFlow không chỉ là một thư viện. Nó thực sự là một nền tảng, nơi chứa một hệ sinh thái rộng lớn bao gồm các thành phần, một số do Google phát triển, một số do bên thứ ba phát triển. Ví dụ: có TFX để quản lý quy trình học máy mạnh mẽ trong ngành, TF-Serving để triển khai sản xuất, Bộ công cụ tối ưu hóa TF để lượng tử hóa và cắt tỉa mô hình cũng như TFLite và MediaPipe để triển khai ứng dụng di động.

Cùng với nhau, các thành phần này bao gồm rất nhiều trường hợp sử dụng, từ nghiên cứu tiên tiến đến ứng dụng sản xuất quy mô lớn.

### Những bước đầu tiên với TensorFlow

Trong các đoạn tiếp theo, bạn sẽ làm quen với tất cả những điều cơ bản về TensorFlow. Chúng tôi sẽ đề cập đến các khái niệm chính sau:

* Tenxơ và biến
* Các phép toán số trong TensorFlow
* Tính toán độ dốc bằng `GradientTape`
* Làm cho các hàm TensorFlow hoạt động nhanh chóng bằng cách sử dụng tính năng biên dịch đúng lúc

Sau đó, chúng tôi sẽ kết thúc phần giới thiệu bằng một ví dụ hoàn chỉnh: triển khai hồi quy tuyến tính TensorFlow thuần túy.

Hãy làm cho các tensor đó trôi chảy.

#### Tensor và biến trong TensorFlow

Để làm bất cứ điều gì trong TensorFlow, chúng ta sẽ cần một số tensor. Có một số cách khác nhau để bạn có thể tạo chúng.

##### Các tensor không đổi

Cần phải tạo các tensor với một số giá trị ban đầu, vì vậy các cách phổ biến để tạo tensor là thông qua `tf.ones` (tương đương với `np.ones`) và `tf.zeros` (tương đương với `np.zeros`). Bạn cũng có thể tạo một tensor từ các giá trị Python hoặc NumPy bằng cách sử dụng `tf.constant`.

```python
>>> import tensorflow as tf
>>> # Equivalent to np.ones(shape=(2, 1))
>>> tf.ones(shape=(2, 1))
tf.Tensor([[1.], [1.]], shape=(2, 1), dtype=float32)
>>> # Equivalent to np.zeros(shape=(2, 1))
>>> tf.zeros(shape=(2, 1))
tf.Tensor([[0.], [0.]], shape=(2, 1), dtype=float32)
>>> # Equivalent to np.array([1, 2, 3], dtype="float32")
>>> tf.constant([1, 2, 3], dtype="float32")
tf.Tensor([1., 2., 3.], shape=(3,), dtype=float32)
```

[Liệt kê 3.1](#listing-3-1): Tenxơ toàn số một hoặc toàn số không

##### Các tensor ngẫu nhiên

Bạn cũng có thể tạo các tensor chứa các giá trị ngẫu nhiên thông qua một trong các phương thức của mô-đun con `tf.random` (tương đương với mô-đun con `np.random`).

```python
>>> # Tensor of random values drawn from a normal distribution with
>>> # mean 0 and standard deviation 1. Equivalent to
>>> # np.random.normal(size=(3, 1), loc=0., scale=1.).
>>> x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
>>> print(x)
tf.Tensor(
[[-0.14208166]
 [-0.95319825]
 [ 1.1096532 ]], shape=(3, 1), dtype=float32)
>>> # Tensor of random values drawn from a uniform distribution between
>>> # 0 and 1. Equivalent to np.random.uniform(size=(3, 1), low=0.,
>>> # high=1.).
>>> x = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
>>> print(x)
tf.Tensor(
[[0.33779848]
 [0.06692922]
 [0.7749394 ]], shape=(3, 1), dtype=float32)
```

[Liệt kê 3.2](#listing-3-2): Các tensor ngẫu nhiên

##### Phép gán tensor và lớp Biến

Một sự khác biệt đáng kể giữa mảng NumPy và tensor TensorFlow là tensor TensorFlow không thể gán được: chúng không đổi. Ví dụ: trong NumPy, bạn có thể thực hiện các thao tác sau.

```python
import numpy as np

x = np.ones(shape=(2, 2))
x[0, 0] = 0.0
```

[Liệt kê 3.3](#listing-3-3): Các mảng NumPy có thể được gán

Hãy thử làm điều tương tự trong TensorFlow: bạn sẽ gặp lỗi ``Đối tượng EagerTensor không hỗ trợ gán mục`.

```python
x = tf.ones(shape=(2, 2))
# This will fail, as a tensor isn't assignable.
x[0, 0] = 0.0
```

[Liệt kê 3.4](#listing-3-4): Không thể gán các tensor TensorFlow

Để huấn luyện một mô hình, chúng ta cần cập nhật trạng thái của nó, đó là một tập hợp các tensor. Nếu tensor không thể gán được thì chúng ta phải làm thế nào? Đó là nơi các biến xuất hiện. `tf.Variable` là lớp dùng để quản lý trạng thái có thể sửa đổi trong TensorFlow.

Để tạo một biến, bạn cần cung cấp một số giá trị ban đầu, chẳng hạn như một tensor ngẫu nhiên.

```python
>>> v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
>>> print(v)
array([[-0.75133973],
       [-0.4872893 ],
       [ 1.6626885 ]], dtype=float32)>
```

[Liệt kê 3.5](#listing-3-5): Tạo một `tf.Variable`

Trạng thái của một biến có thể được sửa đổi thông qua phương thức `sign` của nó.

```python
>>> v.assign(tf.ones((3, 1)))
array([[1.],
       [1.],
       [1.]], dtype=float32)>
```

[Liệt kê 3.6](#listing-3-6): Gán một giá trị cho một `Biến`

Phép gán cũng có tác dụng đối với một tập hợp con các hệ số.

```python
>>> v[0, 0].assign(3.)
array([[3.],
       [1.],
       [1.]], dtype=float32)>
```

[Liệt kê 3.7](#listing-3-7): Gán một giá trị cho tập hợp con của một `Biến`

Tương tự, `sign_add` và `sign_sub` là những giá trị tương đương hiệu quả của `+=` và `-=`.

```python
>>> v.assign_add(tf.ones((3, 1)))
array([[2.],
       [2.],
       [2.]], dtype=float32)>
```

[Liệt kê 3.8](#listing-3-8): Sử dụng `sign_add`

#### Hoạt động của Tensor: Làm toán trong TensorFlow

Cũng giống như NumPy, TensorFlow cung cấp một tập hợp lớn các phép toán tensor để biểu diễn các công thức toán học. Dưới đây là một vài ví dụ.

```python
a = tf.ones((2, 2))
# Takes the square, same as np.square
b = tf.square(a)
# Takes the square root, same as np.sqrt
c = tf.sqrt(a)
# Adds two tensors (element-wise)
d = b + c
# Takes the product of two tensors (see chapter 2), same as np.matmul
e = tf.matmul(a, b)
# Concatenates a and b along axis 0, same as np.concatenate
f = tf.concat((a, b), axis=0)
```

[Liệt kê 3.9](#listing-3-9): Một vài phép toán cơ bản trong TensorFlow

Đây là lớp tương đương với lớp `Dense` mà chúng ta đã thấy ở chương 2:

```python
def dense(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
```

#### Chuyển màu trong TensorFlow: Cái nhìn thứ hai về API gradientTape

Cho đến nay, TensorFlow có vẻ rất giống NumPy. Nhưng đây là điều mà NumPy không thể làm: truy xuất độ dốc của bất kỳ biểu thức khả vi nào đối với bất kỳ đầu vào nào của nó. Chỉ cần mở phạm vi `GradientTape`, áp dụng một số tính toán cho một hoặc một số tensor đầu vào và truy xuất độ dốc của kết quả tương ứng với đầu vào.

```python
input_var = tf.Variable(initial_value=3.0)
with tf.GradientTape() as tape:
    result = tf.square(input_var)
gradient = tape.gradient(result, input_var)
```

[Liệt kê 3.10](#listing-3-10): Sử dụng `GradientTape`

Điều này được sử dụng phổ biến nhất để truy xuất độ dốc của việc mất mô hình so với trọng số của nó: `gradients = tape.gradient(loss,weights)`.

Trong chương 2, bạn đã biết cách `GradientTape` hoạt động trên một đầu vào hoặc một danh sách các đầu vào và cách các đầu vào có thể là vô hướng hoặc tensor nhiều chiều.

Cho đến nay, bạn chỉ mới thấy trường hợp các tensor đầu vào trong `tape.gradient()` là các biến TensorFlow. Trên thực tế, những đầu vào này có thể là bất kỳ tensor tùy ý nào. Tuy nhiên, chỉ *các biến có thể huấn luyện* mới được theo dõi theo mặc định. Với một tenxơ không đổi, bạn sẽ phải đánh dấu thủ công nó là đang được theo dõi bằng cách gọi `tape.watch()` trên đó.

```python
input_const = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(input_const)
    result = tf.square(input_const)
gradient = tape.gradient(result, input_const)
```

[Liệt kê 3.11](#listing-3-11): Sử dụng `GradientTape` với đầu vào tensor không đổi

Tại sao? Bởi vì sẽ quá tốn kém nếu lưu trữ trước thông tin cần thiết để tính toán độ dốc của bất kỳ thứ gì đối với bất kỳ thứ gì. Để tránh lãng phí tài nguyên, băng cần biết nên xem gì. Các biến có thể huấn luyện được theo dõi theo mặc định vì tính toán độ dốc của tổn thất đối với danh sách các biến có thể huấn luyện là trường hợp sử dụng phổ biến nhất của băng gradient.

Băng chuyển màu là một tiện ích mạnh mẽ, thậm chí có khả năng tính toán *độ dốc bậc hai* - nghĩa là độ dốc của độ dốc. Chẳng hạn, độ dốc vị trí của một vật thể theo thời gian là tốc độ của vật thể đó và độ dốc bậc hai là gia tốc của nó.

Nếu bạn đo vị trí của một quả táo rơi dọc theo trục thẳng đứng theo thời gian và thấy rằng nó xác minh `vị trí (thời gian) = 4,9 * thời gian ** 2`, thì gia tốc của nó là bao nhiêu? Hãy sử dụng hai băng chuyển màu lồng nhau để tìm hiểu.

```python
time = tf.Variable(0.0)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time**2
    speed = inner_tape.gradient(position, time)
# We use the outer tape to compute the gradient of the gradient from
# the inner tape. Naturally, the answer is 4.9 * 2 = 9.8.
acceleration = outer_tape.gradient(speed, time)
```

[Liệt kê 3.12](#listing-3-12): Sử dụng các băng gradient lồng nhau để tính toán các gradient bậc hai

#### Làm cho các hàm TensorFlow nhanh chóng bằng cách sử dụng trình biên dịch

Tất cả mã TensorFlow bạn đã viết cho đến nay đều đang được thực thi một cách “háo hức”. Điều này có nghĩa là các thao tác được thực thi lần lượt trong thời gian chạy Python, giống như bất kỳ mã Python hoặc mã NumPy nào. Việc thực thi háo hức là điều tuyệt vời để gỡ lỗi, nhưng nó thường khá chậm. Thông thường, việc song song hóa một số thao tác tính toán hoặc "cầu chì" — thay thế hai thao tác liên tiếp, như `matmul` theo sau là `relu`, bằng một thao tác duy nhất, hiệu quả hơn thực hiện điều tương tự mà không hiện thực hóa đầu ra trung gian.

Điều này có thể đạt được thông qua *biên dịch*. Ý tưởng chung của việc biên dịch là sử dụng một số hàm nhất định mà bạn đã viết bằng Python, loại bỏ chúng khỏi Python, tự động viết lại chúng thành một “chương trình được biên dịch” nhanh hơn và hiệu quả hơn, sau đó gọi chương trình đó từ thời gian chạy Python.

Lợi ích chính của việc biên dịch là cải thiện hiệu suất. Cũng có một nhược điểm: mã bạn viết không còn là mã được thực thi nữa, điều này có thể khiến quá trình gỡ lỗi trở nên khó khăn. Chỉ bật tính năng biên dịch sau khi bạn đã sửa lỗi mã của mình trong thời gian chạy Python.

Bạn có thể áp dụng quá trình biên dịch cho bất kỳ hàm TensorFlow nào bằng cách gói nó trong một trình trang trí `tf.function`, như thế này:

```python
@tf.function
def dense(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
```

Khi bạn thực hiện việc này, bất kỳ lệnh gọi nào tới `dense()` đều được thay thế bằng lệnh gọi đến một chương trình đã biên dịch triển khai phiên bản hàm được tối ưu hóa hơn. Lần gọi hàm đầu tiên sẽ lâu hơn một chút vì TensorFlow sẽ biên dịch mã của bạn. Điều này chỉ xảy ra một lần — tất cả các lệnh gọi tiếp theo tới cùng một chức năng sẽ diễn ra nhanh chóng.

TensorFlow có hai chế độ biên dịch:

* Đầu tiên, chế độ mặc định mà chúng tôi gọi là “chế độ biểu đồ”. Bất kỳ chức năng
được trang trí bằng `@tf.function` chạy ở chế độ biểu đồ.
* Thứ hai, biên dịch bằng XLA, một trình biên dịch hiệu suất cao dành cho ML (viết tắt là
cho Đại số tuyến tính gia tốc). Bạn có thể bật nó bằng cách chỉ định
`jit_compile=True`, như thế này:

```python
@tf.function(jit_compile=True)
def dense(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
```

Thông thường, việc biên dịch hàm bằng XLA sẽ làm cho hàm đó chạy nhanh hơn chế độ biểu đồ — mặc dù sẽ mất nhiều thời gian hơn để thực thi hàm trong lần đầu tiên vì trình biên dịch có nhiều việc phải làm hơn.

### Một ví dụ toàn diện: Trình phân loại tuyến tính trong TensorFlow thuần túy

Bạn biết về tensor, biến và các phép toán tensor cũng như biết cách tính gradient. Thế là đủ để xây dựng bất kỳ mô hình học máy dựa trên TensorFlow nào dựa trên độ dốc giảm dần. Hãy xem qua một ví dụ từ đầu đến cuối để đảm bảo mọi thứ đều rõ ràng.

Trong một cuộc phỏng vấn xin việc về học máy, bạn có thể được yêu cầu triển khai bộ phân loại tuyến tính ngay từ đầu: một nhiệm vụ rất đơn giản đóng vai trò như một bộ lọc giữa những ứng viên có nền tảng học máy tối thiểu và những người không có nền tảng về học máy. Hãy giúp bạn vượt qua bộ lọc đó và sử dụng kiến ​​thức mới tìm thấy của bạn về TensorFlow để triển khai bộ phân loại tuyến tính như vậy.

Trước tiên, hãy nghĩ ra một số dữ liệu tổng hợp có thể phân tách tuyến tính độc đáo để làm việc: hai loại điểm trong mặt phẳng 2D.

```python
import numpy as np

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    # Generates the first class of points: 1,000 random 2D points with
    # specified "mean" and "covariance matrix." Intuitively, the
    # "covariance matrix" describes the shape of the point cloud, and
    # the "mean" describes its position in the plane. `cov=[[1,
    # 0.5],[0.5, 1]]` corresponds to "an oval-like point cloud oriented
    # from bottom left to top right."
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    # Generates the other class of points with a different mean and the
    # same covariance matrix (point cloud with a different position and
    # the same shape)
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)
```

[Liệt kê 3.13](#listing-3-13): Tạo hai lớp điểm ngẫu nhiên trong mặt phẳng 2D

`mẫu_âm` và `mẫu_dương` đều là mảng có hình dạng `(1000, 2)`. Hãy xếp chúng thành một mảng duy nhất có hình dạng `(2000, 2)`.

```python
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
```

[Liệt kê 3.14](#listing-3-14): Xếp chồng hai lớp thành một mảng có hình dạng `(2000, 2)`

Hãy tạo các nhãn mục tiêu tương ứng, một mảng gồm các số 0 và 1 có hình dạng `(2000, 1)`, trong đó `targets[i, 0]` là 0 nếu `inputs[i]` thuộc về lớp 0 (và ngược lại).

```python
targets = np.vstack(
    (
        np.zeros((num_samples_per_class, 1), dtype="float32"),
        np.ones((num_samples_per_class, 1), dtype="float32"),
    )
)
```

[Danh sách 3.15](#listing-3-15): Tạo các mục tiêu tương ứng (0 và 1)

Hãy vẽ biểu đồ dữ liệu của chúng ta bằng Matplotlib, một thư viện trực quan hóa dữ liệu Python nổi tiếng (thư viện này được cài đặt sẵn trong Colab nên bạn không cần phải tự cài đặt), như trong hình 3.1.

```python
import matplotlib.pyplot as plt

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()
```

[Liệt kê 3.16](#listing-3-16): Vẽ đồ thị của hai lớp điểm

![](../images/ch03/linear_model_inputs.282fc3b6.png)

[Figure 3.1](#figure-3-1): Our synthetic data: two classes of random points in the 2D plane

Bây giờ, hãy tạo một bộ phân loại tuyến tính có thể học cách tách hai đốm màu này. Bộ phân loại tuyến tính là một phép biến đổi affine (`dự đoán = matmul(input, W) + b`) được huấn luyện để giảm thiểu bình phương chênh lệch giữa dự đoán và mục tiêu.

Như bạn sẽ thấy, đây thực sự là một ví dụ đơn giản hơn nhiều so với ví dụ toàn diện về mạng thần kinh hai lớp đồ chơi ở cuối chương 2. Tuy nhiên, lần này, bạn sẽ có thể hiểu mọi thứ về mã, từng dòng một.

Hãy tạo các biến `W` và `b`, được khởi tạo với các giá trị ngẫu nhiên và số 0 tương ứng.

```python
# The inputs will be 2D points.
input_dim = 2
# The output predictions will be a single score per sample (close to 0
# if the sample is predicted to be in class 0, and close to 1 if the
# sample is predicted to be in class 1).
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
```

[Liệt kê 3.17](#listing-3-17): Tạo các biến phân loại tuyến tính

Đây là chức năng chuyển tiếp của chúng tôi.

```python
def model(inputs, W, b):
    return tf.matmul(inputs, W) + b
```

[Liệt kê 3.18](#listing-3-18): Hàm chuyển tiếp

Vì trình phân loại tuyến tính của chúng tôi hoạt động trên đầu vào 2D nên `W` thực sự chỉ là hai hệ số vô hướng: `W = [[w1], [w2]]`. Trong khi đó, `b` là hệ số vô hướng đơn. Như vậy, đối với điểm đầu vào cho trước `[x, y]`, giá trị dự đoán của nó là `prediction = [[w1], [w2]] • [x, y] + b = w1 * x + w2 * y + b`.

Đây là chức năng mất mát của chúng tôi.

```python
def mean_squared_error(targets, predictions):
    # per_sample_losses will be a tensor with the same shape as targets
    # and predictions, containing per-sample loss scores.
    per_sample_losses = tf.square(targets - predictions)
    # We need to average these per-sample loss scores into a single
    # scalar loss value: reduce_mean does this.
    return tf.reduce_mean(per_sample_losses)
```

[Liệt kê 3.19](#listing-3-19): Hàm mất mát sai số bình phương trung bình

Bây giờ, chúng ta chuyển sang bước huấn luyện, bước này nhận một số dữ liệu huấn luyện và cập nhật các trọng số `W` và `b` để giảm thiểu việc mất dữ liệu.

```python
learning_rate = 0.1

# Wraps the function in a tf.function decorator to speed it up
@tf.function(jit_compile=True)
def training_step(inputs, targets, W, b):
    # Forward pass, inside of a gradient tape scope
    with tf.GradientTape() as tape:
        predictions = model(inputs, W, b)
        loss = mean_squared_error(predictions, targets)
    # Retrieves the gradient of the loss with regard to weights
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    # Updates the weights
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss
```

[Liệt kê 3.20](#listing-3-20): Hàm bước huấn luyện

Để đơn giản, chúng tôi sẽ thực hiện *đào tạo theo đợt* thay vì *đào tạo theo đợt nhỏ*: chúng tôi sẽ chạy từng bước huấn luyện (tính toán độ dốc và cập nhật trọng số) trên toàn bộ dữ liệu, thay vì lặp lại dữ liệu theo từng đợt nhỏ. Một mặt, điều này có nghĩa là mỗi bước huấn luyện sẽ mất nhiều thời gian hơn để chạy vì chúng tôi tính toán chuyển tiếp và độ dốc cho 2.000 mẫu cùng một lúc. Mặt khác, mỗi lần cập nhật gradient sẽ hiệu quả hơn nhiều trong việc giảm tổn thất trên dữ liệu huấn luyện, vì nó sẽ bao gồm thông tin từ tất cả các mẫu huấn luyện thay vì chỉ 128 mẫu ngẫu nhiên. Do đó, chúng tôi sẽ cần ít bước đào tạo hơn và nên sử dụng tốc độ học tập lớn hơn tốc độ chúng tôi thường sử dụng cho đào tạo theo đợt nhỏ (chúng tôi sẽ sử dụng `learning_rate = 0,1`, như đã xác định trước đó).

```python
for step in range(40):
    loss = training_step(inputs, targets, W, b)
    print(f"Loss at step {step}: {loss:.4f}")
```

[Liệt kê 3.21](#listing-3-21): Vòng lặp đào tạo theo lô

Sau 40 bước, mức suy giảm huấn luyện dường như đã ổn định ở khoảng 0,025. Hãy vẽ sơ đồ cách mô hình tuyến tính của chúng tôi phân loại các điểm dữ liệu huấn luyện, như trong hình 3.2. Vì mục tiêu của chúng tôi là 0 và 1, nên một điểm đầu vào nhất định sẽ được phân loại là “0” nếu giá trị dự đoán của nó dưới 0,5 và là “1” nếu nó cao hơn 0,5:

```python
predictions = model(inputs, W, b)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
```

![](../images/ch03/linear_model_predictions.3e5424ac.png)

[Figure 3.2](#figure-3-2): Our model’s predictions on the training inputs: pretty similar to the training targets

Hãy nhớ lại rằng giá trị dự đoán cho một điểm nhất định `[x, y]` chỉ đơn giản là `prediction == [[w1], [w2]] • [x, y] + b == w1 * x + w2 * y + b`. Do đó, lớp “0” được định nghĩa là `w1 * x + w2 * y + b < 0,5` và lớp “1” được định nghĩa là `w1 * x + w2 * y + b > 0,5`. Bạn sẽ nhận thấy rằng những gì bạn đang nhìn thực sự là phương trình của một đường thẳng trong mặt phẳng 2D: `w1 * x + w2 * y + b = 0,5`. Loại 1 nằm trên dòng; lớp 0 nằm dưới dòng. Bạn có thể quen nhìn các phương trình đường ở định dạng `y = a * x + b`; ở cùng định dạng, dòng của chúng ta trở thành `y = - w1 / w2 * x + (0,5 - b) / w2`.

Hãy vẽ đường này, như trong hình 3.3:

```python
# Generates 100 regularly spaced numbers between -1 and 4, which we
# will use to plot our line
x = np.linspace(-1, 4, 100)
# This is our line's equation.
y = -W[0] / W[1] * x + (0.5 - b) / W[1]
# Plots our line (`"-r"` means "plot it as a red line")
plt.plot(x, y, "-r")
# Plots our model's predictions on the same plot
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
```

![](../images/ch03/linear_model_with_plotted_line.fd88e7bc.png)

[Figure 3.3](#figure-3-3): Our model, visualized as a line

Đây thực sự là mục đích của một trình phân loại tuyến tính: tìm các tham số của một đường thẳng (hoặc, trong không gian nhiều chiều hơn, một siêu phẳng) phân tách gọn gàng hai lớp dữ liệu.

### Điều gì làm cho phương pháp tiếp cận TensorFlow trở nên độc đáo

Giờ đây, bạn đã quen với tất cả các API cơ bản làm nền tảng cho quy trình công việc dựa trên TensorFlow và bạn sắp tìm hiểu sâu hơn về nhiều khung công tác hơn — đặc biệt là PyTorch và JAX. Điều gì khiến làm việc với TensorFlow khác với làm việc với bất kỳ framework nào khác? Khi nào bạn nên sử dụng TensorFlow và khi nào bạn có thể sử dụng thứ khác?

Nếu bạn hỏi chúng tôi thì đây là những lợi ích chính của TensorFlow:

* Nhờ chế độ biểu đồ và biên dịch XLA, nó rất nhanh. Nó thường nhanh hơn đáng kể so với PyTorch và NumPy, mặc dù JAX thậm chí còn nhanh hơn.
* Nó cực kỳ đầy đủ tính năng. Duy nhất trong số tất cả các khung, nó có hỗ trợ cho các tensor chuỗi cũng như “tensor rách rưới” (tensor trong đó các mục khác nhau
có thể có các kích thước khác nhau - rất hữu ích để xử lý các chuỗi mà không cần phải đệm chúng theo chiều dài chung). Nó cũng có sự hỗ trợ vượt trội cho dữ liệu
tiền xử lý, thông qua API `tf.data` hiệu suất cao. `tf.data` tốt đến mức ngay cả JAX cũng khuyên dùng nó để xử lý trước dữ liệu.
Dù bạn cần làm gì, TensorFlow đều có giải pháp cho việc đó.
* Hệ sinh thái triển khai sản xuất của nó là hoàn thiện nhất trong số tất cả các khung, đặc biệt là khi triển khai trên thiết bị di động hoặc trên trình duyệt.

Tuy nhiên, TensorFlow cũng có một số sai sót đáng chú ý:

* Nó có một API rộng lớn - mặt trái của nó là rất đầy đủ tính năng. TensorFlow bao gồm hàng nghìn thao tác khác nhau.
* API số của nó đôi khi không nhất quán với API NumPy, khiến việc tiếp cận khó hơn một chút nếu bạn đã quen với NumPy.
* Nền tảng chia sẻ mô hình được đào tạo trước phổ biến Ôm mặt có ít hỗ trợ hơn cho TensorFlow, điều đó có nghĩa là
các mô hình AI thế hệ mới nhất có thể không phải lúc nào cũng có sẵn trong TensorFlow.

Bây giờ, hãy chuyển sang PyTorch.

## Giới thiệu về PyTorch

PyTorch là một khung máy học nguồn mở dựa trên Python được phát triển chủ yếu bởi Meta (trước đây là Facebook). Nó được phát hành lần đầu vào tháng 9 năm 2016 (như một phản hồi cho việc phát hành TensorFlow), với phiên bản 1.0 ra mắt vào năm 2018 và phiên bản 2.0 ra mắt vào năm 2023. PyTorch kế thừa phong cách lập trình từ khung Chainer hiện không còn tồn tại, được lấy cảm hứng từ NumPy-Autograd. PyTorch được sử dụng rộng rãi trong cộng đồng nghiên cứu máy học.

Giống như TensorFlow, PyTorch là trung tâm của một hệ sinh thái lớn gồm các gói liên quan, chẳng hạn như `torchvision`, `torchaudio` hoặc nền tảng chia sẻ mô hình phổ biến Hugging Face.

API PyTorch ở cấp độ cao hơn TensorFlow và JAX: nó bao gồm các lớp và trình tối ưu hóa, như Keras. Các lớp và trình tối ưu hóa này tương thích với quy trình làm việc của Keras khi bạn sử dụng Keras với phần phụ trợ PyTorch.

### Những bước đầu tiên với PyTorch

Trong các đoạn tiếp theo, bạn sẽ làm quen với tất cả những điều cơ bản về PyTorch. Chúng tôi sẽ đề cập đến các khái niệm chính sau:

* Tenxơ và các thông số
* Các phép toán số trong PyTorch
* Tính toán độ dốc bằng phương thức `backward()`
* Tính toán đóng gói với lớp `Module`
* Tăng tốc PyTorch bằng cách sử dụng trình biên dịch

Chúng tôi sẽ kết thúc phần giới thiệu bằng cách triển khai lại ví dụ hồi quy tuyến tính từ đầu đến cuối trong PyTorch thuần túy.

#### Tensors và tham số trong PyTorch

Điều đáng chú ý đầu tiên về PyTorch là gói này không có tên là `pytorch`. Thực ra nó có tên là 'ngọn đuốc`. Bạn sẽ cài đặt nó qua `pip install torch` và bạn sẽ nhập nó qua `import torch`.

Giống như trong NumPy và TensorFlow, đối tượng trung tâm của framework là tensor. Trước tiên, chúng ta hãy sử dụng một số tensor PyTorch.

##### Các tensor không đổi

Dưới đây là một số tensor không đổi.

```python
>>> import torch
>>> # Unlike in other frameworks, the shape argument is named "size"
>>> # rather than "shape."
>>> torch.ones(size=(2, 1))
tensor([[1.], [1.]])
>>> torch.zeros(size=(2, 1))
tensor([[0.], [0.]])
>>> # Unlike in other frameworks, you cannot pass dtype="float32" as a
>>> # string. The dtype argument must be a torch dtype instance.
>>> torch.tensor([1, 2, 3], dtype=torch.float32)
tensor([1., 2., 3.])
```

[Liệt kê 3.22](#listing-3-22): Tenxơ toàn số một hoặc toàn số không

##### Các tensor ngẫu nhiên

Việc tạo tensor ngẫu nhiên tương tự như NumPy và TensorFlow, nhưng có cú pháp khác nhau. Hãy xem xét hàm `bình thường`: nó không có đối số hình dạng. Thay vào đó, giá trị trung bình và độ lệch chuẩn phải được cung cấp dưới dạng thang đo PyTorch với hình dạng đầu ra dự kiến.

```python
>>> # Equivalent to tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
>>> torch.normal(
... mean=torch.zeros(size=(3, 1)),
... std=torch.ones(size=(3, 1)))
tensor([[-0.9613],
        [-2.0169],
        [ 0.2088]])
```

[Liệt kê 3.23](#listing-3-23): Các tensor ngẫu nhiên

Đối với việc tạo một tenxơ đồng nhất ngẫu nhiên, bạn sẽ thực hiện điều đó thông qua `torch.rand`. Không giống như `np.random.uniform` hoặc `tf.random.uniform`, hình dạng đầu ra phải được cung cấp dưới dạng đối số độc lập cho từng thứ nguyên, như sau:

```python
>>> # Equivalent to tf.random.uniform(shape=(3, 1), minval=0.,
>>> # maxval=1.)
>>> torch.rand(3, 1)
```

##### Phép gán tensor và lớp Parameter

Giống như mảng NumPy, nhưng không giống như tensor TensorFlow, tensor PyTorch có thể gán được. Bạn có thể thực hiện các thao tác như thế này:

```python
>>> x = torch.zeros(size=(2, 1))
>>> x[0, 0] = 1.
>>> x
tensor([[1.],
        [0.]])
```

Mặc dù bạn chỉ có thể sử dụng `torch.Tensor` thông thường để lưu trữ trạng thái có thể huấn luyện của một mô hình, PyTorch cung cấp một lớp con tensor chuyên biệt cho mục đích đó, lớp `torch.nn.parameter.Parameter`. So với một tenxơ thông thường, nó cung cấp sự rõ ràng về mặt ngữ nghĩa — nếu bạn nhìn thấy `Tham số`, bạn sẽ biết đó là một phần của trạng thái có thể huấn luyện được, trong khi `Tensor` có thể là bất cứ thứ gì. Do đó, nó cho phép PyTorch tự động theo dõi và truy xuất `Tham số` mà bạn gán cho các mô hình PyTorch — tương tự như những gì Keras thực hiện với các phiên bản `Variable` của Keras.

Đây là một `Tham số`.

```python
>>> x = torch.zeros(size=(2, 1))
>>> # A Parameter can only be created using a torch.Tensor value — no
>>> # NumPy arrays allowed.
>>> p = torch.nn.parameter.Parameter(data=x)
```

[Liệt kê 3.24](#listing-3-24): Tạo tham số PyTorch

#### Hoạt động của tensor: Làm toán trong PyTorch

Toán học trong PyTorch hoạt động giống như toán học trong NumPy hoặc TensorFlow, mặc dù giống như TensorFlow, API PyTorch thường khác biệt với API NumPy theo những cách tinh tế.

```python
a = torch.ones((2, 2))
# Takes the square, same as np.square
b = torch.square(a)
# Takes the square root, same as np.sqrt
c = torch.sqrt(a)
# Adds two tensors (element-wise)
d = b + c
# Takes the product of two tensors (see chapter 2), same as np.matmul
e = torch.matmul(a, b)
# Concatenates a and b along axis 0, same as np.concatenate
f = torch.cat((a, b), dim=0)
```

[Liệt kê 3.25](#listing-3-25): Một vài phép toán cơ bản trong PyTorch

Đây là một lớp dày đặc:

```python
def dense(inputs, W, b):
    return torch.nn.relu(torch.matmul(inputs, W) + b)
```

#### Tính toán độ dốc với PyTorch

Không có “băng chuyển màu” rõ ràng trong PyTorch. Tồn tại một cơ chế tương tự: khi bạn chạy bất kỳ tính toán nào trong PyTorch, khung này sẽ tạo biểu đồ tính toán một lần (“băng”) ghi lại những gì vừa xảy ra. Tuy nhiên, đoạn băng đó bị ẩn khỏi người dùng. API công khai để sử dụng nó ở cấp độ tensor: bạn có thể gọi `tensor.backward()` để chạy lan truyền ngược thông qua tất cả các hoạt động được thực hiện trước đó dẫn đến tensor đó. Làm như vậy sẽ điền thuộc tính `.grad` của tất cả các tensor đang theo dõi độ dốc.

```python
>>> # To compute gradients with respect to a tensor, it must be created
>>> # with requires_grad=True.
>>> input_var = torch.tensor(3.0, requires_grad=True)
>>> result = torch.square(input_var)
>>> # Calling backward() populates the "grad" attribute on all tensors
>>> # create with requires_grad=True.
>>> result.backward()
>>> gradient = input_var.grad
>>> gradient
tensor(6.)
```

[Liệt kê 3.26](#listing-3-26): Tính toán một gradient với `.backward()`

Nếu bạn gọi `backward()` nhiều lần liên tiếp, thuộc tính `.grad` sẽ "tích lũy" độ dốc: mỗi lệnh gọi mới sẽ tính tổng độ dốc mới với độ dốc có sẵn. Ví dụ: trong đoạn mã sau, `input_var.grad` không phải là gradient của `square(input_var)` đối với `input_var`; đúng hơn, nó là tổng của gradient đó và gradient được tính toán trước đó — giá trị của nó đã tăng gấp đôi kể từ đoạn mã cuối cùng của chúng tôi:

```python
>>> result = torch.square(input_var)
>>> result.backward()
>>> # .grad will sum all gradient values from each time backward() is
>>> # called.
>>> input_var.grad
tensor(12.)
```

Để đặt lại độ dốc, bạn chỉ cần đặt `.grad` thành `None`:

```python
>>> input_var.grad = None
```

Bây giờ chúng ta hãy áp dụng điều này vào thực tế!

### Ví dụ toàn diện: Trình phân loại tuyến tính trong PyTorch thuần túy

Bây giờ bạn đã biết đủ để viết lại trình phân loại tuyến tính của chúng tôi trong PyTorch. Nó sẽ rất giống với TensorFlow - điểm khác biệt lớn duy nhất là cách chúng tôi tính toán độ dốc.

Hãy bắt đầu bằng cách tạo các biến mô hình của chúng tôi. Đừng quên truyền `requires_grad=True` để chúng ta có thể tính toán độ dốc tương ứng với chúng:

```python
input_dim = 2
output_dim = 1

W = torch.rand(input_dim, output_dim, requires_grad=True)
b = torch.zeros(output_dim, requires_grad=True)
```

Đây là mô hình của chúng tôi - cho đến nay không có gì khác biệt. Chúng ta vừa chuyển từ `tf.matmul` sang `torch.matmul`:

```python
def model(inputs, W, b):
    return torch.matmul(inputs, W) + b
```

Đây là chức năng mất mát của chúng tôi. Chúng ta chỉ cần chuyển từ `tf.square` sang `torch.square` và từ `tf.reduce_mean` sang `torch.mean`:

```python
def mean_squared_error(targets, predictions):
    per_sample_losses = torch.square(targets - predictions)
    return torch.mean(per_sample_losses)
```

Bây giờ đến bước đào tạo. Đây là cách nó hoạt động:

1. `loss.backward()` chạy lan truyền ngược bắt đầu từ nút đầu ra `loss` và điền thuộc tính `tensor.grad` trên tất cả các tensor có liên quan đến tính toán `loss`.    `tensor.grad` biểu thị độ dốc của tổn thất đối với tensor đó. 2. Chúng tôi sử dụng thuộc tính `.grad` để khôi phục độ dốc bị mất đối với `W` và `b`. 3. Chúng tôi cập nhật `W` và `b` bằng cách sử dụng các chuyển màu đó. Vì những cập nhật này không nhằm mục đích trở thành một phần của quá trình chuyển ngược nên chúng tôi thực hiện chúng trong phạm vi `torch.no_grad()`, bỏ qua tính toán độ dốc cho mọi thứ bên trong nó. 4. Chúng tôi đặt lại nội dung của thuộc tính `.grad` của các tham số `W` và `b` bằng cách đặt nó thành `None`.    Nếu chúng tôi không làm điều này, các giá trị gradient sẽ tích lũy qua nhiều lệnh gọi tới `training_step()`, dẫn đến các giá trị không hợp lệ:

```python
learning_rate = 0.1

def training_step(inputs, targets, W, b):
    # Forward pass
    predictions = model(inputs)
    loss = mean_squared_error(targets, predictions)
    # Computes gradients
    loss.backward()
    # Retrieves gradients
    grad_loss_wrt_W, grad_loss_wrt_b = W.grad, b.grad
    with torch.no_grad():
        # Updates weights inside a no_grad scope
        W -= grad_loss_wrt_W * learning_rate
        b -= grad_loss_wrt_b * learning_rate
    # Resets gradients
    W.grad = None
    b.grad = None
    return loss
```

Điều này thậm chí có thể được thực hiện đơn giản hơn - hãy xem cách thực hiện.

#### Trạng thái đóng gói và tính toán với lớp Module

PyTorch cũng có API hướng đối tượng, cấp độ cao hơn để thực hiện lan truyền ngược, yêu cầu dựa vào hai lớp mới: lớp `torch.nn.Module` và một lớp trình tối ưu hóa từ mô-đun `torch.optim`, chẳng hạn như `torch.optim.SGD` (tương đương với `keras.optimizers.SGD`).

Ý tưởng chung là định nghĩa một lớp con của `torch.nn.Module`, lớp này sẽ

* Giữ một số `Thông số` để lưu trữ các biến trạng thái. Chúng được định nghĩa trong phương thức `__init__()`.
* Thực hiện tính toán chuyển tiếp trong phương thức `forward()`.

Nó sẽ trông giống như sau.

```python
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.rand(input_dim, output_dim))
        self.b = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        return torch.matmul(inputs, self.W) + self.b
```

[Liệt kê 3.27](#listing-3-27): Xác định một `torch.nn.Module`

Bây giờ chúng ta có thể khởi tạo `LinearModel` của mình:

```python
model = LinearModel()
```

Khi sử dụng một phiên bản của `torch.nn.Module`, thay vì gọi trực tiếp phương thức `forward()`, bạn sẽ sử dụng `__call__()` (tức là gọi trực tiếp lớp mô hình trên đầu vào), chuyển hướng đến `forward()` nhưng thêm một vài framework hook vào nó:

```python
torch_inputs = torch.tensor(inputs)
output = model(torch_inputs)
```

Bây giờ, hãy bắt tay vào sử dụng trình tối ưu hóa PyTorch. Để khởi tạo nó, bạn sẽ cần cung cấp danh sách các tham số mà trình tối ưu hóa dự định cập nhật. Bạn có thể truy xuất nó từ phiên bản `Module` của chúng tôi thông qua `.parameters()`:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Bằng cách sử dụng phiên bản `Module` và trình tối ưu hóa PyTorch `SGD`, chúng ta có thể chạy một bước đào tạo đơn giản hóa:

```python
def training_step(inputs, targets):
    predictions = model(inputs)
    loss = mean_squared_error(targets, predictions)
    loss.backward()
    optimizer.step()
    model.zero_grad()
    return loss
```

Trước đây, việc cập nhật các tham số mô hình trông như thế này:

```python
with torch.no_grad():
    W -= grad_loss_wrt_W * learning_rate
    b -= grad_loss_wrt_b * learning_rate
```

Bây giờ chúng ta chỉ cần thực hiện `optimizer.step()`.

Tương tự, trước đây chúng ta cần đặt lại gradient tham số bằng tay bằng cách thực hiện `tensor.grad = None` trên mỗi tham số. Bây giờ chúng ta chỉ cần thực hiện `model.zero_grad()`.

Nhìn chung, điều này có thể hơi khó hiểu — bằng cách nào đó, tensor mất mát, trình tối ưu hóa và phiên bản `Module` dường như đều nhận biết được nhau thông qua một số cơ chế nền ẩn. Tất cả họ đều tương tác với nhau thông qua hành động ma quái ở khoảng cách xa. Tuy nhiên, đừng lo lắng — bạn chỉ có thể coi chuỗi các bước này (`loss.backward()` - `optimizer.step()` - `model.zero_grad()`) như một câu thần chú để đọc bất cứ khi nào bạn cần viết hàm bước huấn luyện. Chỉ cần đảm bảo không quên `model.zero_grad()`. Đó sẽ là một lỗi lớn (và tiếc là nó khá phổ biến)!

#### Làm cho các mô-đun PyTorch nhanh chóng bằng cách sử dụng trình biên dịch

Một điều cuối cùng. Tương tự như cách TensorFlow cho phép bạn biên dịch các hàm để có hiệu suất tốt hơn, PyTorch cho phép bạn biên dịch các hàm hoặc thậm chí các phiên bản `Module` thông qua tiện ích `torch.compile()`. API này sử dụng trình biên dịch rất riêng của PyTorch, có tên là Dynamo.

Hãy thử hồi quy tuyến tính `Module` của chúng tôi:

```python
compiled_model = torch.compile(model)
```

Đối tượng kết quả được thiết kế để hoạt động giống hệt với đối tượng ban đầu — ngoại trừ việc chuyển tiếp và lùi sẽ chạy nhanh hơn.

Bạn cũng có thể sử dụng `torch.compile()` làm công cụ trang trí hàm:

```python
@torch.compile
def dense(inputs, W, b):
    return torch.nn.relu(torch.matmul(inputs, W) + b)
```

Trong thực tế, hầu hết mã PyTorch hiện không sử dụng trình biên dịch và chỉ chạy một cách háo hức, vì trình biên dịch có thể không phải lúc nào cũng hoạt động với tất cả các mô hình và không phải lúc nào cũng có thể tăng tốc khi nó hoạt động. Không giống như trong TensorFlow và Jax, nơi trình biên dịch được tích hợp sẵn từ khi thành lập thư viện, trình biên dịch của PyTorch là một bổ sung tương đối gần đây.

### Điều gì làm cho phương pháp tiếp cận PyTorch trở nên độc đáo

So với TensorFlow và JAX mà chúng tôi sẽ đề cập tiếp theo, điều gì khiến PyTorch nổi bật? Tại sao bạn nên sử dụng nó hoặc không sử dụng nó?

Dưới đây là hai điểm mạnh chính của PyTorch:

* Mã PyTorch thực thi mạnh mẽ theo mặc định, giúp dễ dàng gỡ lỗi.
Lưu ý rằng đây cũng là trường hợp của mã TensorFlow và mã JAX, nhưng có một điểm khác biệt lớn là PyTorch thường được thiết kế để
luôn chạy một cách háo hức, trong khi bất kỳ dự án TensorFlow hoặc JAX nghiêm túc nào chắc chắn sẽ cần được biên dịch tại một số điểm, điều này có thể ảnh hưởng đáng kể đến trải nghiệm gỡ lỗi.
* Nền tảng chia sẻ mô hình được đào tạo trước phổ biến Ôm mặt có hỗ trợ hạng nhất cho PyTorch, nền tảng này
có nghĩa là bất kỳ mô hình nào bạn muốn sử dụng đều có sẵn trong PyTorch.
Đây là động lực chính thúc đẩy việc áp dụng PyTorch ngày nay.

Đồng thời, cũng có một số nhược điểm khi sử dụng PyTorch:

* Giống như TensorFlow, API PyTorch không nhất quán với NumPy. Hơn nữa, nó cũng không nhất quán trong nội bộ. Ví dụ: từ khóa thường được sử dụng `axis` đôi khi được đặt tên là `dim`, tùy thuộc vào chức năng.
Một số thao tác tạo số giả ngẫu nhiên lấy đối số `seed`; những người khác thì không. Và vân vân.
Điều này có thể khiến PyTorch nản lòng khi học, đặc biệt khi đến từ NumPy.
* Do tập trung vào việc thực thi háo hức, PyTorch khá chậm - chậm nhất
của tất cả các khuôn khổ chính với một biên độ lớn. Đối với hầu hết các kiểu máy, bạn có thể thấy tốc độ tăng 20% ​​hoặc 30% với JAX.
Đối với một số kiểu máy — đặc biệt là những kiểu máy lớn — bạn thậm chí có thể thấy tốc độ tăng lên 3× hoặc 5× với JAX, ngay cả sau khi sử dụng `torch.compile()`.
* Mặc dù có thể làm cho mã PyTorch nhanh hơn thông qua `torch.compile()`, trình biên dịch PyTorch Dynamo
đến thời điểm này (năm 2025) vẫn khá kém hiệu quả và đầy cửa sập. Kết quả là chỉ có một tỷ lệ rất nhỏ trong số
Cơ sở người dùng PyTorch sử dụng trình biên dịch. Có lẽ điều này sẽ được cải thiện trong các phiên bản sau!

## Giới thiệu về JAX

JAX là một thư viện mã nguồn mở dành cho tính toán vi phân, chủ yếu được phát triển bởi Google. Sau khi phát hành vào năm 2018, JAX nhanh chóng thu hút được sự chú ý trong cộng đồng nghiên cứu, đặc biệt là nhờ khả năng sử dụng TPU của Google trên quy mô lớn. Ngày nay, JAX được hầu hết những công ty hàng đầu trong lĩnh vực AI tổng hợp sử dụng - các công ty như DeepMind, Apple, Midjourney, Anthropic, Cohere, v.v.

JAX áp dụng cách tiếp cận *không trạng thái* để tính toán, nghĩa là các hàm trong JAX không duy trì bất kỳ trạng thái liên tục nào. Điều này trái ngược với lập trình mệnh lệnh truyền thống, trong đó các biến có thể giữ giá trị giữa các lệnh gọi hàm.

Bản chất không trạng thái của các hàm JAX có một số ưu điểm. Đặc biệt, nó cho phép tính toán phân tán và song song hóa tự động hiệu quả, vì các chức năng có thể được thực thi độc lập mà không cần đồng bộ hóa. Khả năng mở rộng tối đa của JAX là điều cần thiết để xử lý các vấn đề về máy học ở quy mô rất lớn mà các công ty như Google và DeepMind gặp phải.

### Những bước đầu tiên với JAX

Chúng ta sẽ xem xét các khái niệm chính sau:

* Lớp `mảng`
* Hoạt động ngẫu nhiên trong JAX
* Các phép toán số trong JAX
* Tính toán độ dốc thông qua `jax.grad` và `jax.value_and_grad`
* Làm cho các hàm JAX trở nên nhanh chóng bằng cách tận dụng tính năng biên dịch đúng lúc

Hãy bắt đầu.

### Tenxơ trong JAX

Một trong những tính năng tốt nhất của JAX là nó không cố gắng triển khai API số độc lập, tương tự như NumPy nhưng hơi khác nhau của riêng nó. Thay vào đó, nó chỉ triển khai API NumPy. Nó có sẵn dưới dạng không gian tên `jax.numpy` và bạn sẽ thường thấy nó được nhập dưới dạng viết tắt là `jnp`.

Dưới đây là một số mảng JAX.

```python
>>> from jax import numpy as jnp
>>> jnp.ones(shape=(2, 1))
Array([[1.],
       [1.]], dtype=float32)
>>> jnp.zeros(shape=(2, 1))
Array([[0.],
       [0.]], dtype=float32)
>>> jnp.array([1, 2, 3], dtype="float32")
Array([1., 2., 3.], dtype=float32)
```

[Liệt kê 3.28](#listing-3-28): Tenxơ toàn số một hoặc toàn số không

Tuy nhiên, có hai điểm khác biệt nhỏ giữa `jax.numpy` và API NumPy thực tế: tạo số ngẫu nhiên và gán mảng. Chúng ta hãy xem xét.

### Tạo số ngẫu nhiên trong JAX

Sự khác biệt đầu tiên giữa JAX và NumPy liên quan đến cách JAX xử lý các hoạt động ngẫu nhiên - cái được gọi là các hoạt động “PRNG” (Tạo số giả ngẫu nhiên). Chúng tôi đã nói trước đó rằng JAX là *không trạng thái*, ngụ ý rằng mã JAX không thể dựa vào bất kỳ trạng thái toàn cầu ẩn nào. Hãy xem xét mã NumPy sau đây.

```python
>>> np.random.normal(size=(3,))
array([-1.68856166,  0.16489586,  0.67707523])
>>> np.random.normal(size=(3,))
array([-0.73671259,  0.3053194 ,  0.84124895])
```

[Liệt kê 3.29](#listing-3-29): Các tensor ngẫu nhiên

Làm thế nào mà lệnh gọi thứ hai tới `np.random.normal()` biết trả về một giá trị khác với lệnh gọi đầu tiên? Đúng vậy - đó là một phần ẩn của trạng thái toàn cầu. Bạn thực sự có thể truy xuất trạng thái toàn cục đó thông qua `np.random.get_state()` và thiết lập nó thông qua `np.random.seed(seed)`.

Trong khuôn khổ không quốc tịch, chúng ta không thể có bất kỳ trạng thái toàn cầu nào như vậy. Lệnh gọi API giống nhau phải luôn trả về cùng một giá trị. Do đó, trong phiên bản NumPy không trạng thái, bạn sẽ phải dựa vào việc chuyển các đối số gốc khác nhau cho lệnh gọi `np.random` của mình để nhận được các giá trị khác nhau.

Hiện nay, thường xảy ra trường hợp các lệnh gọi PRNG của bạn sẽ nằm trong các hàm được gọi nhiều lần và nhằm mục đích sử dụng các giá trị ngẫu nhiên khác nhau mỗi lần. Nếu bạn không muốn dựa vào bất kỳ trạng thái toàn cầu nào, điều này yêu cầu bạn phải quản lý trạng thái hạt giống của mình bên ngoài hàm mục tiêu, như sau:

```python
def apply_noise(x, seed):
    np.random.seed(seed)
    x = x * np.random.normal((3,))
    return x

seed = 1337
y = apply_noise(x, seed)
seed += 1
z = apply_noise(x, seed)
```

Về cơ bản nó giống nhau trong JAX. Tuy nhiên, JAX không sử dụng hạt giống số nguyên. Nó sử dụng các cấu trúc mảng đặc biệt gọi là *keys*. Bạn có thể tạo một giá trị từ một giá trị số nguyên, như thế này:

```python
import jax

seed_key = jax.random.key(1337)
```

Để buộc bạn phải luôn cung cấp “khóa” hạt giống cho các cuộc gọi PRNG, tất cả các hoạt động sử dụng JAX PRNG đều lấy `key` (hạt giống ngẫu nhiên) làm đối số vị trí đầu tiên của chúng. Đây là cách sử dụng `random.normal()`:

```python
>>> seed_key = jax.random.key(0)
>>> jax.random.normal(seed_key, shape=(3,))
Array([ 1.8160863 , -0.48262316,  0.33988908], dtype=float32)
```

Hai lệnh gọi tới `random.normal()` nhận được cùng một khóa hạt giống sẽ luôn trả về cùng một giá trị.

```python
>>> seed_key = jax.random.key(123)
>>> jax.random.normal(seed_key, shape=(3,))
Array([-0.1470326,  0.5524756,  1.648498 ], dtype=float32)
>>> jax.random.normal(seed_key, shape=(3,))
Array([-0.1470326,  0.5524756,  1.648498 ], dtype=float32)
```

[Danh sách 3.30](#listing-3-30): Sử dụng hạt giống ngẫu nhiên trong Jax

Nếu cần khóa hạt giống mới, bạn chỉ cần tạo một khóa mới từ khóa hiện có bằng cách sử dụng hàm `jax.random.split()`. Nó mang tính quyết định, do đó, trình tự phân tách giống nhau sẽ luôn dẫn đến cùng một khóa hạt giống cuối cùng:

```python
>>> seed_key = jax.random.key(123)
>>> jax.random.normal(seed_key, shape=(3,))
Array([-0.1470326,  0.5524756,  1.648498 ], dtype=float32)
>>> # You could even split your key into multiple new keys at once!
>>> new_seed_key = jax.random.split(seed_key, num=1)[0]
>>> jax.random.normal(new_seed_key, shape=(3,))
Array([ 0.5362355, -1.1920372,  2.450225 ], dtype=float32)
```

Đây chắc chắn là công việc nhiều hơn `np.random`! Nhưng lợi ích của việc không trạng thái vượt xa chi phí: nó làm cho mã của bạn *có thể vector hóa* (tức là trình biên dịch JAX có thể tự động biến nó thành mã song song cao) trong khi vẫn duy trì tính xác định (tức là bạn có thể chạy cùng một mã hai lần với cùng một kết quả). Điều đó là không thể đạt được với trạng thái PRNG toàn cầu.

#### Phép gán tensor

Sự khác biệt thứ hai giữa JAX và NumPy là phép gán tensor. Giống như trong TensorFlow, mảng JAX không thể gán được tại chỗ. Đó là bởi vì bất kỳ loại sửa đổi tại chỗ nào cũng sẽ đi ngược lại thiết kế không trạng thái của JAX. Thay vào đó, nếu bạn cần cập nhật một tensor, bạn phải tạo một tensor mới với giá trị mong muốn. JAX làm cho việc này trở nên dễ dàng bằng cách cung cấp API `at()`/`set()`. Các phương pháp này cho phép bạn tạo một tenxơ mới với phần tử được cập nhật tại một chỉ mục cụ thể. Dưới đây là ví dụ về cách bạn cập nhật phần tử đầu tiên của mảng JAX thành giá trị mới.

```python
>>> x = jnp.array([1, 2, 3], dtype="float32")
>>> new_x = x.at[0].set(10)
```

[Liệt kê 3.31](#listing-3-31): Sửa đổi các giá trị trong một mảng JAX

Đủ đơn giản!

#### Hoạt động của tensor: Làm toán trong JAX

Làm toán trong JAX trông giống hệt như trong NumPy. Không cần phải học bất cứ điều gì mới lần này!

```python
a = jnp.ones((2, 2))
# Takes the square
b = jnp.square(a)
# Takes the square root
c = jnp.sqrt(a)
# Adds two tensors (element-wise)
d = b + c
# Takes the product of two tensors (see chapter 2)
e = jnp.matmul(a, b)
# Multiplies two tensors (element-wise)
e *= d
```

[Liệt kê 3.32](#listing-3-32): Một vài phép toán cơ bản trong JAX

Đây là một lớp dày đặc:

```python
def dense(inputs, W, b):
    return jax.nn.relu(jnp.matmul(inputs, W) + b)
```

#### Tính toán độ dốc với JAX

Không giống như TensorFlow và PyTorch, JAX sử dụng phương pháp *siêu lập trình* để tính toán độ dốc. Siêu lập trình đề cập đến ý tưởng có *hàm trả về hàm* — bạn có thể gọi chúng là “siêu hàm”. Trong thực tế, JAX cho phép bạn *biến hàm tính toán tổn thất thành hàm tính toán độ dốc*. Vì vậy, tính toán độ dốc trong JAX là một quá trình gồm ba bước:

1. Xác định hàm mất mát, `compute_loss()`. 2. Gọi `grad_fn = jax.grad(compute_loss)` để truy xuất hàm tính toán độ dốc. 3. Gọi `grad_fn` để lấy các giá trị gradient.

Hàm mất sẽ xác minh các thuộc tính sau:

* Nó sẽ trả về một giá trị tổn thất vô hướng.
* Đối số đầu tiên của nó (trong ví dụ sau đây cũng là đối số duy nhất) phải chứa các mảng trạng thái mà chúng ta cần gradient.
Đối số này thường được đặt tên là `state`. Chẳng hạn, đối số đầu tiên này có thể là một mảng đơn, một danh sách các mảng hoặc một mệnh đề mảng.

Chúng ta hãy xem một ví dụ đơn giản. Đây là hàm tính toán tổn thất nhận một giá trị vô hướng duy nhất, `input_var` và trả về giá trị tổn thất vô hướng — chỉ là bình phương của đầu vào:

```python
def compute_loss(input_var):
    return jnp.square(input_var)
```

Bây giờ chúng ta có thể gọi tiện ích JAX `jax.grad()` trên hàm mất mát này. Nó trả về một hàm tính toán độ dốc - một hàm lấy các đối số giống như hàm mất mát ban đầu và trả về độ dốc của tổn thất đối với `input_var`:

```python
grad_fn = jax.grad(compute_loss)
```

Khi bạn đã nhận được `grad_fn()`, bạn có thể gọi nó với các đối số tương tự như `compute_loss()` và nó sẽ trả về các mảng gradient tương ứng với đối số đầu tiên của `compute_loss()`. Trong trường hợp của chúng tôi, đối số đầu tiên của chúng tôi là một mảng duy nhất, do đó, `grad_fn()` trực tiếp trả về độ dốc của phần mất đối với một mảng đó:

```python
input_var = jnp.array(3.0)
grad_of_loss_wrt_input_var = grad_fn(input_var)
```

#### Các phương pháp hay nhất về tính toán độ dốc JAX

Cho đến nay rất tốt! Lập trình meta là một từ lớn, nhưng hóa ra nó lại khá đơn giản. Bây giờ, trong các trường hợp sử dụng trong thế giới thực, bạn sẽ cần phải tính đến một số điều nữa. Chúng ta hãy xem xét.

##### Trả về giá trị tổn thất

Thông thường, bạn không chỉ cần mảng gradient; bạn cũng cần giá trị tổn thất. Sẽ khá kém hiệu quả nếu tính toán lại nó một cách độc lập bên ngoài `grad_fn()`, vì vậy thay vào đó, bạn chỉ có thể định cấu hình `grad_fn()` của mình để trả về giá trị tổn thất. Điều này được thực hiện bằng cách sử dụng tiện ích JAX `jax.value_and_grad()` thay vì `jax.grad()`. Nó hoạt động giống hệt nhau, nhưng nó trả về một bộ giá trị, trong đó mục nhập đầu tiên là giá trị tổn thất và mục nhập thứ hai là (các) độ dốc:

```python
grad_fn = jax.value_and_grad(compute_loss)
output, grad_of_loss_wrt_input_var = grad_fn(input_var)
```

##### Lấy gradient cho một hàm phức tạp

Bây giờ, nếu bạn cần độ dốc cho nhiều biến thì sao? Và điều gì sẽ xảy ra nếu hàm `compute_loss()` của bạn có nhiều hơn một đầu vào?

Giả sử trạng thái của bạn chứa ba biến, `a`, `b` và `c` và hàm mất mát của bạn có hai đầu vào, `x` và `y`. Bạn chỉ cần cấu trúc nó như thế này:

```python
# state contains a, b, and c. It must be the first argument.
def compute_loss(state, x, y):
    ...
    return loss

grad_fn = jax.value_and_grad(compute_loss)
state = (a, b, c)
# grads_of_loss_wrt_state has the same structure as state.
loss, grads_of_loss_wrt_state = grad_fn(state, x, y)
```

Lưu ý rằng `state` không nhất thiết phải là một bộ dữ liệu - nó có thể là một lệnh, một danh sách hoặc bất kỳ cấu trúc lồng nhau nào của các bộ dữ liệu, lệnh và danh sách. Theo cách nói của JAX, cấu trúc lồng nhau như vậy được gọi là *cây*.

##### Trả về đầu ra phụ trợ

Cuối cùng, điều gì sẽ xảy ra nếu hàm `compute_loss()` của bạn cần trả về nhiều hơn là chỉ phần mất? Giả sử bạn muốn trả về một giá trị bổ sung `đầu ra` được tính là sản phẩm phụ của quá trình tính toán tổn thất. Làm thế nào để lấy nó ra?

Bạn sẽ sử dụng đối số `has_aux`:

1. Chỉnh sửa hàm mất để trả về một bộ dữ liệu trong đó mục nhập đầu tiên là phần mất và mục nhập thứ hai là đầu ra bổ sung của bạn. 2. Truyền đối số `has_aux=True` cho `value_and_grad()`. Điều này yêu cầu `value_and_grad()` trả về không chỉ gradient mà còn cả (các) đầu ra “phụ trợ” của `compute_loss()`, như thế này:

```python
def compute_loss(state, x, y):
    ...
    # Returns a tuple
    return loss, output

# Passes has_aux=True here
grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
# Gets back a nested tuple
loss, (grads_of_loss_wrt_state, output) = grad_fn(state, x, y)
```

Phải thừa nhận rằng mọi thứ đang bắt đầu khá phức tạp vào thời điểm này. Tuy nhiên, đừng lo lắng; điều này khó khăn như JAX vậy! Hầu hết mọi thứ khác đều đơn giản hơn khi so sánh.

#### Tạo các hàm JAX nhanh chóng với @jax.jit

Một điều nữa. Là người dùng JAX, bạn sẽ thường xuyên sử dụng trình trang trí `@jax.jit`, hoạt động giống hệt với trình trang trí `@tf.function(jit_compile=True)`. Nó biến bất kỳ hàm JAX không trạng thái nào thành một đoạn mã do XLA biên dịch, thường mang lại tốc độ thực thi đáng kể:

```python
@jax.jit
def dense(inputs, W, b):
    return jax.nn.relu(jnp.matmul(inputs, W) + b)
```

Hãy lưu ý rằng bạn chỉ có thể trang trí một hàm không trạng thái - bất kỳ tensor nào được hàm cập nhật đều phải là một phần của giá trị trả về của hàm đó.

### Ví dụ toàn diện: Trình phân loại tuyến tính trong JAX thuần túy

Bây giờ bạn đã biết đủ JAX để viết phiên bản JAX của ví dụ về trình phân loại tuyến tính của chúng tôi. Có hai điểm khác biệt chính so với phiên bản TensorFlow và PyTorch mà bạn đã thấy:

* Tất cả các chức năng chúng tôi sẽ tạo sẽ *không trạng thái*. Điều đó có nghĩa là trạng thái (mảng `W` và `b`) sẽ được cung cấp
làm đối số của hàm và nếu chúng được hàm sửa đổi, giá trị mới của chúng sẽ được hàm trả về.
* Độ dốc được tính toán bằng tiện ích JAX `value_and_grad()`.

Hãy bắt đầu. Hàm mô hình và hàm lỗi bình phương trung bình trông quen thuộc:

```python
def model(inputs, W, b):
    return jnp.matmul(inputs, W) + b

def mean_squared_error(targets, predictions):
    per_sample_losses = jnp.square(targets - predictions)
    return jnp.mean(per_sample_losses)
```

Để tính toán độ dốc, chúng ta cần gói tính toán tổn thất trong một hàm `compute_loss()` duy nhất. Nó trả về tổng tổn thất dưới dạng vô hướng và lấy `state` làm đối số đầu tiên - một bộ gồm tất cả các tensor mà chúng ta cần gradient cho:

```python
def compute_loss(state, inputs, targets):
    W, b = state
    predictions = model(inputs, W, b)
    loss = mean_squared_error(targets, predictions)
    return loss
```

Việc gọi `jax.value_and_grad()` trên hàm này sẽ mang lại cho chúng ta một hàm mới, với cùng đối số như `compute_loss`, trả về cả phần mất mát và độ dốc của phần mất mát đối với các phần tử của `state`:

```python
grad_fn = jax.value_and_grad(compute_loss)
```

Tiếp theo, chúng ta có thể thiết lập chức năng bước đào tạo của mình. Nó có vẻ đơn giản. Hãy lưu ý rằng, không giống như các đối tượng tương đương TensorFlow và PyTorch của nó, nó cần phải ở trạng thái không trạng thái và do đó, nó phải trả về các giá trị được cập nhật của các tensor `W` và `b`:

```python
learning_rate = 0.1

# We use the jax.jit decorator to take advantage of XLA compilation.
@jax.jit
def training_step(inputs, targets, W, b):
    # Computes the forward pass and backward pass in one go
    loss, grads = grad_fn((W, b), inputs, targets)
    grad_wrt_W, grad_wrt_b = grads
    # Updates W and b
    W = W - grad_wrt_W * learning_rate
    b = b - grad_wrt_b * learning_rate
    # Make sure to return the new values of W and b in addition to the
    # loss!
    return loss, W, b
```

Vì chúng ta sẽ không thay đổi `learning_rate` trong ví dụ của mình nên chúng ta có thể coi đó là một phần của hàm chứ không phải trạng thái của mô hình. Nếu chúng tôi muốn sửa đổi tốc độ học tập của mình trong quá trình đào tạo, chúng tôi cũng cần phải thông qua nó.

Cuối cùng, chúng ta đã sẵn sàng chạy vòng đào tạo đầy đủ. Chúng tôi khởi tạo `W` và `b` và liên tục cập nhật chúng thông qua lệnh gọi không trạng thái tới `training_step()`:

```python
input_dim = 2
output_dim = 1

W = jax.numpy.array(np.random.uniform(size=(input_dim, output_dim)))
b = jax.numpy.array(np.zeros(shape=(output_dim,)))
state = (W, b)
for step in range(40):
    loss, W, b = training_step(inputs, targets, W, b)
    print(f"Loss at step {step}: {loss:.4f}")
```

Thế thôi! Bây giờ bạn có thể viết vòng đào tạo tùy chỉnh trong JAX.

### Điều gì làm cho phương pháp tiếp cận JAX trở nên độc đáo

Điều chính làm cho JAX trở nên độc đáo trong số các khung học máy hiện đại là triết lý phi trạng thái, chức năng của nó. Mặc dù ban đầu nó có vẻ gây ra xích mích, nhưng nó chính là thứ giúp giải phóng sức mạnh của JAX - khả năng biên dịch thành mã cực nhanh và mở rộng quy mô thành các mô hình lớn tùy ý và nhiều thiết bị tùy ý.

Có rất nhiều điều thích về JAX:

* Nó nhanh. Đối với hầu hết các mô hình, đây là khung nhanh nhất trong số tất cả các khung bạn từng thấy cho đến nay.
* API số của nó hoàn toàn phù hợp với NumPy, khiến việc học trở nên thú vị.
* Nó phù hợp nhất cho các mô hình đào tạo về TPU vì nó được phát triển từ đầu cho XLA và TPU.

Việc sử dụng JAX cũng có thể gây ra một số trở ngại cho nhà phát triển:

* Việc sử dụng siêu lập trình và biên dịch có thể khiến việc gỡ lỗi khó hơn đáng kể so với thực thi háo hức thuần túy.
* Các vòng đào tạo cấp thấp có xu hướng dài dòng hơn và khó viết hơn so với TensorFlow hoặc PyTorch.

Tại thời điểm này, bạn đã biết kiến ​​thức cơ bản về TensorFlow, PyTorch và JAX, đồng thời bạn có thể sử dụng các khung này để triển khai trình phân loại tuyến tính cơ bản từ đầu. Đó là một nền tảng vững chắc để xây dựng. Bây giờ là lúc chuyển sang một con đường hiệu quả hơn để học sâu: API Keras.

## Giới thiệu về Keras

Keras là API học sâu dành cho Python, cung cấp một cách thuận tiện để xác định và đào tạo bất kỳ loại mô hình học sâu nào. Nó được phát hành vào tháng 3 năm 2015, với v2 vào năm 2017 và v3 vào năm 2023.

Người dùng Keras bao gồm từ các nhà nghiên cứu hàn lâm, kỹ sư và nhà khoa học dữ liệu ở cả các công ty khởi nghiệp và công ty lớn cho đến sinh viên mới tốt nghiệp và những người có sở thích. Keras được sử dụng tại Google, Netflix, Uber, YouTube, CERN, NASA, Yelp, Instacart, Square, Waymo, YouTube và hàng nghìn tổ chức nhỏ hơn đang giải quyết nhiều vấn đề khác nhau trong mọi ngành. Đề xuất YouTube của bạn bắt nguồn từ mô hình Keras. Xe tự lái Waymo dựa vào mô hình Keras để xử lý dữ liệu cảm biến. Keras cũng là một framework phổ biến trên Kaggle, trang web cạnh tranh về học máy.

Bởi vì Keras có cơ sở người dùng đa dạng nên nó không buộc bạn phải tuân theo một cách xây dựng và đào tạo mô hình “đúng” duy nhất. Đúng hơn, nó cho phép thực hiện nhiều quy trình công việc khác nhau, từ cấp rất cao đến cấp rất thấp, tương ứng với các hồ sơ người dùng khác nhau. Ví dụ: bạn có nhiều cách để xây dựng mô hình và nhiều cách để đào tạo chúng, mỗi cách thể hiện sự cân bằng nhất định giữa khả năng sử dụng và tính linh hoạt. Trong chương 7, chúng ta sẽ xem xét chi tiết một phần lớn các quy trình làm việc này.

### Những bước đầu tiên với Keras

Trước khi chúng ta viết mã Keras, có một số điều cần cân nhắc khi thiết lập thư viện trước khi nhập.

#### Chọn một khung phụ trợ

Keras có thể được sử dụng cùng với JAX, TensorFlow hoặc PyTorch. Chúng là “khuôn khổ phụ trợ” của Keras. Thông qua các khung phụ trợ này, Keras có thể chạy trên nhiều loại phần cứng khác nhau (xem hình 3.4) — GPU, TPU hoặc CPU đơn giản — có thể được mở rộng quy mô liền mạch cho hàng nghìn máy và có thể được triển khai trên nhiều nền tảng khác nhau.

![](../images/ch03/keras_and_backends.7fcf768f.png)

[Figure 3.4](#figure-3-4): Keras and its backends. A backend is a low-level tensor-computing platform; Keras is a high-level deep learning API.

Các khung phụ trợ có thể cắm được: bạn có thể chuyển sang một khung phụ trợ khác *sau khi* bạn đã viết một số mã Keras. Bạn không bị bó buộc vào một khung duy nhất và một hệ sinh thái duy nhất — bạn có thể di chuyển các mô hình của mình từ JAX sang TensorFlow sang PyTorch tùy theo nhu cầu hiện tại của bạn. Ví dụ: khi phát triển mô hình Keras, bạn có thể gỡ lỗi nó bằng PyTorch, huấn luyện nó trên TPU bằng JAX để đạt hiệu quả tối đa và cuối cùng chạy suy luận bằng công cụ tuyệt vời từ hệ sinh thái TensorFlow.

Phần phụ trợ mặc định cho Keras hiện tại là TensorFlow, vì vậy nếu bạn chạy `nhập keras` trong một môi trường mới mà chưa định cấu hình bất cứ thứ gì, thì bạn sẽ chạy trên TensorFlow. Có hai cách để chọn một chương trình phụ trợ khác:

* Đặt biến môi trường `KERAS_BACKEND`. Trước khi bắt đầu thay thế `python`, bạn có thể
chạy lệnh shell sau để sử dụng JAX làm chương trình phụ trợ Keras của bạn: `export KERAS_BACKEND=jax`.
Ngoài ra, bạn có thể thêm đoạn mã sau vào đầu tệp hoặc sổ ghi chép Python của mình
(lưu ý rằng nó nhất thiết phải đi trước `máy ảnh nhập khẩu` đầu tiên):

```python
import os

# Sets the environment variable from within the Python runtime
os.environ["KERAS_BACKEND"] = "jax"

# Only then should you import Keras.
import keras
```

* Chỉnh sửa tệp cấu hình Keras cục bộ của bạn tại `~/.keras/keras.json`. Nếu bạn đã nhập
Keras một lần, tệp này đã được tạo với cài đặt mặc định.
Bạn có thể sử dụng bất kỳ trình soạn thảo văn bản nào để mở và sửa đổi nó - đó là tệp JSON mà con người có thể đọc được. Nó sẽ trông như thế này:

```python
{
    # Default floating-point precision. It should typically not be
    # changed.
    "floatx": "float32",
    # Default numerical fuzzing factor. It should typically not be
    # changed.
    "epsilon": 1e-07,
    # Change "tensorflow" to "jax" or "torch."
    "backend": "tensorflow",
    # This is the default image layout. We'll talk about this in
    # chapter 8.
    "image_data_format": "channels_last",
}
```



Khi định cấu hình phần phụ trợ Keras, bạn nên sử dụng chuỗi `"torch"` để chỉ phần phụ trợ PyTorch, thay vì chuỗi `"pytorch"`, chuỗi này sẽ không hợp lệ. Điều này là do tên gói PyTorch là `torch` (như trong `import torch` hoặc `pip install torch`).

Bây giờ, bạn có thể hỏi, tôi nên chọn phần phụ trợ nào? Đó thực sự là sự lựa chọn của riêng bạn: tất cả các ví dụ về mã Keras trong phần còn lại của cuốn sách sẽ tương thích với cả ba phần phụ trợ. Nếu phát sinh nhu cầu về mã dành riêng cho phần phụ trợ (chẳng hạn như trong chương 7), tôi sẽ hiển thị cho bạn cả ba phiên bản - TensorFlow, PyTorch, JAX. Nếu bạn không có tùy chọn phụ trợ cụ thể nào, đề xuất cá nhân của tôi là JAX. Nó thường là phần phụ trợ hiệu quả nhất.

Khi phần phụ trợ của bạn được định cấu hình, bạn có thể bắt đầu thực sự xây dựng và đào tạo các mô hình Keras. Chúng ta hãy xem xét.

### Lớp: Các khối xây dựng của học sâu

Cấu trúc dữ liệu cơ bản trong mạng thần kinh là *lớp* mà bạn đã được giới thiệu ở chương 2. Lớp là một mô-đun xử lý dữ liệu lấy một hoặc nhiều tensor làm đầu vào và xuất ra một hoặc nhiều tensor. Một số lớp không có trạng thái, nhưng các lớp thường xuyên hơn có một trạng thái: *trọng lượng* của lớp, một hoặc một số tensor được học với độ dốc giảm dần ngẫu nhiên, cùng chứa *kiến thức* của mạng.

Các loại lớp khác nhau phù hợp với các định dạng tensor khác nhau và các kiểu xử lý dữ liệu khác nhau. Ví dụ: dữ liệu vectơ đơn giản, được lưu trữ trong các thang đo 2D có hình dạng `(mẫu, tính năng)`, thường được xử lý bởi các lớp *được kết nối dày đặc*, còn được gọi là các lớp *được kết nối đầy đủ* hoặc *dày đặc* (lớp `Dense` trong Keras). Dữ liệu trình tự, được lưu trữ dưới dạng tensor 3D có hình dạng `(mẫu, dấu thời gian, tính năng)`, thường được xử lý bởi các lớp *recurrent*, chẳng hạn như lớp `LSTM` hoặc lớp tích chập 1D (`Conv1D`). Dữ liệu hình ảnh, được lưu trữ trong tensor cấp 4, thường được xử lý bởi các lớp tích chập 2D (`Conv2D`).

Bạn có thể coi các lớp giống như những viên gạch LEGO của học sâu, một phép ẩn dụ được Keras thể hiện rõ ràng. Việc xây dựng các mô hình deep learning trong Keras được thực hiện bằng cách ghép các lớp tương thích lại với nhau để tạo thành các quy trình chuyển đổi dữ liệu hữu ích.

#### Lớp `Layer` cơ sở trong Keras

Một API đơn giản phải có một sự trừu tượng hóa duy nhất để tập trung mọi thứ vào giữa. Trong Keras, đó là lớp `Layer`. Mọi thứ trong Keras đều là `Lớp` hoặc thứ gì đó tương tác chặt chẽ với `Lớp`.

`Lớp` là một đối tượng bao gồm một số trạng thái (trọng số) và một số tính toán (chuyển tiếp). Các trọng số thường được xác định trong `build()` (mặc dù chúng cũng có thể được tạo trong hàm tạo `__init__()`) và tính toán được xác định trong phương thức `call()`.

Trong chương trước, chúng ta đã triển khai một lớp `NaiveDense` chứa hai trọng số `W` và `b` và áp dụng phép tính `output = activate(matmul(input, W) + b)`. Sau đây là hình ảnh của cùng một lớp trong Keras.

```python
import keras

# All Keras layers inherit from the base Layer class.
class SimpleDense(keras.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    # Weight creation takes place in the build() method.
    def build(self, input_shape):
        batch_dim, input_dim = input_shape
        # add_weight is a shortcut method for creating weights. It's
        # also possible to create standalone variables and assign them
        # as layer attributes, like self.W = keras.Variable(shape=...,
        # initializer=...).
        self.W = self.add_weight(
            shape=(input_dim, self.units), initializer="random_normal"
        )
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")

    # We define the forward pass computation in the call() method.
    def call(self, inputs):
        y = keras.ops.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y
```

[Danh sách 3.33](#listing-3-33): Một lớp dày đặc đơn giản từ đầu trong Keras

Trong phần tiếp theo, chúng tôi sẽ trình bày chi tiết mục đích của các phương thức `build()` và `call()` này. Đừng lo lắng nếu bạn chưa hiểu mọi thứ!

Sau khi được khởi tạo, một lớp như thế này có thể được sử dụng giống như một hàm, lấy đầu vào là tensor:

```python
>>> # Instantiates our layer, defined previously
>>> my_dense = SimpleDense(units=32, activation=keras.ops.relu)
>>> # Creates some test inputs
>>> input_tensor = keras.ops.ones(shape=(2, 784))
>>> # Calls the layer on the inputs, just like a function
>>> output_tensor = my_dense(input_tensor)
>>> print(output_tensor.shape)
(2, 32)
```

Bây giờ, có lẽ bạn đang thắc mắc, tại sao chúng ta phải triển khai `call()` và `build()`, vì cuối cùng chúng ta đã sử dụng lớp của mình bằng cách gọi nó một cách rõ ràng, nghĩa là bằng cách sử dụng phương thức `__call__` của nó? Đó là bởi vì chúng tôi muốn có thể tạo trạng thái đúng lúc. Hãy xem nó hoạt động như thế nào.

#### Suy luận hình dạng tự động: Xây dựng các lớp một cách nhanh chóng

Giống như những viên gạch LEGO, bạn chỉ có thể “ghép” các lớp *tương thích* lại với nhau. Khái niệm *khả năng tương thích của lớp* ở đây đề cập cụ thể đến thực tế là mỗi lớp sẽ chỉ chấp nhận các tensor đầu vào có hình dạng nhất định và sẽ trả về các tensor đầu ra có hình dạng nhất định. Hãy xem xét ví dụ sau:

```python
from keras import layers

# A dense layer with 32 output units
layer = layers.Dense(32, activation="relu")
```

Lớp này sẽ trả về một tensor có kích thước không theo lô là 32. Nó chỉ có thể được kết nối với lớp hạ lưu có vectơ 32 chiều làm đầu vào.

Khi sử dụng Keras, bạn không phải lo lắng về khả năng tương thích kích thước trong hầu hết các trường hợp vì các lớp bạn thêm vào mô hình của mình được tạo động để phù hợp với hình dạng của đầu vào đến. Ví dụ: giả sử bạn viết như sau:

```python
from keras import models
from keras import layers

model = models.Sequential(
    [
        layers.Dense(32, activation="relu"),
        layers.Dense(32),
    ]
)
```

Các lớp không nhận được bất kỳ thông tin nào về hình dạng đầu vào của chúng. Thay vào đó, họ tự động suy ra hình dạng đầu vào của mình là hình dạng của đầu vào đầu tiên họ nhìn thấy.

Trong phiên bản đồ chơi của lớp `Dense` mà chúng tôi đã triển khai ở chương 2, chúng tôi phải chuyển rõ ràng kích thước đầu vào của lớp cho hàm tạo để có thể tạo các trọng số của nó. Điều đó không lý tưởng vì nó sẽ dẫn đến các mô hình trông như thế này, trong đó mỗi lớp mới cần phải được biết về hình dạng của lớp trước nó:

```python
model = NaiveSequential(
    [
        NaiveDense(input_size=784, output_size=32, activation="relu"),
        NaiveDense(input_size=32, output_size=64, activation="relu"),
        NaiveDense(input_size=64, output_size=32, activation="relu"),
        NaiveDense(input_size=32, output_size=10, activation="softmax"),
    ]
)
```

Sẽ còn tệ hơn khi các quy tắc được sử dụng bởi một lớp để tạo ra hình dạng đầu ra của nó rất phức tạp. Ví dụ: điều gì sẽ xảy ra nếu lớp của chúng ta trả về kết quả đầu ra có dạng `(batch, input_size * 2 if input_size % 2 == 0 else input_size * 3)`?

Nếu chúng ta triển khai lại lớp `NaiveDense` của mình dưới dạng lớp Keras có khả năng suy luận hình dạng tự động, thì nó sẽ trông giống như lớp `SimpleDense`, với các phương thức `build()` và `call()` của nó.

Trong Keras `SimpleDense`, chúng tôi không còn tạo trọng số trong hàm tạo như trong ví dụ trước nữa. Thay vào đó, chúng tôi tạo chúng theo phương thức tạo trạng thái chuyên dụng `build()`, phương thức này nhận làm đối số là hình dạng đầu vào đầu tiên mà lớp nhìn thấy. Phương thức `build()` được gọi tự động trong lần đầu tiên lớp được gọi (thông qua phương thức `__call__()` của nó). Trên thực tế, đó là lý do tại sao chúng tôi xác định tính toán theo một phương thức `call()` riêng biệt thay vì trực tiếp trong phương thức `__call__()`! Phương thức `__call__()` của lớp cơ sở có sơ đồ như thế này:

```python
def __call__(self, inputs):
    if not self.built:
        self.build(inputs.shape)
        self.built = True
    return self.call(inputs)
```

Với suy luận hình dạng tự động, ví dụ trước của chúng tôi trở nên đơn giản và gọn gàng:

```python
model = keras.Sequential(
    [
        SimpleDense(32, activation="relu"),
        SimpleDense(64, activation="relu"),
        SimpleDense(32, activation="relu"),
        SimpleDense(10, activation="softmax"),
    ]
)
```

Lưu ý rằng suy luận hình dạng tự động không phải là điều duy nhất mà phương thức `__call__()` của lớp `Layer` xử lý. Nó xử lý nhiều thứ khác, đặc biệt là việc định tuyến giữa thực thi *eager* và *graph* cũng như che dấu đầu vào (chúng tôi sẽ đề cập đến trong chương 14). Bây giờ, chỉ cần nhớ: khi triển khai các lớp của riêng bạn, hãy đặt chuyển tiếp vào phương thức `call()`.

### Từ lớp đến mô hình

Một mô hình học sâu là một biểu đồ của các lớp. Trong Keras, đó là lớp `Model`. Hiện tại, bạn chỉ thấy các mô hình `Tuần tự` (một lớp con của `Model`), là các lớp xếp chồng đơn giản, ánh xạ một đầu vào duy nhất thành một đầu ra duy nhất. Nhưng khi tiến về phía trước, bạn sẽ được tiếp xúc với nhiều cấu trúc liên kết mạng đa dạng hơn nhiều. Một số cái phổ biến là

* Mạng hai nhánh
* Mạng đa đầu
* Kết nối dư

Cấu trúc liên kết mạng có thể tham gia khá nhiều. Ví dụ, hình 3.5 thể hiện cấu trúc liên kết của biểu đồ các lớp của Transformer, một kiến ​​trúc phổ biến được thiết kế để xử lý dữ liệu văn bản.

![](../images/ch03/transformer.cb3f137f.png)

[Figure 3.5](#figure-3-5): The Transformer architecture. There’s a lot going on here. Throughout the next few chapters, you’ll climb your way up to understanding it (in chapter 15).

Nhìn chung, có hai cách để xây dựng các mô hình như vậy trong Keras: bạn có thể trực tiếp phân lớp lớp `Model` hoặc bạn có thể sử dụng API chức năng, cho phép bạn làm được nhiều việc hơn với ít mã hơn. Chúng tôi sẽ đề cập đến cả hai cách tiếp cận trong chương 7.

Cấu trúc liên kết của một mô hình xác định một *không gian giả thuyết*. Bạn có thể nhớ rằng trong chương 1, chúng tôi đã mô tả học máy là “tìm kiếm các cách trình bày hữu ích của một số dữ liệu đầu vào, trong một *không gian khả năng* được xác định trước, sử dụng hướng dẫn từ tín hiệu phản hồi”. Bằng cách chọn cấu trúc liên kết mạng, bạn giới hạn không gian khả năng (không gian giả thuyết) của mình thành một chuỗi các phép toán tensor cụ thể, ánh xạ dữ liệu đầu vào sang dữ liệu đầu ra. Sau đó, những gì bạn sẽ tìm kiếm là một tập hợp các giá trị phù hợp cho các tensor trọng lượng liên quan đến các phép toán tensor này.

Để học từ dữ liệu, bạn phải đưa ra các giả định về nó. Những giả định này xác định những gì có thể học được. Như vậy, cấu trúc của không gian giả thuyết - kiến ​​trúc mô hình của bạn - là cực kỳ quan trọng. Nó mã hóa các giả định mà bạn đưa ra về vấn đề của mình, kiến ​​thức có sẵn mà mô hình bắt đầu. Ví dụ: nếu bạn đang giải quyết một vấn đề phân loại hai lớp với một mô hình được tạo từ một lớp `Dense` duy nhất không có kích hoạt (một phép biến đổi affine thuần túy), thì bạn đang giả định rằng hai lớp của bạn có thể phân tách tuyến tính.

Chọn kiến ​​trúc mạng phù hợp là một nghệ thuật hơn là khoa học và mặc dù có một số nguyên tắc và phương pháp thực hành tốt nhất mà bạn có thể dựa vào, nhưng chỉ thực hành mới có thể giúp bạn trở thành một kiến ​​trúc sư mạng nơ-ron đúng nghĩa. Một số chương tiếp theo sẽ dạy cho bạn những nguyên tắc rõ ràng để xây dựng mạng lưới thần kinh và giúp bạn phát triển trực giác về những gì hiệu quả hoặc không hiệu quả đối với các vấn đề cụ thể. Bạn sẽ xây dựng trực giác vững chắc về loại kiến ​​trúc mô hình nào phù hợp với các loại vấn đề khác nhau, cách xây dựng các mạng này trong thực tế, cách chọn cấu hình học tập phù hợp và cách điều chỉnh mô hình cho đến khi nó mang lại kết quả mà bạn muốn xem.

### Bước “biên dịch”: Cấu hình quá trình học tập

Khi kiến ​​trúc mô hình đã được xác định, bạn vẫn phải chọn thêm ba thứ nữa:

* *Hàm mất mát (hàm mục tiêu)*  — Số lượng sẽ
được giảm thiểu trong quá trình đào tạo. Nó đại diện cho thước đo thành công của
nhiệm vụ trước mắt.

* *Trình tối ưu hóa*  — Xác định cách mạng sẽ được cập nhật dựa trên tổn thất
chức năng. Nó triển khai một biến thể cụ thể của phương pháp giảm độ dốc ngẫu nhiên (SGD).

* *Số liệu* — Các thước đo thành công mà bạn muốn theo dõi trong quá trình đào tạo và
xác nhận, chẳng hạn như độ chính xác phân loại.
Không giống như mất mát, quá trình đào tạo sẽ không tối ưu hóa trực tiếp cho các số liệu này.
Như vậy, các số liệu không cần phải khác biệt.

Sau khi đã chọn mức mất mát, trình tối ưu hóa và số liệu, bạn có thể sử dụng các phương thức `compile()` và `fit()` tích hợp sẵn để bắt đầu huấn luyện mô hình của mình. Ngoài ra, bạn có thể viết các vòng đào tạo tùy chỉnh của riêng mình - chúng tôi sẽ đề cập đến cách thực hiện điều này trong chương 7. Sẽ còn nhiều việc phải làm hơn! Bây giờ, chúng ta hãy xem `compile()` và `fit()`.

Phương thức `compile()` định cấu hình quy trình đào tạo — bạn đã được giới thiệu về nó trong ví dụ đầu tiên về mạng nơ-ron ở chương 2. Nó nhận các đối số `optimizer`, `loss` và `metrics` (một danh sách):

```python
# Defines a linear classifier
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(
    # Specifies the optimizer by name: RMSprop (it's case-insensitive)
    optimizer="rmsprop",
    # Specifies the loss by name: mean squared error
    loss="mean_squared_error",
    # Specifies a list of metrics: in this case, only accuracy
    metrics=["accuracy"],
)
```

Trong lệnh gọi `compile()` trước đó, chúng ta đã chuyển trình tối ưu hóa, mất dữ liệu và số liệu dưới dạng chuỗi (chẳng hạn như `"rmsprop"`). Các chuỗi này thực chất là các phím tắt được chuyển đổi thành đối tượng Python. Ví dụ: `"rmsprop"` trở thành `keras.optimizers.RMSprop()`. Điều quan trọng là cũng có thể chỉ định các đối số này làm phiên bản đối tượng, như sau:

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.BinaryAccuracy()],
)
```

Điều này hữu ích nếu bạn muốn chuyển các số liệu hoặc tổn thất tùy chỉnh của riêng mình hoặc nếu bạn muốn định cấu hình thêm các đối tượng bạn đang sử dụng — ví dụ: bằng cách chuyển đối số `learning_rate` tới trình tối ưu hóa:

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
    loss=my_custom_loss,
    metrics=[my_custom_metric_1, my_custom_metric_2],
)
```

Trong chương 7, chúng tôi đề cập đến cách tạo số liệu và tổn thất tùy chỉnh. Nói chung, bạn sẽ không phải tạo các khoản lỗ, số liệu hoặc trình tối ưu hóa của riêng mình từ đầu vì Keras cung cấp nhiều tùy chọn tích hợp có thể bao gồm những gì bạn cần:

* *Trình tối ưu hóa*
+ `SGD()` (có hoặc không có động lượng)
+ `RMSprop()`
+ `Adam()`
+ V.v.
* *Mất mát*
+ `CategoricalCrossentropy()`
+ `SparseCategoricalCrossentropy()`
+ `BinaryCrossentropy()`
+ `MeanSquaredError()`
+ `KLD Phân kỳ()`
+ `CosineTương tự()`
+ V.v.
* *Số liệu*
+ `Độ chính xác phân loại()`
+ `Độ chính xác phân loại thưa()`
+ `Độ chính xác nhị phân()`
+ `AUC()`
+ `Độ chính xác()`
+ `Thu hồi()`
+ V.v.

Xuyên suốt cuốn sách này, bạn sẽ thấy những ứng dụng cụ thể của nhiều lựa chọn này.

### Chọn một hàm mất mát

Việc chọn hàm tổn thất phù hợp cho đúng vấn đề là cực kỳ quan trọng: mạng của bạn sẽ thực hiện bất kỳ lối tắt nào có thể để giảm thiểu tổn thất. Vì vậy, nếu mục tiêu không hoàn toàn tương quan với thành công của nhiệm vụ hiện tại thì mạng lưới của bạn sẽ thực hiện những việc mà bạn có thể không mong muốn. Hãy tưởng tượng một AI ngu ngốc, toàn năng được đào tạo thông qua SGD, với chức năng mục tiêu được lựa chọn kém này: “Tối đa hóa phúc lợi trung bình của tất cả con người còn sống”. Để thực hiện công việc của mình dễ dàng hơn, AI này có thể chọn giết tất cả con người ngoại trừ một số ít và tập trung vào sức khỏe của những người còn lại vì sức khỏe trung bình không bị ảnh hưởng bởi số lượng người còn lại. Đó có thể không phải là điều bạn dự định! Chỉ cần nhớ rằng tất cả các mạng lưới thần kinh mà bạn xây dựng sẽ tàn nhẫn trong việc giảm hàm mất mát của chúng, vì vậy hãy chọn mục tiêu một cách khôn ngoan, nếu không bạn sẽ phải đối mặt với những tác dụng phụ ngoài ý muốn.

May mắn thay, khi gặp các vấn đề phổ biến như phân loại, hồi quy và dự đoán trình tự, có những nguyên tắc đơn giản mà bạn có thể làm theo để chọn tổn thất chính xác. Ví dụ: bạn sẽ sử dụng entropy chéo nhị phân cho bài toán phân loại hai lớp, entropy chéo phân loại cho bài toán phân loại nhiều lớp, v.v. Chỉ khi bạn đang nghiên cứu những vấn đề nghiên cứu thực sự mới, bạn mới phải phát triển các hàm mất mát của riêng mình. Trong một số chương tiếp theo, chúng tôi sẽ trình bày chi tiết một cách rõ ràng những hàm mất mát nào cần chọn cho nhiều nhiệm vụ phổ biến.

### Hiểu phương pháp phù hợp

Sau `compile()` là `fit()`. Phương thức `fit` tự triển khai vòng lặp huấn luyện. Đối số chính của nó là

* *dữ liệu* (đầu vào và mục tiêu) để đào tạo. Nó thường sẽ được thông qua
ở dạng mảng NumPy hoặc đối tượng TensorFlow `Dataset`. Bạn sẽ
tìm hiểu thêm về API `Dataset` trong các chương tiếp theo.
* Số *kỷ nguyên* cần huấn luyện: vòng lặp huấn luyện bao nhiêu lần
nên lặp lại dữ liệu được truyền.
* Kích thước lô sẽ sử dụng trong mỗi giai đoạn giảm dần độ dốc của lô nhỏ:
số lượng ví dụ huấn luyện được xem xét để tính toán độ dốc cho
một bước cập nhật trọng lượng.

```python
history = model.fit(
    # The input examples, as a NumPy array
    inputs,
    # The corresponding training targets, as a NumPy array
    targets,
    # The training loop will iterate over the data 5 times.
    epochs=5,
    # The training loop will iterate over the data in batches of 128
    # examples.
    batch_size=128,
)
```

[Danh sách 3.34](#listing-3-34): Gọi `fit` với dữ liệu NumPy

Lệnh gọi `fit` trả về một đối tượng `History`. Đối tượng này chứa trường `history`, là khóa ánh xạ chính tả, chẳng hạn như `"loss"` hoặc tên số liệu cụ thể vào danh sách các giá trị mỗi kỷ nguyên của chúng:

```python
>>> history.history
{"binary_accuracy": [0.855, 0.9565, 0.9555, 0.95, 0.951],
 "loss": [0.6573270302042366,
  0.07434618508815766,
  0.07687718723714351,
  0.07412414988875389,
  0.07617757616937161]}
```

### Giám sát tổn thất và số liệu trên dữ liệu xác thực

Mục tiêu của học máy không phải là thu được các mô hình hoạt động tốt trên dữ liệu huấn luyện, điều này thật dễ dàng - tất cả những gì bạn phải làm là tuân theo độ dốc. Mục tiêu là có được các mô hình hoạt động tốt nói chung, đặc biệt là trên các điểm dữ liệu mà mô hình chưa từng gặp trước đây. Chỉ vì một mô hình hoạt động tốt trên dữ liệu huấn luyện của nó không có nghĩa là nó sẽ hoạt động tốt trên dữ liệu mà nó chưa từng thấy! Ví dụ: có thể mô hình của bạn cuối cùng chỉ *ghi nhớ* ánh xạ giữa các mẫu đào tạo của bạn và mục tiêu của chúng, điều này sẽ vô ích đối với nhiệm vụ dự đoán mục tiêu cho dữ liệu mà mô hình chưa từng thấy trước đây. Chúng ta sẽ đề cập đến điểm này chi tiết hơn trong chương 5.

Để theo dõi cách mô hình hoạt động trên dữ liệu mới, thông lệ tiêu chuẩn là đặt trước một tập hợp con dữ liệu huấn luyện dưới dạng "dữ liệu xác thực": bạn sẽ không huấn luyện mô hình trên dữ liệu này nhưng bạn sẽ sử dụng nó để tính toán giá trị tổn thất và giá trị số liệu. Bạn thực hiện việc này bằng cách sử dụng đối số `validation_data` trong `fit()`. Giống như dữ liệu huấn luyện, dữ liệu xác thực có thể được truyền dưới dạng mảng NumPy hoặc dưới dạng đối tượng TensorFlow `Dataset`.

```python
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# To avoid having samples from only one class in the validation data,
# shuffles the inputs and targets using a random indices permutation
indices_permutation = np.random.permutation(len(inputs))
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]

# Reserves 30% of the training inputs and targets for "validation."
# (We'll exclude these samples from training and reserve them to
# compute the "validation loss" and metrics).
num_validation_samples = int(0.3 * len(inputs))
val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples:]
training_targets = shuffled_targets[num_validation_samples:]
model.fit(
    # Training data, used to update the weights of the model
    training_inputs,
    training_targets,
    epochs=5,
    batch_size=16,
    # Validation data, used only to monitor the "validation loss" and
    # metrics
    validation_data=(val_inputs, val_targets),
)
```

[Liệt kê 3.35](#listing-3-35): Sử dụng đối số dữ liệu xác thực

Giá trị bị mất trên dữ liệu xác thực được gọi là *mất xác thực*, để phân biệt với *mất huấn luyện*. Lưu ý rằng điều cần thiết là phải tách biệt hoàn toàn dữ liệu huấn luyện và dữ liệu xác thực: mục đích của việc xác thực là để theo dõi xem những gì mô hình đang học có thực sự hữu ích trên dữ liệu mới hay không. Nếu mô hình nhìn thấy bất kỳ dữ liệu xác thực nào trong quá trình đào tạo, thì việc mất xác thực và số liệu của bạn sẽ bị thiếu sót.

Nếu bạn muốn tính toán tổn thất xác thực và số liệu sau khi quá trình đào tạo hoàn tất, bạn có thể gọi phương thức `evaluate`:

`loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size=128)`

`evaluate()` sẽ lặp lại theo lô (có kích thước `batch_size`) trên dữ liệu được truyền và trả về danh sách các đại lượng vô hướng, trong đó mục nhập đầu tiên là mất xác thực và các mục nhập sau là số liệu xác thực. Nếu mô hình không có số liệu thì chỉ trả về phần mất xác thực (chứ không phải danh sách).

### Suy luận: Sử dụng mô hình sau khi đào tạo

Sau khi đào tạo mô hình của mình, bạn sẽ muốn sử dụng mô hình đó để đưa ra dự đoán về dữ liệu mới. Điều này được gọi là *suy luận*. Để làm điều này, một cách tiếp cận đơn giản chỉ đơn giản là `__call__` mô hình:

```python
# Takes a NumPy array or a tensor for your current backend and returns
# a tensor for your current backend
predictions = model(new_inputs)
```

Tuy nhiên, thao tác này sẽ xử lý tất cả dữ liệu đầu vào trong `new_inputs` cùng một lúc, điều này có thể không khả thi nếu bạn đang xem nhiều dữ liệu (đặc biệt, dữ liệu có thể yêu cầu nhiều bộ nhớ hơn GPU của bạn).

Cách tốt hơn để suy luận là sử dụng phương thức `predict()`. Nó sẽ lặp lại dữ liệu theo từng đợt nhỏ và trả về một mảng dự đoán NumPy. Và không giống như `__call__`, nó cũng có thể xử lý các đối tượng TensorFlow `Dataset`:

```python
# Takes a NumPy array or a Dataset and returns a NumPy array
predictions = model.predict(new_inputs, batch_size=128)
```

Ví dụ: nếu chúng ta sử dụng `predict()` trên một số dữ liệu xác thực bằng mô hình tuyến tính mà chúng ta đã đào tạo trước đó, thì chúng ta sẽ nhận được điểm vô hướng tương ứng với dự đoán của mô hình cho từng mẫu đầu vào:

```python
>>> predictions = model.predict(val_inputs, batch_size=128)
>>> print(predictions[:10])
[[0.3590725 ]
 [0.82706255]
 [0.74428225]
 [0.682058  ]
 [0.7312616 ]
 [0.6059811 ]
 [0.78046083]
 [0.025846  ]
 [0.16594526]
 [0.72068727]]
```

Hiện tại, đây là tất cả những gì bạn cần biết về các mô hình Keras. Tại thời điểm này, bạn đã sẵn sàng chuyển sang giải quyết các vấn đề về máy trong thế giới thực với Keras trong chương tiếp theo.

## Bản tóm tắt

* TensorFlow, PyTorch và JAX là ba cấp độ thấp phổ biến
các khuôn khổ cho tính toán số và tự vi phân.
Tất cả họ đều có cách làm việc riêng và điểm mạnh và điểm yếu riêng.
* Keras là API cấp cao để xây dựng và đào tạo mạng lưới thần kinh. Nó có thể được sử dụng với một trong hai
TensorFlow, PyTorch hoặc JAX — chỉ cần chọn phần phụ trợ mà bạn thích nhất.
* Lớp trung tâm của Keras là `Layer`. Một lớp đóng gói một số trọng số
và một số phép tính. Các lớp được tập hợp thành mô hình.
* Trước khi bắt đầu huấn luyện một mô hình, bạn cần chọn một trình tối ưu hóa, một hàm mất mát,
và một số số liệu mà bạn chỉ định thông qua phương thức `model.compile()`.
* Để huấn luyện một mô hình, bạn có thể sử dụng phương thức `fit()`, phương thức này chạy gradient theo lô nhỏ
hạ cánh cho bạn. Bạn cũng có thể sử dụng nó để theo dõi tổn thất và số liệu của mình về
dữ liệu xác thực, một tập hợp đầu vào mà mô hình không nhìn thấy trong quá trình đào tạo.
* Khi mô hình của bạn được huấn luyện, bạn có thể sử dụng phương thức `model.predict()` để tạo
dự đoán về đầu vào mới.

#### **Slide**

<div class="pdf-container" style="margin-bottom: 20px; width: 100%; height: 85vh;">
  <iframe src="TaiLieu/slideDL/Chapter03.pdf#view=FitH" width="100%" height="100%" style="border: none;"></iframe>
</div>

#### ** 💻 Luyện tập **

<div class="practice-container" style="background: #f8faff; border: 1px solid #cce0ff; border-radius: 8px; padding: 20px; margin-top: 15px;">
  <h3 style="margin-top:0; color: #1a73e8; display:flex; align-items:center; gap:8px;">🚀 Bài tập Thực hành Jupyter Notebook</h3>
  <p>Dưới đây là sổ tay (notebook) chứa mã nguồn Python thực hành cho chương này. Bạn có thể mở trực tiếp trên Google Colab để chạy thử nghiệm, hoặc tải file về máy.</p>
  <ul style="list-style-type: none; padding-left: 0;">
    <li style="margin-bottom: 15px; padding: 15px; background: white; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
      <strong style="font-size:16px;">Introduction To Ml Frameworks</strong><br>
      <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
        <a href="https://colab.research.google.com/github/tranthanhthangbmt/WebDeepLearning_2026/blob/main/TaiLieu/NotebookJupyter/chapter03_introduction-to-ml-frameworks.ipynb" target="_blank" style="background: #fbbc04; color: #fff; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(251,188,4,0.3);">🔥 Mở trên Google Colab</a>
        <a href="TaiLieu/NotebookJupyter/chapter03_introduction-to-ml-frameworks.ipynb" download style="background: #1a73e8; color: white; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(26,115,232,0.3);">💾 Tải file .ipynb về máy</a>
      </div>
    </li>
  </ul>
</div>


#### ** 🎥 Video **

<iframe src="TaiLieu/Video/Chapter_03/index.html" width="100%" height="600px" style="border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" allowfullscreen></iframe>

<!-- tabs:end -->
