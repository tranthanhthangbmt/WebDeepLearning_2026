<!-- tabs:start -->

#### **Tiếng Anh (English)**

# Chapter 7: A deep dive on Keras

This chapter covers

* The different ways to create Keras models: the `Sequential` class,
  the Functional API, and model subclassing
* How to use the built-in Keras training and evaluation
  loops, including how to use custom metrics and custom losses
* Using Keras callbacks to further customize how training proceeds
* Using TensorBoard for monitoring your training and evaluation metrics over
  time
* How to write your own training and evaluation loops from scratch

You’re starting to have some amount of experience with Keras. You’re
familiar with the `Sequential` model, `Dense` layers, and built-in APIs
for training, evaluation, and inference — `compile()`, `fit()`, `evaluate()`,
and `predict()`. You’ve even learned in chapter 3 how to inherit from the
`Layer` class to create custom layers, and how to use the gradient APIs in
TensorFlow, JAX and PyTorch to implement a step-by-step training loop.

In the coming chapters, we’ll dig into computer vision, timeseries forecasting,
natural language
processing, and generative deep learning. These complex applications will
require much more than a `Sequential` architecture and the default `fit()` loop.
So let’s first turn you into a Keras expert!
In this chapter, you’ll get a complete overview of the key ways
to work with Keras APIs: everything you’re going to need to handle
the advanced deep learning use cases you’ll encounter next.

## A spectrum of workflows

The design of the Keras API is guided by the principle of
*progressive disclosure of complexity*: make it easy
to get started, yet make it possible to handle high-complexity use cases,
only requiring incremental learning at each step.
Simple use cases should be easy and approachable, and arbitrarily advanced
workflows should be *possible*: no matter how niche and complex
the thing you want to do, there should be a clear path to it,
a path that builds upon the various things you’ve learned from simpler workflows.
This means that you can grow from beginner to expert and still
use the same tools — only in different ways.

As such, there’s not a single “true” way of using Keras. Rather,
Keras offers a *spectrum of workflows*, from the very simple to the very
flexible. There are different ways to build Keras models, and different ways to
train them, answering different needs.

For instance, you have a range
of ways to build models and an array of ways to train them,
each representing a certain tradeoff between usability and flexibility.
You could be using Keras like you would use
scikit-learn — just calling `fit()` and letting the framework do its thing —
or you could be using it like NumPy —
taking full control of every little detail.

Because all these workflows are based on shared APIs, such as `Layer` and `Model`,
components from any workflow can be used in any other workflow:
they can all talk to each other. This means that everything you’re learning now as you’re getting started will
still be relevant once you’ve become an expert. You can
get started easily and then gradually dive into workflows where you’re writing
more and more logic from scratch. You won’t have to switch to an entirely
different framework as you go from student to researcher, or from data scientist
to deep learning engineer.

This philosophy is not unlike that of Python itself!
Some languages only offer one way to write programs — for instance,
object-oriented programming or functional programming. Meanwhile, Python
is a multiparadigm language: it offers a range of possible usage patterns,
which all work nicely together. This makes Python suitable for a wide range
of very different use cases: system administration, data science,
machine learning engineering, web development, or just learning how to program.
Likewise, you can think of Keras as the Python of deep learning: a user-friendly
deep learning language that offers a variety of workflows for different
user profiles.

## Different ways to build Keras models

There are three APIs for building models in Keras, as shown in figure 7.1:

* The *Sequential model* is the most approachable API — it’s basically a Python list.
  As such, it’s limited to simple stacks of layers.
* The *Functional API*, which focuses on graph-like model architectures.
  It represents a nice mid-point between usability and flexibility, and as such,
  it’s the most commonly used model-building API.
* *Model subclassing*, a low-level option where you write everything yourself
  from scratch. This is ideal if you want full control over every little thing.
  However, you won’t get access to many built-in Keras features,
  and you will be more at risk of making mistakes.

![](../images/ch07/progressive_disclosure_of_complexity_models.f43bcdb0.png)


[Figure 7.1](#figure-7-1): Progressive disclosure of complexity for model building

### The Sequential model

The simplest way to build a Keras model is the `Sequential` model,
which you already know about.

```python
import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
```

[Listing 7.1](#listing-7-1): The `Sequential` class

Note that it’s possible to build the same model incrementally via the `add()`
method, similar to the `append()` method of a Python list.

```python
model = keras.Sequential()
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
```

[Listing 7.2](#listing-7-2): Incrementally building a `Sequential` model

You’ve seen in chapter 3 that layers only get built
(which is to say, create their weights)
when they are called for the first time.
That’s because the shape of the layers’ weights depends on the shape of their
input: until the input shape is known, they can’t be created.

As such, the previous `Sequential` model does not have any weights until you
actually call it on some data, or call its `build()` method with an input shape.

```python
>>> # At that point, the model isn't built yet.
>>> model.weights
[]
```

[Listing 7.3](#listing-7-3): Models that aren’t yet built have no weights



```python
>>> # Builds the model. Now the model will expect samples of shape
>>> # (3,). The None in the input shape signals that the batch size
>>> # could be anything.
>>> model.build(input_shape=(None, 3))
>>> # Now you can retrieve the model's weights.
>>> model.weights
[<Variable shape=(3, 64), dtype=float32, path=sequential/dense_2/kernel ...>,
 <Variable shape=(64,), dtype=float32, path=sequential/dense_2/bias ...>,
 <Variable shape=(64, 10), dtype=float32, path=sequential/dense_3/kernel ...>,
 <Variable shape=(10,), dtype=float32, path=sequential/dense_3/bias ...>>]
```

[Listing 7.4](#listing-7-4): Calling a model for the first time to build it

After the model is built, you can display its contents via the `summary()`
method, which comes in handy for debugging.

```python
>>> model.summary()
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_2 (Dense)                   │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_3 (Dense)                   │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```

[Listing 7.5](#listing-7-5): The summary method

As you can see, your model happens to be named `sequential_1`. You can actually
give names to everything in Keras — every model, every layer.

```python
>>> model = keras.Sequential(name="my_example_model")
>>> model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
>>> model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
>>> model.build((None, 3))
>>> model.summary()
Model: "my_example_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ my_first_layer (Dense)            │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ my_last_layer (Dense)             │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```

[Listing 7.6](#listing-7-6): Naming models and layers with the `name` argument

When building a `Sequential` model incrementally, it’s useful to be
able to print a summary of what the current model looks like after you add each
layer. But you can’t print a summary until the model is built! There’s actually
a way to have your `Sequential` model get built on the fly: just declare the
shape of the model’s inputs in advance. You can do this via the `Input` class.

```python
model = keras.Sequential()
# Use an Input to declare the shape of the inputs. Note that the shape
# argument must be the shape of each sample, not the shape of one
# batch.
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(64, activation="relu"))
```

[Listing 7.7](#listing-7-7): Specifying the input shape of your model in advance

Now you can use `summary()` to follow how the output shape of your model changes
as you add more layers:

```python
>>> model.summary()
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_4 (Dense)                   │ (None, 64)               │           256 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 256 (1.00 KB)
 Trainable params: 256 (1.00 KB)
 Non-trainable params: 0 (0.00 B)

>>> model.add(layers.Dense(10, activation="softmax"))
>>> model.summary()
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_4 (Dense)                   │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_5 (Dense)                   │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```

This is a pretty common debugging workflow when dealing with layers that
transform their inputs in complex ways, such as the convolutional layers you’ll
learn about in chapter 8.

### The Functional API

The `Sequential` model is easy to use, but its applicability is
extremely limited: it can only express models with a single input and a single
output, applying one layer after the other in a sequential fashion.
In practice, it’s pretty common to encounter models with multiple inputs
(say, an image and its metadata), multiple outputs
(different things you want to predict about the data), or a nonlinear topology.

In such cases, you’d build your model using the Functional API. This is what
most Keras models you’ll encounter in the wild use. It’s fun and powerful —
it feels like playing with LEGO bricks.

#### A simple example

Let’s start with something simple: the two-layer stack we used in the previous
section. Its Functional API version looks like the following listing.

```python
inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")
```

[Listing 7.8](#listing-7-8): A simple Functional model with two `Dense` layers

Let’s go over this step by step.
We started by declaring an `Input`
(note that you can also give names to these input objects, like everything else):

```python
inputs = keras.Input(shape=(3,), name="my_input")
```

This `inputs` object holds information about the shape and `dtype` of the data
that the model will process:

```python
>>> # The model will process batches where each sample has shape (3,).
>>> # The number of samples per batch is variable (indicated by the
>>> # None batch size).
>>> inputs.shape
(None, 3)
>>> # These batches will have dtype float32.
>>> inputs.dtype
"float32"
```

We call such an object a *symbolic tensor*. It doesn’t contain any actual data,
but it encodes the specifications of the actual tensors of data that the
model will see when you use it. It *stands for* future tensors of data.

Next, we created a layer and called it on the input:

```python
features = layers.Dense(64, activation="relu")(inputs)
```

All Keras layers can be called both on real tensors of data or on these
symbolic tensors. In the latter case, they return a new symbolic tensor,
with updated shape and dtype information:

```python
>>> features.shape
(None, 64)
```

After obtaining the final outputs, we instantiated the model by specifying
its inputs and outputs in the `Model` constructor:

```python
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")
```

Here’s the summary of our model:

```python
>>> model.summary()
Model: "my_functional_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ my_input (InputLayer)             │ (None, 3)                │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_8 (Dense)                   │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_9 (Dense)                   │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```

#### Multi-input, multi-output models

Unlike this toy model, most deep learning models don’t look like lists —
they look like graphs. They may, for instance, have multiple inputs
or multiple outputs.
It’s for this kind of model that the Functional API really shines.

Let’s say you’re building a system to rank customer support tickets
by priority and route them to the appropriate department. Your model has three
inputs:

* The title of the ticket (text input)
* The text body of the ticket (text input)
* Any tags added by the user (categorical input, assumed here to be multi-hot encoded)

We can encode the text inputs as arrays of 1s and 0s of size `vocabulary_size`
(see chapter 14 for detailed information about text encoding techniques).

Your model also has two outputs:

* The priority score of the ticket, a scalar between 0 and 1 (sigmoid output)
* The department that should handle the ticket (a softmax over the set of departments)

You can build this model in a few lines with the Functional API.

```python
vocabulary_size = 10000
num_tags = 100
num_departments = 4

# Defines model inputs
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

# Combines input features into a single tensor, features, by
# concatenating them
features = layers.Concatenate()([title, text_body, tags])
# Applies intermediate layer to recombine input features into richer
# representations
features = layers.Dense(64, activation="relu", name="dense_features")(features)

# Defines model outputs
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(
    num_departments, activation="softmax", name="department"
)(features)

# Creates the model by specifying its inputs and outputs
model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department],
)
```

[Listing 7.9](#listing-7-9): A multi-input, multi-output Functional model

The Functional API is a simple, LEGO-like, yet very flexible way to define
arbitrary graphs of layers like these.

#### Training a multi-input, multi-output model

You can train your model in much the same way as you would train a `Sequential` model,
by calling `fit()` with lists of input and output data.
These lists of data should respect the same order
as the inputs you passed to the `Model()` constructor.

```python
import numpy as np

num_samples = 1280

# Dummy input data
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# Dummy target data
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, num_departments, size=(num_samples, 1))

model.compile(
    optimizer="adam",
    loss=["mean_squared_error", "sparse_categorical_crossentropy"],
    metrics=[["mean_absolute_error"], ["accuracy"]],
)
model.fit(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data],
    epochs=1,
)
model.evaluate(
    [title_data, text_body_data, tags_data], [priority_data, department_data]
)
priority_preds, department_preds = model.predict(
    [title_data, text_body_data, tags_data]
)
```

[Listing 7.10](#listing-7-10): Training a model by providing lists of input and target arrays

If you don’t want to rely on input order (for instance, because you have many inputs or outputs),
you can also use the names you gave to the `Input` objects and to the output layers, and
pass data via dictionaries.

```python
model.compile(
    optimizer="adam",
    loss={
        "priority": "mean_squared_error",
        "department": "sparse_categorical_crossentropy",
    },
    metrics={
        "priority": ["mean_absolute_error"],
        "department": ["accuracy"],
    },
)
model.fit(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data},
    epochs=1,
)
model.evaluate(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data},
)
priority_preds, department_preds = model.predict(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)
```

[Listing 7.11](#listing-7-11): Training a model by providing dicts of input and target arrays

#### The power of the Functional API: Access to layer connectivity

A Functional model is an explicit graph data structure. This makes it possible
to *inspect how layers are connected* and *reuse previous graph nodes*
(which are layer outputs) as part of new models. It also nicely fits
the “mental model” that most researchers use when thinking about a deep neural
network: a graph of layers.

This enables two important use cases: model visualization and feature extraction.
Let’s take a look.

##### Plotting layer connectivity

Let’s visualize the connectivity of the model we just defined
(the *topology* of the model). You can plot a Functional model as a graph with
the `plot_model()` utility, as shown in figure 7.2:

```python
keras.utils.plot_model(model, "ticket_classifier.png")
```


![](../images/ch07/ticket_classifier.6d8d5711.png)


[Figure 7.2](#figure-7-2): Plot generated by `plot_model()` on our ticket classifier model

You can add to this plot the input and output shapes of each layer in the model,
as well as layer names (rather than just layer types),
which can be helpful during debugging (figure 7.3):

```python
keras.utils.plot_model(
    model,
    "ticket_classifier_with_shape_info.png",
    show_shapes=True,
    show_layer_names=True,
)
```


![](../images/ch07/ticket_classifier_with_shapes.1c26af81.png)


[Figure 7.3](#figure-7-3): Model plot with shape information added

The `None` in the tensor shapes represents the batch size: this model allows
batches of any size.

##### Feature extraction with a Functional model

Access to layer connectivity also means that you can inspect and reuse
individual nodes (layer calls) in the graph. The model property `model.layers`
provides the list of layers that make up the model,
and for each layer, you can query `layer.input` and `layer.output`.

```python
>>> model.layers
[<InputLayer name=title, built=True>,
 <InputLayer name=text_body, built=True>,
 <InputLayer name=tags, built=True>,
 <Concatenate name=concatenate, built=True>,
 <Dense name=dense_10, built=True>,
 <Dense name=priority, built=True>,
 <Dense name=department, built=True>]

>>> model.layers[3].input
[<KerasTensor shape=(None, 10000), dtype=float32, sparse=None, name=title>,
 <KerasTensor shape=(None, 10000), dtype=float32, sparse=None, name=text_body>,
 <KerasTensor shape=(None, 100), dtype=float32, sparse=None, name=tags>]

>>> model.layers[3].output
<KerasTensor shape=(None, 20100), dtype=float32, sparse=False>
```

[Listing 7.12](#listing-7-12): Retrieving the inputs or outputs of a layer in a Functional model

This enables you to do *feature extraction*: creating models that reuse intermediate
features from another model.

Let’s say you want to add another output to the model we previously defined — you
want to also predict an estimate of how long a given issue ticket will take to
resolve, a kind of difficulty rating.
You could do this via a classification layer over three categories — “quick,”
“medium,” and “difficult.” You don’t need to recreate and retrain a model from scratch!
You can just start from the intermediate features of your previous model, since you
have access to them.

```python
# layers[4] is our intermediate Dense layer.
features = model.layers[4].output
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)

new_model = keras.Model(
    inputs=[title, text_body, tags], outputs=[priority, department, difficulty]
)
```

[Listing 7.13](#listing-7-13): Creating a new model by reusing intermediate layer outputs

Let’s plot our new model, as shown in figure 7.4:

```python
keras.utils.plot_model(
    new_model,
    "updated_ticket_classifier.png",
    show_shapes=True,
    show_layer_names=True,
)
```


![](../images/ch07/updated_ticket_classifier.d0baf1fe.png)


[Figure 7.4](#figure-7-4): Plot of our new model

### Subclassing the Model class

The last model-building pattern you should know about is the most advanced one:
`Model` subclassing. You’ve already learned in chapter 3
how to subclass the `Layer` class to create custom layers.
Subclassing `Model` is pretty similar:

* In the `__init__` method, define the layers the model will use.
* In the `call` method, define the forward pass of the model,
  reusing the layers previously created.
* Instantiate your subclass and call it on data to create its weights.

#### Rewriting our previous example as a subclassed model

Let’s take a look at a simple example: we will reimplement the customer support
ticket management model using a `Model` subclass.

```python
class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        # Don't forget to call the super constructor!
        super().__init__()
        # Defines sublayers in the constructor
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(
            num_departments, activation="softmax"
        )

    # Defines the forward pass in the call() method
    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department
```

[Listing 7.14](#listing-7-14): A simple subclassed model

Once you’ve defined the model, you can instantiate it. Note that it will only create its weights
the first time you call it on some data — much like `Layer` subclasses:

```python
model = CustomerTicketModel(num_departments=4)

priority, department = model(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)
```

So far, everything looks very similar to `Layer` subclassing, a workflow
you’ve already encountered in chapter 3. What, then, is the difference between
a `Layer` subclass and a `Model` subclass? It’s simple: a *layer* is a building
block you use to create models, and a *model* is the top-level object that you
will actually train, export for inference, etc. In short, a `Model` has a
`fit()`, `evaluate()`, and `predict()` method. Layers don’t.
Other than that, the two classes are virtually identical
(another difference is that you can *save* a model to a file on disk —
which we will cover in a few sections).

You can compile and train a `Model` subclass just like a Sequential
or Functional model:

```python
model.compile(
    optimizer="adam",
    # The structure of what you pass as the loss and metrics must match
    # exactly what gets returned by call() — since we returned a list
    # of two elements, so should loss and metrics be lists of two
    # elements.
    loss=["mean_squared_error", "sparse_categorical_crossentropy"],
    metrics=[["mean_absolute_error"], ["accuracy"]],
)
model.fit(
    # The structure of the input data must match exactly what is
    # expected by the call() method, and the structure of the target
    # data must match exactly what gets returned by the call() method.
    # Here, the input data must be a dict with three keys (title,
    # text_body, and tags) and the target data must be a list of two
    # elements.
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    [priority_data, department_data],
    epochs=1,
)
model.evaluate(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    [priority_data, department_data],
)
priority_preds, department_preds = model.predict(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)
```

The `Model` subclassing workflow is the most flexible way to build a model: it enables
you to build models that cannot be expressed as directed acyclic graphs of layers —
imagine, for instance, a model where the `call()` method uses layers inside a `for` loop,
or even calls them recursively. Anything is possible — you’re in charge.

#### Beware: What subclassed models don’t support

This freedom comes at a cost: with subclassed models, you are responsible for more
of the model logic, which means your potential error surface is much larger.
As a result, you will have more debugging work to do.
You are developing a new Python object, not just snapping together LEGO bricks.

Functional and subclassed models are also substantially different in nature:
a Functional model is an explicit data structure — a graph of layers,
which you can view, inspect, and modify.
Meanwhile, a subclassed model is a piece of bytecode —
a Python class with a `call()` method that contains raw code.
This is the source of the subclassing workflow’s flexibility —
you can just code up whatever functionality you like —
but it introduces new limitations.

For instance, because the way layers are connected to each other
is hidden inside the body of the `call()` method, you cannot access that information.
Calling `summary()` will not display layer connectivity, and you cannot plot
the model topology via `plot_model()`. Likewise, if you have a subclassed model,
you cannot access the nodes of the graph
of layers to do feature extraction — because there is simply no graph.
Once the model is instantiated, its forward pass becomes a complete black box.

### Mixing and matching different components

Crucially, choosing one of these patterns —
the `Sequential` model, the Functional API, `Model` subclassing —
does not lock you out of the others. All models in the Keras API can
smoothly interoperate with each other, whether they’re Sequential models,
Functional models, or subclassed models written from scratch.
They’re all part of the same spectrum of workflows.
For instance, you can use a subclassed layer or model in a Functional model.

```python
class Classifier(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = "sigmoid"
        else:
            num_units = num_classes
            activation = "softmax"
        self.dense = layers.Dense(num_units, activation=activation)

    def call(self, inputs):
        return self.dense(inputs)

inputs = keras.Input(shape=(3,))
features = layers.Dense(64, activation="relu")(inputs)
outputs = Classifier(num_classes=10)(features)
model = keras.Model(inputs=inputs, outputs=outputs)
```

[Listing 7.15](#listing-7-15): Creating a Functional model that includes a subclassed model

Inversely, you can use a Functional model as part of a subclassed layer or model.

```python
inputs = keras.Input(shape=(64,))
outputs = layers.Dense(1, activation="sigmoid")(inputs)
binary_classifier = keras.Model(inputs=inputs, outputs=outputs)

class MyModel(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        self.dense = layers.Dense(64, activation="relu")
        self.classifier = binary_classifier

    def call(self, inputs):
        features = self.dense(inputs)
        return self.classifier(features)

model = MyModel()
```

[Listing 7.16](#listing-7-16): Creating a subclassed model that includes a Functional model

### Remember: Use the right tool for the job

You’ve learned about the spectrum of workflows for building Keras models,
from the simplest workflow — the `Sequential` model — to the most advanced one,
model subclassing. When should you use one over the other?
Each one has its pros and cons — pick the one most suitable for the job at hand.

In general, the Functional API provides you with a pretty good tradeoff between
ease of use and flexibility. It also gives you direct access to layer connectivity,
which is very powerful for use cases such as model plotting or feature extraction.
If you *can* use the Functional API — that is, if your model can be expressed as
a directed acyclic graph of layers — we recommend using it over model subclassing.

Going forward, all examples in this book will use the Functional API — simply
because all of the models we will work with are expressible as graphs of layers.
We will, however, make frequent use of subclassed layers.
In general, using Functional models
that include subclassed layers provides the best of both worlds: high development
flexibility while retaining the advantages of the Functional API.

## Using built-in training and evaluation loops

The principle of progressive disclosure of complexity — access to a spectrum
of workflows that go from dead easy to arbitrarily flexible, one step at a time
— also applies to model training. Keras provides
you with different workflows for training models — it can be as simple as calling
`fit()` on your data or as advanced as writing a new training algorithm from scratch.

You are already familiar with the `compile()`, `fit()`, `evaluate()`, `predict()`
workflow. As a reminder, it looks like the following listing.

```python
from keras.datasets import mnist

# Creates a model. (We factor this into a separate function so as to
# reuse it later.)
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

# Loads your data, reserving some for validation
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()
# Compiles the model by specifying its optimizer, the loss function to
# minimize, and metrics to monitor
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
# Uses `fit()` to train the model, optionally providing validation data
# to monitor performance on unseen data
model.fit(
    train_images,
    train_labels,
    epochs=3,
    validation_data=(val_images, val_labels),
)
# Uses `evaluate()` to compute the loss and metrics on new data
test_metrics = model.evaluate(test_images, test_labels)
# Uses `predict()` to compute classification probabilities on new data
predictions = model.predict(test_images)
```

[Listing 7.17](#listing-7-17): The standard workflow: `compile()`, `fit()`, `evaluate()`, `predict()`

There are a couple of ways you can customize this simple workflow:

* By providing your own custom metrics
* By passing *callbacks* to the `fit()` method to schedule actions
  to be taken at specific points during training

Let’s take a look at these.

### Writing your own metrics

Metrics are key to measuring the performance of your model — in particular,
to measure the difference between its performance on the training data
and its performance on the test data. Commonly used metrics for classification
and regression are already part of the built-in `keras.metrics` module —
most of the time, that’s what you will use.
But if you’re doing anything out of the ordinary, you will need to be able
to write your own metrics. It’s simple!

A Keras metric is a subclass of the `keras.metrics.Metric` class. Similarly
to layers, a metric has an internal state stored in Keras variables.
Unlike layers, these variables aren’t updated via backpropagation,
so you have to write the state update logic yourself — which happens
in the `update_state()` method.
For example, here’s a simple custom metric that measures the root mean squared error (RMSE).

```python
from keras import ops

# Subclasses the Metric class
class RootMeanSquaredError(keras.metrics.Metric):
    # Defines the state variables in the constructor. Like for layers,
    # you have access to the add_weight() method.
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(
            name="total_samples", initializer="zeros"
        )

    # Implements the state update logic in update_state(). The y_true
    # argument is the targets (or labels) for one batch, while y_pred
    # represents the corresponding predictions from the model. To match
    # our MNIST model, we expect categorical predictions and integer
    # labels. You can ignore the sample_weight argument; we won't use
    # it here.
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.one_hot(y_true, num_classes=ops.shape(y_pred)[1])
        mse = ops.sum(ops.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = ops.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)
```

[Listing 7.18](#listing-7-18): Implementing a custom metric by subclassing the `Metric` class

You use the `result()` method to return the current value of the metric:

```python
    def result(self):
        return ops.sqrt(self.mse_sum / self.total_samples)
```

Meanwhile, you also need to expose a way to reset the metric state without
having to reinstantiate it — this enables the same metric objects to
be used across different epochs of training or across both training and evaluation.
You do this in the `reset_state()` method:

```python
    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0.)
```

Custom metrics can be used just like built-in ones. Let’s test-drive our own metric:

```python
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", RootMeanSquaredError()],
)
model.fit(
    train_images,
    train_labels,
    epochs=3,
    validation_data=(val_images, val_labels),
)
test_metrics = model.evaluate(test_images, test_labels)
```

You can now see the `fit()` progress bar display the RMSE of your
model.

### Using callbacks

Launching a training run on a large dataset for tens of epochs using `model.fit()`
can be a bit like launching a paper airplane: past the initial impulse,
you don’t have any control over its trajectory or its landing spot.
If you want to avoid bad outcomes (and thus wasted paper airplanes),
it’s smarter to use, not a paper plane, but a drone that can sense its environment,
send data back to its operator, and automatically make steering decisions based
on its current state. The Keras *callbacks* API will help you
transform your call to `model.fit()` from a paper airplane into a smart,
autonomous drone that can self-introspect and dynamically take action.

A *callback* is an object (a class instance implementing specific methods)
that is passed to the model in the call to `fit()` and that is called by the model
at various points during training. It has access to all the available data about
the state of the model and its performance, and it can take action:
interrupt training, save a model, load a different weight set,
or otherwise alter the state of the model.

Here are some examples of ways you can use callbacks:

* *Model checkpointing* — Saving the current state of the model
  at different points during training.

* *Early stopping* — Interrupting training when the validation loss is
  no longer improving (and of course, saving the best model obtained during training).

* *Dynamically adjusting the value of certain parameters during training* —
  Such as the learning rate of the optimizer.

* *Logging training and validation metrics during training, or visualizing the representations learned by the model as they’re updated* —
  The `fit()` progress bar that you’re familiar with is in fact a callback!

The `keras.callbacks` module includes a number of built-in callbacks (this is not an exhaustive list):

```python
keras.callbacks.ModelCheckpoint
keras.callbacks.EarlyStopping
keras.callbacks.LearningRateScheduler
keras.callbacks.ReduceLROnPlateau
keras.callbacks.CSVLogger
```

Let’s review two of them to give you an idea of how to use them:
`EarlyStopping` and `ModelCheckpoint`.

#### The EarlyStopping and ModelCheckpoint callbacks

When you’re training a model, there are many things you can’t predict at the start.
In particular, you can’t tell how many epochs will be needed to get to an optimal
validation loss. Our examples so far have adopted the strategy of training
for enough epochs that you begin overfitting, using the first run to figure
out the optimal number of epochs, and then finally launching
a new training run from scratch using this optimal number.
Of course, this approach is wasteful. A much better way to handle this is to
stop training when you measure that the validation loss is no longer improving.
This can be achieved using the `EarlyStopping` callback.

The `EarlyStopping` callback interrupts training once a target
metric being monitored has stopped improving for a fixed number of epochs.
For instance, this callback allows you to interrupt training as soon as you
start overfitting, thus avoiding having to retrain your model for a smaller
number of epochs. This callback is typically used in
combination with `ModelCheckpoint`, which lets you continually save the model
during training (and, optionally, save only the current best model so far:
the version of the model that achieved the best performance at the end of an epoch).

```python
# Callbacks are passed to the model via the callbacks argument in
# fit(), which takes a list of callbacks. You can pass any number of
# callbacks.
callbacks_list = [
    # Interrupts training when improvement stops
    keras.callbacks.EarlyStopping(
        # Monitors the model's validation accuracy
        monitor="accuracy",
        # Interrupts training when accuracy has stopped improving for
        # more than one epoch (that is, two epochs)
        patience=1,
    ),
    # Saves the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath="checkpoint_path.keras",
        # These two arguments mean you won't overwrite the model file
        # unless val_loss has improved, which allows you to keep the
        # best model seen during training.
        monitor="val_loss",
        save_best_only=True,
    ),
]
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    # You monitor accuracy, so it should be part of the model's
    # metrics.
    metrics=["accuracy"],
)
# Because the callback will monitor validation loss and validation
# accuracy, you need to pass validation_data to the call to fit().
model.fit(
    train_images,
    train_labels,
    epochs=10,
    callbacks=callbacks_list,
    validation_data=(val_images, val_labels),
)
```

[Listing 7.19](#listing-7-19): Using the `callbacks` argument in the `fit()` method

Note that you can always save models manually after training as well —
just call `model.save("my_checkpoint_path.keras")`.
To reload the model you’ve saved, use

```python
model = keras.models.load_model("checkpoint_path.keras")
```

### Writing your own callbacks

If you need to take a specific action during training that isn’t covered by
one of the built-in callbacks, you can write your own callback.
Callbacks are implemented by subclassing the class `keras.callbacks.Callback`.
You can then implement any number of the following transparently named methods,
which are called at various points during training:

```python
# Called at the start of every epoch
on_epoch_begin(epoch, logs)
# Called at the end of every epoch
on_epoch_end(epoch, logs)
# Called right before processing each batch
on_batch_begin(batch, logs)
# Called right after processing each batch
on_batch_end(batch, logs)
# Called at the start of training
on_train_begin(logs)
# Called at the end of training
on_train_end(logs)
```

These methods are all called with a `logs` argument, which is a dictionary
containing information about the previous batch, epoch, or training run:
training and validation metrics, and so on. The `on_epoch_*` and `on_batch_*`
methods also take the epoch or batch index as first argument (an integer).

Here’s a simple example callback that saves a list of per-batch loss values during training
and plots these values at the end of each epoch.

```python
from matplotlib import pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(
            range(len(self.per_batch_losses)),
            self.per_batch_losses,
            label="Training loss for each batch",
        )
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_at_epoch_{epoch}", dpi=300)
        self.per_batch_losses = []
```

[Listing 7.20](#listing-7-20): Creating a custom callback by subclassing the `Callback` class

Let’s test-drive it:

```python
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_images,
    train_labels,
    epochs=10,
    callbacks=[LossHistory()],
    validation_data=(val_images, val_labels),
)
```

We get plots that look like figure 7.5.

![](../images/ch07/loss_history_callback_example.1e42f6b2.png)


[Figure 7.5](#figure-7-5): The output of our custom history-plotting callback

### Monitoring and visualization with TensorBoard

To do good research or develop good models, you need rich, frequent feedback
about what’s going on inside your models during your experiments.
That’s the point of running experiments: to get information about how well
a model performs — as much information as possible.
Making progress is an iterative process, a loop: you start with an idea
and express it as an experiment, attempting to validate or invalidate your idea.
You run this experiment and process the information it generates, as shown in figure 7.6.
This inspires your next idea. The more iterations of this loop you’re able to run,
the more refined and powerful your ideas become. Keras helps you go from idea
to experiment in the least possible time, and fast
GPUs can help you get from experiment to result as quickly as possible.
But what about processing the experiment results? That’s where TensorBoard comes in.

![](../images/ch07/the_loop_of_progress.df126e89.png)


[Figure 7.6](#figure-7-6): The loop of progress

TensorBoard is a browser-based application that you can run locally. It’s
the best way to monitor everything that goes on inside your model
during training. With TensorBoard, you can

* Visually monitor metrics during training
* Visualize your model architecture
* Visualize histograms of activations and gradients
* Explore embeddings in 3D

If you’re monitoring more information than just the model’s final loss,
you can develop a clearer vision of what the model does and doesn’t do,
and you can make progress more quickly.

The easiest way to use TensorBoard with a Keras model and the `fit()` method is the
`keras.callbacks.TensorBoard` callback.
In the simplest case, just specify where you want the callback to write logs,
and you’re good to go:

```python
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

tensorboard = keras.callbacks.TensorBoard(
    log_dir="/full_path_to_your_log_dir",
)
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(val_images, val_labels),
    callbacks=[tensorboard],
)
```

Once the model starts running, it will write logs at the target location.
If you are running your Python script on a local machine, you can then
launch the local TensorBoard server using the following command (note that
the `tensorboard` executable should already be available if you have
installed TensorFlow via `pip`; if not, you can install TensorBoard manually
via `pip install tensorboard`):

```python
tensorboard --logdir /full_path_to_your_log_dir
```

You can then navigate to the URL that the command returns to access the
TensorBoard interface.

If you are running your script in a Colab notebook, you can run an embedded
TensorBoard instance as part of your notebook, using the following commands:

```python
%load_ext tensorboard
%tensorboard --logdir /full_path_to_your_log_dir
```

In the TensorBoard interface, you will be able to monitor live graphs of your
training and evaluation metrics, as shown in figure 7.7.

![](../images/ch07/tensorboard.aec6cc75.png)


[Figure 7.7](#figure-7-7): TensorBoard can be used for easy monitoring of training and evaluation metrics.

## Writing your own training and evaluation loops

The `fit()` workflow strikes a nice balance between ease of use and flexibility.
It’s what you will use most of the time.
However, it isn’t meant to support everything a deep learning researcher may want to do
— even with custom metrics, custom losses, and custom callbacks.

After all, the built-in `fit()` workflow is solely focused on *supervised learning*:
a setup where there are known *targets* (also called *labels* or *annotations*)
associated with your input data and where you compute your loss as a function
of these targets and the model’s predictions.
However, not every form of machine learning falls into this category.
There are other setups where no explicit targets are present,
such as *generative learning* (which we will introduce in chapter 16),
*self-supervised learning* (where targets are obtained from the inputs), or
*reinforcement learning* (where learning is driven by occasional “rewards” — much like training a dog).
And even if you’re doing regular supervised learning, as a researcher,
you may want to add some novel bells and whistles that require low-level flexibility.

Whenever you find yourself in a situation where the built-in `fit()` is not enough,
you will need to write your own custom training logic.
You’ve already seen simple examples of low-level training loops in chapters 2 and 3.
As a reminder, the contents of a typical training loop look like this:

1. Run the “forward pass” (compute the model’s output)
   to obtain a loss value for the current batch of data.
2. Retrieve the gradients of the loss with regard to the model’s weights.
3. Update the model’s weights so as to lower the loss value on the current batch of data.

These steps are repeated for as many batches as necessary. This is
essentially what `fit()` does under the hood. In this
section, you will learn to reimplement `fit()` from scratch, which will give
you all the knowledge you need to write any training algorithm you may come up with.

Let’s go over the details. Throughout the next few sections, you’ll work your way up
to writing a fully featured custom training loop in TensorFlow, PyTorch, and JAX.

### Training vs. inference

In the low-level training loop examples you’ve seen so far, step 1 (the forward
pass) was done via `predictions = model(inputs)`, and step 2 (retrieving the
gradients computed by the gradient tape) was done via a backend-specific API,
such as

* `gradients = tape.gradient(loss, model.weights)` in TensorFlow
* `loss.backward()` in PyTorch
* `jax.value_and_grad()` in JAX

In the general case, there are actually two subtleties you need to take into account.

Some Keras layers, such as the `Dropout` layer,
have different behaviors during *training* and during *inference*
(when you use them to generate predictions).
Such layers expose a `training` Boolean argument in their `call()` method.
Calling `dropout(inputs, training=True)` will drop some activation entries, while
calling `dropout(inputs, training=False)` does nothing.
By extension, Functional models and Sequential models also expose this
`training` argument in their `call()` methods. Remember to pass `training=True`
when you call a Keras model during the forward pass! Our forward pass thus
becomes `predictions = model(inputs, training=True)`.

In addition, note that when you retrieve the gradients of the weights
of your model, you should not use `model.weights`,
but rather `model.trainable_weights`. Indeed, layers and
models own two kinds of weights:

* *Trainable weights*, meant to be updated via backpropagation to minimize
  the loss of the model, such as the kernel and bias of a `Dense` layer.
* *Non-trainable weights*, which are meant to be updated during the forward pass
  by the layers that own them. For instance, if you wanted a custom layer to keep a counter
  of how many batches it has processed so far, that information would be stored
  in a non-trainable weight, and at each batch, your layer would increment the
  counter by one.

Among Keras built-in layers, the only layer that features non-trainable weights
is the `BatchNormalization` layer, which we will introduce in chapter 9.
The `BatchNormalization` layer needs non-trainable weights
to track information about the mean and standard deviation of the data that
passes through it, so as to perform an online approximation of
*feature normalization* (a concept you’ve learned about in chapters 4 and 6).

### Writing custom training step functions

Taking into account these two details, a supervised learning training step
ends up looking like this in pseudocode:

```python
def train_step(inputs, targets):
    # Runs the forward pass
    predictions = model(inputs, training=True)
    # Computes the loss for the current batch
    loss = loss_fn(targets, predictions)
    # Retrieves the gradients of the loss with regard to the model's
    # trainable weights This function doesn't actually exist!
    gradients = get_gradients_of(loss, wrt=model.trainable_weights)
    # Updates the model's trainable weights based on the gradients
    optimizer.apply(gradients, model.trainable_weights)
```

This snippet is pseudocode rather than real code because it includes an
imaginary function, `get_gradients_of()`. In reality, retrieving gradients
is done in a way that is specific to your current backend — JAX, TensorFlow, or PyTorch.

Let’s use what you learned about each framework in chapter 3 to implement a real version
of this `train_step()` function. We’ll start with TensorFlow and PyTorch because
these two make the job relatively easy, so they’re a good place to start. We’ll end with JAX,
which is quite a bit more complex.

#### A TensorFlow training step function

TensorFlow lets you write code that looks pretty much like our pseudocode snippet.
The only difference is that your forward pass should take place inside a `GradientTape`
scope. You can then use the `tape` object to retrieve the gradients:

```python
import tensorflow as tf

model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

def train_step(inputs, targets):
    # Opens a GradientTape
    with tf.GradientTape() as tape:
        # Runs the forward pass
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    # Retrieves the gradients from the tape
    gradients = tape.gradient(loss, model.trainable_weights)
    # Updates the model's trainable weights based on the gradients
    optimizer.apply(gradients, model.trainable_weights)
    return loss
```

Let’s run it for a single step:

```python
batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = train_step(inputs, targets)
```

Easy enough! Let’s do PyTorch next.

#### A PyTorch training step function

When you use the PyTorch backend, all of your Keras layers and models inherit from the PyTorch `torch.nn.Module`
class and expose the native `Module` API. As a result, your model, its trainable weights, and your loss tensor are all aware of each other
and interact via three methods: `loss.backward()`, `weight.value.grad`, and `model.zero_grad()`.

As a reminder from chapter 3, the mental model you’ve got to keep in mind is this:

* With each forward pass, PyTorch builds up a one-time computation graph that keeps
  track of the computation that just happened.
* Calling `.backward()` on any given scalar node of this graph (like your loss) will run the graph backward
  starting from that node, automatically populating a `tensor.grad` attribute on all tensors involved (if they satisfy `requires_grad=True`),
  containing the gradient of the output node with respect to that tensor. In particular, it will populate the `grad` attribute
  of your trainable parameters.
* To clear the contents of that `tensor.grad` attribute, you should call `tensor.grad = None` on all
  your tensors. Because it would be a bit cumbersome to do this on all model variables individually,
  you can just do it at the model level via `model.zero_grad()` — the `zero_grad()` call will propagate to all variables
  tracked by the model. Clearing gradients is critical because calls to `backward()` are additive: if you don’t
  clear the gradients at each step, the gradient values will accumulate and training won’t proceed.

Let’s chain these steps:

```python
import torch

model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

def train_step(inputs, targets):
    # Runs the forward pass
    predictions = model(inputs, training=True)
    loss = loss_fn(targets, predictions)
    # Runs the backward pass, populating gradient values
    loss.backward()
    # Recovers the gradient associated with each trainable variable.
    # That weight.value is the PyTorch tensor that contains the
    # variable's value.
    gradients = [weight.value.grad for weight in model.trainable_weights]
    # Updates the model's trainable weights based on the gradients.
    # This must be done in a no_grad() scope.
    with torch.no_grad():
        optimizer.apply(gradients, model.trainable_weights)
    # Don't forget to clear the gradients!
    model.zero_grad()
    return loss
```

Let’s run it for a single step:

```python
batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = train_step(inputs, targets)
```

That wasn’t too difficult! Now, let’s move on to JAX.

#### A JAX training step function

When it comes to low-level training code, JAX tends to be the most complex
of the three backends because of its fully stateless nature. Statelessness
makes JAX highly performant and scalable, making it amenable to
compilation and automatic performance optimizations. However, writing stateless code
requires you to jump through some hoops.

Since the gradient function is obtained via metaprogramming,
you first need to define the function that returns your loss.
Further, this function needs to be stateless, so it needs to take as arguments all
the variables it’s going to be using, and it needs to return the value of any variable
it has updated. Remember those non-trainable weights that can get modified during the
forward pass? Those are the variables we need to return.

To make it easier to work with the stateless programming paradigm of JAX,
Keras models make available a stateless forward pass method:
the `stateless_call()` method. It behaves just like `__call__`, except that

* It takes as input the model’s trainable weights and non-trainable weights, in addition
  to the `inputs` and `training` arguments.
* It returns the model’s updated non-trainable weights, in addition to the model’s outputs.

It works like this:

```python
outputs, non_trainable_weights = model.stateless_call(
    trainable_weights, non_trainable_weights, inputs
)
```

We can use `stateless_call()` to implement our JAX loss function. Since the loss function
also computes updates for all non-trainable variables, we name it `compute_loss_and_updates()`:

```python
model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# Gradients are computed for the entries in the first argument
# (trainable_variables here)
def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, inputs, targets
):
    # Calls stateless_call
    outputs, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, inputs, training=True
    )
    loss = loss_fn(targets, outputs)
    # Returns the scalar loss value and the updated non-trainable
    # weights
    return loss, non_trainable_variables
```

Once we have this `compute_loss_and_updates()` function, we can pass it
to `jax.value_and_grad` to obtain the gradient computation:

```python
import jax

grad_fn = jax.value_and_grad(fn)
loss, gradients = grad_fn(...)
```

Now, there’s just a small problem. Both `jax.grad()` and `jax.value_and_grad()`
require `fn` to return a scalar value only. Our `compute_loss_and_updates()`
function returns a scalar value as its first output, but it also returns the new value for the
non-trainable weights. Remember what you learned in chapter 3?
The solution is to pass a `has_aux` argument to `grad()` or `value_and_grad()`, like this:

```python
import jax

grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
```

You would use it like this:

```python
(loss, non_trainable_weights), gradients = grad_fn(
    trainable_variables, non_trainable_variables, inputs, targets
)
```

Okay, that was a lot of JAXiness. But now we’ve got almost everything we need to assemble
our JAX training step. We just need the last piece of the puzzle: `optimizer.apply()`.

When you wrote your first basic training step in TensorFlow at the beginning of chapter 2,
you wrote an update step function that looked like this:

```python
learning_rate = 1e-3

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign(w - g * learning_rate)
```

This corresponds to what the optimizer `keras.optimizers.SGD` would do. However, every other
optimizer in the Keras API is somewhat more complex than that and keeps track of auxiliary variables
that help speed up training — in particular, most optimizers use some form of *momentum*,
which you learned about in chapter 2. These extra variables get updated at each step of training,
and in the JAX world, that means that you need to get your hands on a stateless function
that takes these variables as arguments and returns their new value.

To make this easy, Keras makes available the `stateless_apply()` method on all optimizers.
It works like this:

```python
trainable_variables, optimizer_variables = optimizer.stateless_apply(
    optimizer_variables, grads, trainable_variables
)
```

Now, we have enough to assemble an end-to-end training step:

```python
optimizer = keras.optimizers.Adam()
optimizer.build(model.trainable_variables)

# The state is part of the function arguments.
def train_step(state, inputs, targets):
    # Unpacks the state
    (trainable_variables, non_trainable_variables, optimizer_variables) = state
    # Computes gradients and updates to non-trainable variables
    (loss, non_trainable_variables), grads = grad_fn(
        trainable_variables, non_trainable_variables, inputs, targets
    )
    # Updates trainable variables and optimizer variables
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    return loss, (
        # Returns the updated state alongside the loss
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )
```

Let’s run it for a single step:

```python
batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]

trainable_variables = [v.value for v in model.trainable_variables]
non_trainable_variables = [v.value for v in model.non_trainable_variables]
optimizer_variables = [v.value for v in optimizer.variables]

state = (trainable_variables, non_trainable_variables, optimizer_variables)
loss, state = train_step(state, inputs, targets)
```

It’s definitely a bit more work than TensorFlow and PyTorch, but the speed and scalability benefits
of JAX more than make up for it.

Next, let’s take a look at another important element of a custom training loop: *metrics*.

### Low-level usage of metrics

In a low-level training loop, you will probably want to use Keras metrics
(whether custom ones or the built-in ones). You’ve already learned about
the metrics API: simply call `update_state(y_true, y_pred)` for each
batch of targets and predictions, and then use `result()` to query the current
metric value:

```python
from keras import ops

metric = keras.metrics.SparseCategoricalAccuracy()
targets = ops.array([0, 1, 2])
predictions = ops.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
metric.update_state(targets, predictions)
current_result = metric.result()
print(f"result: {current_result:.2f}")
```

You may also need to track the average of a scalar value, such as the model’s
loss. You can do this via the `keras.metrics.Mean` metric:

```python
values = ops.array([0, 1, 2, 3, 4])
mean_tracker = keras.metrics.Mean()
for value in values:
    mean_tracker.update_state(value)
print(f"Mean of values: {mean_tracker.result():.2f}")
```

Remember to use `metric.reset_state()` when you want to reset the current
results (at the start of a training epoch or at the start of evaluation).

Now, if you’re using JAX, state-modifying methods like `update_state()` or `reset()`
can’t be used inside a stateless function. Instead, you can use the stateless metrics API,
which is similar to the `model.stateless_call()` and `optimizer.stateless_apply()` methods
you’ve already learned about. Here’s how it works:

```python
metric = keras.metrics.SparseCategoricalAccuracy()
targets = ops.array([0, 1, 2])
predictions = ops.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Gets the metric's state variables
metric_variables = metric.variables
# Gets updated values for the metric's state
metric_variables = metric.stateless_update_state(
    metric_variables, targets, predictions
)
# Computes the metric value corresponding to the current state
current_result = metric.stateless_result(metric_variables)
print(f"result: {current_result:.2f}")

# Gets blank variable values for the metric
metric_variables = metric.stateless_reset_state()
```

### Using fit() with a custom training loop

In the previous sections, we were writing our own training logic entirely from scratch.
Doing so provides you with the most flexibility, but you end up writing a lot of code,
while simultaneously missing out on many convenient features of `fit()`,
such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use
the power of the built-in Keras training loop?
There’s actually a middle ground between `fit()` and a training loop written from scratch:
you can provide a custom training step function and let the framework do the rest.

You can do this by overriding the `train_step()` method of the `Model` class.
This is the function that is called by `fit()` for
every batch of data. You will then be able to call `fit()` as usual — and it will be
running your own learning algorithm under the hood.

Here’s how it works:

* Create a new class that subclasses `keras.Model`.
* Override the `train_step()` method. Its contents are
  nearly identical to what we used in the previous section.
* Return a dictionary mapping metric names (including the loss) to their current
  value.

Note the following:

* This pattern does not prevent you from building models with the Functional
  API. You can do this whether you’re building `Sequential` models, Functional API
  models, or subclassed models.
* You don’t need to use a `@tf.function` or `@jax.jit` decorator
  when you override `train_step()` — the framework does it for you.

#### Customizing fit() with TensorFlow

Let’s start by coding a custom TensorFlow train step:

```python
import keras
from keras import layers

loss_fn = keras.losses.SparseCategoricalCrossentropy()
# This metric object will be used to track the average of per-batch
# losses during training and evaluation.
loss_tracker = keras.metrics.Mean(name="loss")

class CustomModel(keras.Model):
    # Overrides the train_step() method
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            # We use self(inputs, training=True) instead of
            # model(inputs, training=True) since our model is the class
            # itself.
            predictions = self(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply(gradients, self.trainable_weights)

        # Updates the loss tracker metric that tracks the average of
        # the loss
        loss_tracker.update_state(loss)
        # Returns the average loss so far by querying the loss tracker
        # metric
        return {"loss": loss_tracker.result()}

    # Listing the loss tracker metric in the model.metrics property
    # enables the model to automatically call reset_state() on it at
    # the start of each epoch and at the start of a call to evaluate()
    # — so you don't have to do it by hand. Any metric you would like
    # to reset across epochs should be listed here.
    @property
    def metrics(self):
        return [loss_tracker]
```

[Listing 7.21](#listing-7-21): Customizing `fit()`: TensorFlow version

We can now instantiate our custom model, compile it (we only pass the optimizer,
since the loss is already defined outside of the model),
and train it using `fit()` as usual.

Let’s put the model definition in its own reusable function:

```python
def get_custom_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam())
    return model
```

Let’s give it a whirl:

```python
model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

#### Customizing fit() with PyTorch

Next, the PyTorch version:

```python
import keras
from keras import layers

loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")

class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        # Runs the forward pass
        predictions = self(inputs, training=True)
        loss = loss_fn(targets, predictions)

        # Retrieves the gradients
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            # Updates weights
            self.optimizer.apply(gradients, trainable_weights)

        # Updates loss tracker metric
        loss_tracker.update_state(loss)
        # Returns the average loss so far by querying the loss tracker
        # metric
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        return [loss_tracker]
```

Let’s try it:

```python
model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

#### Customizing fit() with JAX

Finally, let’s write the JAX version. First we need to define a `compute_loss_and_updates()` method,
similar to the `compute_loss_and_updates()` function we used in our custom training step example:

```python
import keras
from keras import layers

loss_fn = keras.losses.SparseCategoricalCrossentropy()

class CustomModel(keras.Model):
    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        inputs,
        targets,
        training=False,
    ):
        predictions, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            inputs,
            training=training,
        )
        loss = loss_fn(targets, predictions)
        # Returns both the loss and the updated non-trainable variables
        return loss, non_trainable_variables
```

Note we aren’t computing a moving average of the loss like we did for the other two backends.
Instead we just return the per-batch loss value, which is less useful.
We do this to simplify metric state management in the example: the code would get very verbose if we included it (you will learn about metric management in the next section):

```python
    def train_step(self, state, data):
        # Unpacks the state. metrics_variables are part of it, although
        # we won't use them here.
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        inputs, targets = data

        # Gets the gradient function
        grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates, has_aux=True
        )

        # Computes gradients and updates to non-trainable variables
        (loss, non_trainable_variables), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            inputs,
            targets,
            training=True,
        )

        # Updates trainable variables and optimizer variables
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        # We aren't computing a moving average of the loss, instead
        # returning the per-batch value.
        logs = {"loss": loss}
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )
        # Returns metric logs and updated state variables
        return logs, state
```

Let’s try it out:

```python
model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

### Handling metrics in a custom train\_step()

Finally, what about the `loss` and `metrics` that you can pass to `compile()`?
After you’ve called `compile()`, you get access to

* `self.compute_loss` — This combines the loss function you passed to `compile()`
  together with regularization losses that may be added by certain layers.
* `self.metrics` — The list of metrics you passed to `compile()`. Note
  that it also includes a metric that tracks the loss.

#### train\_step() metrics handling with TensorFlow

Here’s what it looks like with TensorFlow:

```python
import keras
from keras import layers

class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            # Computes the loss via self.compute_loss
            loss = self.compute_loss(y=targets, y_pred=predictions)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply(gradients, self.trainable_weights)

        # Updates the model's metrics, including the one that tracks
        # the loss
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(targets, predictions)

        # Returns a dict mapping metric names to their current value
        return {m.name: m.result() for m in self.metrics}
```

Let’s try it:

```python
def get_custom_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

#### train\_step() metrics handling with PyTorch

And here’s what it looks like with PyTorch — it’s exactly the same code change!

```python
import keras
from keras import layers

class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        predictions = self(inputs, training=True)
        loss = self.compute_loss(y=targets, y_pred=predictions)

        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}
```

Let’s see how it runs:

```python
def get_custom_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

#### train\_step() metrics handling with JAX

Finally, here’s what it looks like with JAX.
To start with, you can use `compute_loss()` in your `compute_loss_and_updates()` method
to hit the loss passed to `compile()`:

```python
import keras
from keras import layers

class CustomModel(keras.Model):
    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        inputs,
        targets,
        training=False,
    ):
        predictions, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            inputs,
            training=training,
        )
        loss = self.compute_loss(y=targets, y_pred=predictions)
        return loss, (predictions, non_trainable_variables)
```

Next up: metric management. As usual, it’s a tad more complicated
due to JAX’s statelessness requirements:

```python
    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            # Metric variables are part of the state.
            metrics_variables,
        ) = state
        inputs, targets = data

        grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates, has_aux=True
        )

        (loss, (predictions, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            inputs,
            targets,
            training=True,
        )
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        new_metrics_vars = []
        logs = {}
        # Iterates over metrics
        for metric in self.metrics:
            num_prev = len(new_metrics_vars)
            num_current = len(metric.variables)
            # Grabs the variables of the current metrics
            current_vars = metrics_variables[num_prev : num_prev + num_current]
            # Updates the metric's state
            if metric.name == "loss":
                current_vars = metric.stateless_update_state(current_vars, loss)
            else:
                current_vars = metric.stateless_update_state(
                    current_vars, targets, predictions
                )
            # Stores the results in the logs dict
            logs[metric.name] = metric.stateless_result(current_vars)
            new_metrics_vars += current_vars

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            # Returns the new metrics variables as part of the state
            new_metrics_vars,
        )
        return logs, state
```

That was a lot of information, but by now you know enough to use Keras
to do almost anything!

## Summary

* Keras offers a spectrum of different workflows, based on the principle of
  *progressive disclosure of complexity*. They all smoothly interoperate.
* You can build models via the `Sequential` class, via the Functional API,
  or by subclassing the `Model` class. Most of the time, you’ll be using the Functional API.
* The simplest way to train and evaluate a model is via the default `fit()` and `evaluate()` methods.
* Keras callbacks provide a simple way to monitor models during your call to `fit()`
  and automatically take action based on the state of the model.
* You can also fully control what `fit()` does by overriding the `train_step()` method,
  using APIs from your backend of choice — JAX, TensorFlow, or PyTorch.
* Beyond `fit()`, you can also write your own training loops entirely from scratch,
  in a backend-native way. This is useful for researchers implementing brand-new training algorithms.

#### **Tiếng Việt (Vietnamese)**

# Chương 7: Tìm hiểu sâu về Keras

Chương này bao gồm

* Các cách khác nhau để tạo mô hình Keras: lớp `Sequential`,
API chức năng và phân lớp mô hình
* Cách sử dụng tính năng đào tạo và đánh giá Keras tích hợp
vòng lặp, bao gồm cách sử dụng số liệu tùy chỉnh và tổn thất tùy chỉnh
* Sử dụng lệnh gọi lại Keras để tùy chỉnh thêm cách tiến hành đào tạo
* Sử dụng TensorBoard để theo dõi các số liệu đào tạo và đánh giá của bạn
thời gian
* Cách viết vòng lặp đào tạo và đánh giá của riêng bạn từ đầu

Bạn đang bắt đầu có một số kinh nghiệm với Keras. Bạn đã quen thuộc với mô hình `Tuần tự`, các lớp `Dense` và các API tích hợp để đào tạo, đánh giá và suy luận — `compile()`, `fit()`, `evaluate()` và `predict()`. Bạn thậm chí đã học trong chương 3 cách kế thừa từ lớp `Layer` để tạo các lớp tùy chỉnh và cách sử dụng API gradient trong TensorFlow, JAX và PyTorch để triển khai vòng lặp đào tạo từng bước.

Trong các chương tiếp theo, chúng ta sẽ tìm hiểu sâu về thị giác máy tính, dự báo chuỗi thời gian, xử lý ngôn ngữ tự nhiên và học sâu tổng quát. Những ứng dụng phức tạp này sẽ yêu cầu nhiều hơn kiến ​​trúc `Tuần tự` và vòng lặp `fit()` mặc định. Vì vậy, trước tiên hãy biến bạn thành một chuyên gia về Keras! Trong chương này, bạn sẽ có cái nhìn tổng quan đầy đủ về các cách chính để làm việc với API Keras: mọi thứ bạn cần để xử lý các trường hợp sử dụng deep learning nâng cao mà bạn sẽ gặp tiếp theo.

## Một loạt các quy trình làm việc

Thiết kế của API Keras được hướng dẫn theo nguyên tắc *tiết lộ dần dần về độ phức tạp*: giúp bạn dễ dàng bắt đầu nhưng vẫn có thể xử lý các trường hợp sử dụng có độ phức tạp cao, chỉ yêu cầu học hỏi tăng dần ở mỗi bước. Các trường hợp sử dụng đơn giản phải dễ dàng và dễ tiếp cận, đồng thời các quy trình công việc nâng cao tùy ý phải *có thể*: cho dù việc bạn muốn làm thích hợp và phức tạp đến đâu thì cũng phải có đường dẫn rõ ràng đến việc đó, đường dẫn được xây dựng dựa trên nhiều thứ khác nhau mà bạn đã học được từ các quy trình làm việc đơn giản hơn. Điều này có nghĩa là bạn có thể phát triển từ người mới bắt đầu thành chuyên gia và vẫn sử dụng các công cụ giống nhau — chỉ theo những cách khác nhau.

Như vậy, không có một cách sử dụng Keras “đúng đắn” nào cả. Đúng hơn, Keras cung cấp *phổ các quy trình làm việc*, từ rất đơn giản đến rất linh hoạt. Có nhiều cách khác nhau để xây dựng mô hình Keras và các cách khác nhau để huấn luyện chúng, đáp ứng các nhu cầu khác nhau.

Ví dụ: bạn có nhiều cách để xây dựng mô hình và nhiều cách để huấn luyện chúng, mỗi cách thể hiện sự cân bằng nhất định giữa khả năng sử dụng và tính linh hoạt. Bạn có thể sử dụng Keras giống như sử dụng scikit-learn — chỉ cần gọi `fit()` và để khung làm việc của nó — hoặc bạn có thể sử dụng nó như NumPy — kiểm soát hoàn toàn từng chi tiết nhỏ.

Bởi vì tất cả các quy trình công việc này đều dựa trên các API được chia sẻ, chẳng hạn như `Layer` và `Model`, nên các thành phần từ bất kỳ quy trình công việc nào cũng có thể được sử dụng trong bất kỳ quy trình công việc nào khác: tất cả chúng đều có thể giao tiếp với nhau. Điều này có nghĩa là mọi thứ bạn đang học bây giờ khi bắt đầu sẽ vẫn phù hợp khi bạn đã trở thành chuyên gia. Bạn có thể bắt đầu một cách dễ dàng và sau đó dần dần đi sâu vào các quy trình công việc nơi bạn ngày càng viết nhiều logic hơn từ đầu. Bạn sẽ không phải chuyển sang một khuôn khổ hoàn toàn khác khi từ sinh viên trở thành nhà nghiên cứu hoặc từ nhà khoa học dữ liệu sang kỹ sư học sâu.

Triết lý này không khác gì triết lý của Python! Một số ngôn ngữ chỉ cung cấp một cách để viết chương trình - ví dụ: lập trình hướng đối tượng hoặc lập trình chức năng. Trong khi đó, Python là một ngôn ngữ đa mô hình: nó cung cấp một loạt các kiểu sử dụng có thể có, tất cả đều phối hợp tốt với nhau. Điều này làm cho Python phù hợp với nhiều trường hợp sử dụng rất khác nhau: quản trị hệ thống, khoa học dữ liệu, kỹ thuật học máy, phát triển web hoặc chỉ học cách lập trình. Tương tự như vậy, bạn có thể coi Keras như Python của deep learning: một ngôn ngữ deep learning thân thiện với người dùng, cung cấp nhiều quy trình làm việc khác nhau cho các hồ sơ người dùng khác nhau.

## Các cách khác nhau để xây dựng mô hình Keras

Có ba API để xây dựng mô hình trong Keras, như trong hình 7.1:

* *Mô hình tuần tự* là API dễ tiếp cận nhất — về cơ bản nó là một danh sách Python.
Như vậy, nó bị giới hạn ở các lớp xếp chồng đơn giản.
* *API chức năng*, tập trung vào kiến ​​trúc mô hình giống như đồ thị.
Nó đại diện cho điểm trung gian tốt đẹp giữa khả năng sử dụng và tính linh hoạt, và như vậy,
đó là API xây dựng mô hình được sử dụng phổ biến nhất.
* *Phân lớp mô hình*, một tùy chọn cấp thấp trong đó bạn tự viết mọi thứ
từ đầu. Điều này là lý tưởng nếu bạn muốn toàn quyền kiểm soát mọi điều nhỏ nhặt.
Tuy nhiên, bạn sẽ không có quyền truy cập vào nhiều tính năng Keras tích hợp sẵn,
và bạn sẽ có nhiều nguy cơ mắc sai lầm hơn.

![](../images/ch07/progressive_disclosure_of_complexity_models.f43bcdb0.png)

[Figure 7.1](#figure-7-1): Progressive disclosure of complexity for model building

### Mô hình tuần tự

Cách đơn giản nhất để xây dựng mô hình Keras là mô hình `Sequential` mà bạn đã biết.

```python
import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
```

[Liệt kê 7.1](#listing-7-1): Lớp `Tuần tự`

Lưu ý rằng có thể xây dựng dần dần cùng một mô hình thông qua phương thức `add()`, tương tự như phương thức `append()` của danh sách Python.

```python
model = keras.Sequential()
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
```

[Liệt kê 7.2](#listing-7-2): Xây dựng dần mô hình `Tuần tự`

Bạn đã thấy trong chương 3 rằng các lớp chỉ được xây dựng (nghĩa là tạo ra trọng số của chúng) khi chúng được gọi lần đầu tiên. Đó là bởi vì hình dạng trọng số của các lớp phụ thuộc vào hình dạng đầu vào của chúng: chúng không thể được tạo cho đến khi biết được hình dạng đầu vào.

Như vậy, mô hình `Tuần tự` trước đó không có bất kỳ trọng số nào cho đến khi bạn thực sự gọi nó trên một số dữ liệu hoặc gọi phương thức `build()` của nó với hình dạng đầu vào.

```python
>>> # At that point, the model isn't built yet.
>>> model.weights
[]
```

[Liệt kê 7.3](#listing-7-3): Các mô hình chưa được xây dựng sẽ không có trọng số



```python
>>> # Builds the model. Now the model will expect samples of shape
>>> # (3,). The None in the input shape signals that the batch size
>>> # could be anything.
>>> model.build(input_shape=(None, 3))
>>> # Now you can retrieve the model's weights.
>>> model.weights
[<Variable shape=(3, 64), dtype=float32, path=sequential/dense_2/kernel ...>,
 <Variable shape=(64,), dtype=float32, path=sequential/dense_2/bias ...>,
 <Variable shape=(64, 10), dtype=float32, path=sequential/dense_3/kernel ...>,
 <Variable shape=(10,), dtype=float32, path=sequential/dense_3/bias ...>>]
```

[Liệt kê 7.4](#listing-7-4): Gọi một mô hình lần đầu tiên để xây dựng nó

Sau khi mô hình được xây dựng, bạn có thể hiển thị nội dung của nó thông qua phương thức `summary()`, rất hữu ích cho việc gỡ lỗi.

```python
>>> model.summary()
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_2 (Dense)                   │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_3 (Dense)                   │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```

[Liệt kê 7.5](#listing-7-5): Phương pháp tóm tắt

Như bạn có thể thấy, mô hình của bạn có tên là `sequential_1`. Bạn thực sự có thể đặt tên cho mọi thứ trong Keras - mọi mô hình, mọi lớp.

```python
>>> model = keras.Sequential(name="my_example_model")
>>> model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
>>> model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
>>> model.build((None, 3))
>>> model.summary()
Model: "my_example_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ my_first_layer (Dense)            │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ my_last_layer (Dense)             │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```

[Liệt kê 7.6](#listing-7-6): Đặt tên cho các mô hình và lớp bằng đối số `name`

Khi xây dựng mô hình `Tuần tự` theo từng bước, sẽ rất hữu ích khi có thể in bản tóm tắt về mô hình hiện tại trông như thế nào sau khi bạn thêm từng lớp. Nhưng bạn không thể in bản tóm tắt cho đến khi mô hình được xây dựng! Thực sự có một cách để mô hình `Tuần tự` của bạn được xây dựng nhanh chóng: chỉ cần khai báo trước hình dạng của các đầu vào của mô hình. Bạn có thể thực hiện việc này thông qua lớp `Input`.

```python
model = keras.Sequential()
# Use an Input to declare the shape of the inputs. Note that the shape
# argument must be the shape of each sample, not the shape of one
# batch.
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(64, activation="relu"))
```

[Liệt kê 7.7](#listing-7-7): Chỉ định trước hình dạng đầu vào của mô hình của bạn

Bây giờ bạn có thể sử dụng `summary()` để theo dõi hình dạng đầu ra của mô hình thay đổi như thế nào khi bạn thêm nhiều lớp hơn:

```python
>>> model.summary()
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_4 (Dense)                   │ (None, 64)               │           256 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 256 (1.00 KB)
 Trainable params: 256 (1.00 KB)
 Non-trainable params: 0 (0.00 B)

>>> model.add(layers.Dense(10, activation="softmax"))
>>> model.summary()
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_4 (Dense)                   │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_5 (Dense)                   │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```

Đây là quy trình gỡ lỗi khá phổ biến khi xử lý các lớp biến đổi đầu vào của chúng theo những cách phức tạp, chẳng hạn như các lớp tích chập mà bạn sẽ tìm hiểu trong chương 8.

### API chức năng

Mô hình `Tuần tự` rất dễ sử dụng, nhưng khả năng ứng dụng của nó cực kỳ hạn chế: nó chỉ có thể thể hiện các mô hình có một đầu vào và một đầu ra duy nhất, áp dụng hết lớp này đến lớp khác theo kiểu tuần tự. Trong thực tế, khá phổ biến khi gặp các mô hình có nhiều đầu vào (ví dụ: một hình ảnh và siêu dữ liệu của nó), nhiều đầu ra (những thứ khác nhau mà bạn muốn dự đoán về dữ liệu) hoặc cấu trúc liên kết phi tuyến tính.

Trong những trường hợp như vậy, bạn sẽ xây dựng mô hình của mình bằng API chức năng. Đây là điều mà hầu hết các mẫu Keras bạn sẽ gặp khi sử dụng tự nhiên. Thật thú vị và mạnh mẽ - cảm giác như đang chơi với những viên gạch LEGO.

#### Một ví dụ đơn giản

Hãy bắt đầu với thứ gì đó đơn giản: ngăn xếp hai lớp mà chúng ta đã sử dụng trong phần trước. Phiên bản API chức năng của nó trông giống như danh sách sau.

```python
inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")
```

[Liệt kê 7.8](#listing-7-8): Một mô hình Hàm đơn giản với hai lớp `Dense`

Chúng ta hãy đi qua từng bước này. Chúng tôi bắt đầu bằng cách khai báo một `Input` (lưu ý rằng bạn cũng có thể đặt tên cho các đối tượng đầu vào này, giống như mọi thứ khác):

```python
inputs = keras.Input(shape=(3,), name="my_input")
```

Đối tượng `inputs` này chứa thông tin về hình dạng và `dtype` của dữ liệu mà mô hình sẽ xử lý:

```python
>>> # The model will process batches where each sample has shape (3,).
>>> # The number of samples per batch is variable (indicated by the
>>> # None batch size).
>>> inputs.shape
(None, 3)
>>> # These batches will have dtype float32.
>>> inputs.dtype
"float32"
```

Chúng tôi gọi một đối tượng như vậy là *tensor ký hiệu*. Nó không chứa bất kỳ dữ liệu thực tế nào nhưng nó mã hóa các thông số kỹ thuật của các tensor dữ liệu thực tế mà mô hình sẽ thấy khi bạn sử dụng nó. Nó *là viết tắt của* tensor dữ liệu trong tương lai.

Tiếp theo, chúng tôi tạo một lớp và gọi nó trên đầu vào:

```python
features = layers.Dense(64, activation="relu")(inputs)
```

Tất cả các lớp Keras có thể được gọi cả trên các tensor dữ liệu thực hoặc trên các tensor tượng trưng này. Trong trường hợp sau, chúng trả về một tenxơ tượng trưng mới, với thông tin về hình dạng và dtype được cập nhật:

```python
>>> features.shape
(None, 64)
```

Sau khi có được kết quả đầu ra cuối cùng, chúng tôi đã khởi tạo mô hình bằng cách chỉ định đầu vào và đầu ra của nó trong hàm tạo `Model`:

```python
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")
```

Đây là tóm tắt về mô hình của chúng tôi:

```python
>>> model.summary()
Model: "my_functional_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ my_input (InputLayer)             │ (None, 3)                │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_8 (Dense)                   │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_9 (Dense)                   │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```

#### Mô hình nhiều đầu vào, nhiều đầu ra

Không giống như mô hình đồ chơi này, hầu hết các mô hình deep learning không giống như danh sách mà trông giống như biểu đồ. Ví dụ, chúng có thể có nhiều đầu vào hoặc nhiều đầu ra. Đối với loại mô hình này, API chức năng thực sự tỏa sáng.

Giả sử bạn đang xây dựng một hệ thống để xếp hạng các yêu cầu hỗ trợ khách hàng theo mức độ ưu tiên và chuyển chúng đến bộ phận thích hợp. Mô hình của bạn có ba đầu vào:

* Tiêu đề của vé (nhập văn bản)
* Nội dung văn bản của vé (nhập văn bản)
* Bất kỳ thẻ nào được người dùng thêm vào (đầu vào phân loại, được giả định ở đây là được mã hóa nhiều điểm)

Chúng ta có thể mã hóa đầu vào văn bản dưới dạng mảng 1 và 0 có kích thước `vocabulary_size` (xem chương 14 để biết thông tin chi tiết về kỹ thuật mã hóa văn bản).

Mô hình của bạn cũng có hai kết quả đầu ra:

* Điểm ưu tiên của phiếu, vô hướng từ 0 đến 1 (đầu ra sigmoid)
* Bộ phận sẽ xử lý yêu cầu (softmax trên tập hợp các bộ phận)

Bạn có thể xây dựng mô hình này trong một vài dòng bằng API chức năng.

```python
vocabulary_size = 10000
num_tags = 100
num_departments = 4

# Defines model inputs
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

# Combines input features into a single tensor, features, by
# concatenating them
features = layers.Concatenate()([title, text_body, tags])
# Applies intermediate layer to recombine input features into richer
# representations
features = layers.Dense(64, activation="relu", name="dense_features")(features)

# Defines model outputs
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(
    num_departments, activation="softmax", name="department"
)(features)

# Creates the model by specifying its inputs and outputs
model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department],
)
```

[Liệt kê 7.9](#listing-7-9): Mô hình hàm nhiều đầu vào, nhiều đầu ra

API chức năng là một cách đơn giản, giống LEGO nhưng rất linh hoạt để xác định biểu đồ tùy ý của các lớp như thế này.

#### Huấn luyện mô hình đa đầu vào, đa đầu ra

Bạn có thể huấn luyện mô hình của mình theo cách tương tự như cách bạn huấn luyện mô hình `Tuần tự`, bằng cách gọi `fit()` với danh sách dữ liệu đầu vào và đầu ra. Các danh sách dữ liệu này phải tuân theo cùng thứ tự với các dữ liệu đầu vào mà bạn đã chuyển đến hàm tạo `Model()`.

```python
import numpy as np

num_samples = 1280

# Dummy input data
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# Dummy target data
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, num_departments, size=(num_samples, 1))

model.compile(
    optimizer="adam",
    loss=["mean_squared_error", "sparse_categorical_crossentropy"],
    metrics=[["mean_absolute_error"], ["accuracy"]],
)
model.fit(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data],
    epochs=1,
)
model.evaluate(
    [title_data, text_body_data, tags_data], [priority_data, department_data]
)
priority_preds, department_preds = model.predict(
    [title_data, text_body_data, tags_data]
)
```

[Liệt kê 7.10](#listing-7-10): Huấn luyện một mô hình bằng cách cung cấp danh sách các mảng đầu vào và mục tiêu

Nếu bạn không muốn dựa vào thứ tự đầu vào (ví dụ: vì bạn có nhiều đầu vào hoặc đầu ra), bạn cũng có thể sử dụng tên bạn đã đặt cho các đối tượng `Input` và các lớp đầu ra, đồng thời truyền dữ liệu qua từ điển.

```python
model.compile(
    optimizer="adam",
    loss={
        "priority": "mean_squared_error",
        "department": "sparse_categorical_crossentropy",
    },
    metrics={
        "priority": ["mean_absolute_error"],
        "department": ["accuracy"],
    },
)
model.fit(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data},
    epochs=1,
)
model.evaluate(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data},
)
priority_preds, department_preds = model.predict(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)
```

[Liệt kê 7.11](#listing-7-11): Huấn luyện một mô hình bằng cách cung cấp các mệnh lệnh về mảng đầu vào và mảng mục tiêu

#### Sức mạnh của API chức năng: Truy cập vào kết nối lớp

Mô hình chức năng là một cấu trúc dữ liệu đồ thị rõ ràng. Điều này cho phép *kiểm tra cách các lớp được kết nối* và *sử dụng lại các nút biểu đồ trước đó* (là đầu ra của lớp) như một phần của mô hình mới. Nó cũng rất phù hợp với “mô hình tinh thần” mà hầu hết các nhà nghiên cứu sử dụng khi nghĩ về mạng lưới thần kinh sâu: biểu đồ các lớp.

Điều này cho phép hai trường hợp sử dụng quan trọng: trực quan hóa mô hình và trích xuất tính năng. Chúng ta hãy xem xét.

##### Kết nối lớp vẽ đồ thị

Hãy hình dung khả năng kết nối của mô hình mà chúng ta vừa xác định (*cấu trúc liên kết* của mô hình). Bạn có thể vẽ mô hình Hàm dưới dạng biểu đồ bằng tiện ích `plot_model()`, như trong hình 7.2:

```python
keras.utils.plot_model(model, "ticket_classifier.png")
```

![](../images/ch07/ticket_classifier.6d8d5711.png)

[Figure 7.2](#figure-7-2): Plot generated by `plot_model()` on our ticket classifier model

Bạn có thể thêm vào biểu đồ này hình dạng đầu vào và đầu ra của từng lớp trong mô hình, cũng như tên lớp (thay vì chỉ các loại lớp), điều này có thể hữu ích trong quá trình gỡ lỗi (hình 7.3):

```python
keras.utils.plot_model(
    model,
    "ticket_classifier_with_shape_info.png",
    show_shapes=True,
    show_layer_names=True,
)
```

![](../images/ch07/ticket_classifier_with_shapes.1c26af81.png)

[Figure 7.3](#figure-7-3): Model plot with shape information added

`None` trong các hình tensor biểu thị kích thước lô: mô hình này cho phép các lô có kích thước bất kỳ.

##### Trích xuất tính năng bằng mô hình chức năng

Quyền truy cập vào kết nối lớp cũng có nghĩa là bạn có thể kiểm tra và sử dụng lại các nút riêng lẻ (lệnh gọi lớp) trong biểu đồ. Thuộc tính mô hình `model.layers` cung cấp danh sách các lớp tạo nên mô hình và với mỗi lớp, bạn có thể truy vấn `layer.input` và `layer.output`.

```python
>>> model.layers
[<InputLayer name=title, built=True>,
 <InputLayer name=text_body, built=True>,
 <InputLayer name=tags, built=True>,
 <Concatenate name=concatenate, built=True>,
 <Dense name=dense_10, built=True>,
 <Dense name=priority, built=True>,
 <Dense name=department, built=True>]

>>> model.layers[3].input
[<KerasTensor shape=(None, 10000), dtype=float32, sparse=None, name=title>,
 <KerasTensor shape=(None, 10000), dtype=float32, sparse=None, name=text_body>,
 <KerasTensor shape=(None, 100), dtype=float32, sparse=None, name=tags>]

>>> model.layers[3].output
<KerasTensor shape=(None, 20100), dtype=float32, sparse=False>
```

[Liệt kê 7.12](#listing-7-12): Truy xuất đầu vào hoặc đầu ra của một lớp trong Mô hình chức năng

Điều này cho phép bạn thực hiện *trích xuất tính năng*: tạo mô hình sử dụng lại các tính năng trung gian từ mô hình khác.

Giả sử bạn muốn thêm một đầu ra khác vào mô hình mà chúng tôi đã xác định trước đó - bạn cũng muốn dự đoán ước tính về thời gian giải quyết một phiếu phát hành nhất định, một loại xếp hạng độ khó. Bạn có thể thực hiện việc này thông qua lớp phân loại theo ba loại - “nhanh”, “trung bình” và “khó”. Bạn không cần phải tạo lại và đào tạo lại mô hình từ đầu! Bạn chỉ có thể bắt đầu từ các tính năng trung gian của mô hình trước đó vì bạn có quyền truy cập vào chúng.

```python
# layers[4] is our intermediate Dense layer.
features = model.layers[4].output
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)

new_model = keras.Model(
    inputs=[title, text_body, tags], outputs=[priority, department, difficulty]
)
```

[Liệt kê 7.13](#listing-7-13): Tạo mô hình mới bằng cách sử dụng lại các đầu ra của lớp trung gian

Hãy vẽ mô hình mới của chúng ta, như trong hình 7.4:

```python
keras.utils.plot_model(
    new_model,
    "updated_ticket_classifier.png",
    show_shapes=True,
    show_layer_names=True,
)
```

![](../images/ch07/updated_ticket_classifier.d0baf1fe.png)

[Figure 7.4](#figure-7-4): Plot of our new model

### Phân lớp lớp Model

Mẫu xây dựng mô hình cuối cùng mà bạn nên biết là mẫu nâng cao nhất: phân lớp `Model`. Bạn đã học trong chương 3 cách phân lớp lớp `Layer` để tạo các lớp tùy chỉnh. Phân lớp `Model` khá giống nhau:

* Trong phương thức `__init__`, xác định các lớp mà mô hình sẽ sử dụng.
* Trong phương thức `call`, xác định chuyển tiếp của mô hình,
sử dụng lại các lớp đã tạo trước đó.
* Khởi tạo lớp con của bạn và gọi nó dựa trên dữ liệu để tạo trọng số của nó.

#### Viết lại ví dụ trước của chúng ta dưới dạng mô hình được phân lớp

Chúng ta hãy xem một ví dụ đơn giản: chúng ta sẽ triển khai lại mô hình quản lý phiếu hỗ trợ khách hàng bằng cách sử dụng lớp con `Model`.

```python
class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        # Don't forget to call the super constructor!
        super().__init__()
        # Defines sublayers in the constructor
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(
            num_departments, activation="softmax"
        )

    # Defines the forward pass in the call() method
    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department
```

[Liệt kê 7.14](#listing-7-14): Một mô hình phân lớp đơn giản

Khi bạn đã xác định mô hình, bạn có thể khởi tạo nó. Lưu ý rằng nó sẽ chỉ tạo trọng số trong lần đầu tiên bạn gọi nó trên một số dữ liệu - giống như các lớp con `Layer`:

```python
model = CustomerTicketModel(num_departments=4)

priority, department = model(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)
```

Cho đến nay, mọi thứ trông rất giống với phân lớp `Layer`, một quy trình làm việc mà bạn đã gặp ở chương 3. Vậy thì sự khác biệt giữa lớp con `Layer` và lớp con `Model` là gì? Rất đơn giản: *layer* là khối xây dựng mà bạn sử dụng để tạo mô hình và *model* là đối tượng cấp cao nhất mà bạn sẽ thực sự đào tạo, xuất để suy luận, v.v. Nói tóm lại, `Model` có phương thức `fit()`, `evaluate()` và `predict()`. Các lớp không. Ngoài ra, hai lớp này hầu như giống hệt nhau (một điểm khác biệt nữa là bạn có thể *lưu* một mô hình vào một tệp trên đĩa - chúng tôi sẽ đề cập đến điều này trong một số phần).

Bạn có thể biên dịch và huấn luyện một lớp con `Model` giống như mô hình Tuần tự hoặc Hàm:

```python
model.compile(
    optimizer="adam",
    # The structure of what you pass as the loss and metrics must match
    # exactly what gets returned by call() — since we returned a list
    # of two elements, so should loss and metrics be lists of two
    # elements.
    loss=["mean_squared_error", "sparse_categorical_crossentropy"],
    metrics=[["mean_absolute_error"], ["accuracy"]],
)
model.fit(
    # The structure of the input data must match exactly what is
    # expected by the call() method, and the structure of the target
    # data must match exactly what gets returned by the call() method.
    # Here, the input data must be a dict with three keys (title,
    # text_body, and tags) and the target data must be a list of two
    # elements.
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    [priority_data, department_data],
    epochs=1,
)
model.evaluate(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    [priority_data, department_data],
)
priority_preds, department_preds = model.predict(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)
```

Quy trình làm việc phân lớp con `Model` là cách linh hoạt nhất để xây dựng một mô hình: nó cho phép bạn xây dựng các mô hình không thể được biểu thị dưới dạng biểu đồ tuần hoàn có hướng của các lớp — ví dụ: hãy tưởng tượng một mô hình trong đó phương thức `call()` sử dụng các lớp bên trong vòng lặp `for` hoặc thậm chí gọi chúng theo cách đệ quy. Mọi thứ đều có thể xảy ra - bạn là người chịu trách nhiệm.

#### Cẩn thận: Những mô hình phân lớp nào không hỗ trợ

Sự tự do này phải trả giá: với các mô hình được phân lớp, bạn chịu trách nhiệm về logic mô hình nhiều hơn, điều đó có nghĩa là bề mặt lỗi tiềm ẩn của bạn sẽ lớn hơn nhiều. Kết quả là bạn sẽ có nhiều công việc sửa lỗi hơn để làm. Bạn đang phát triển một đối tượng Python mới, không chỉ ghép các khối LEGO lại với nhau.

Các mô hình chức năng và phân lớp cũng khác nhau đáng kể về bản chất: Mô hình chức năng là một cấu trúc dữ liệu rõ ràng — một biểu đồ gồm các lớp mà bạn có thể xem, kiểm tra và sửa đổi. Trong khi đó, mô hình phân lớp là một đoạn mã byte — một lớp Python có phương thức `call()` chứa mã thô. Đây là nguồn gốc của tính linh hoạt của quy trình làm việc phân lớp — bạn có thể mã hóa bất kỳ chức năng nào bạn thích — nhưng nó đưa ra những hạn chế mới.

Ví dụ: vì cách các lớp được kết nối với nhau bị ẩn bên trong phần thân của phương thức `call()` nên bạn không thể truy cập thông tin đó. Việc gọi `summary()` sẽ không hiển thị kết nối lớp và bạn không thể vẽ sơ đồ cấu trúc liên kết mô hình thông qua `plot_model()`. Tương tự, nếu bạn có một mô hình được phân lớp, bạn không thể truy cập các nút của biểu đồ các lớp để thực hiện trích xuất đặc điểm - vì đơn giản là không có biểu đồ. Sau khi mô hình được khởi tạo, quá trình chuyển tiếp của nó sẽ trở thành một hộp đen hoàn chỉnh.

### Trộn và kết hợp các thành phần khác nhau

Điều quan trọng là việc chọn một trong các mẫu này — mô hình `Tuần tự`, API chức năng, phân lớp con `Mô hình` — không khóa bạn khỏi các mẫu khác. Tất cả các mô hình trong API Keras có thể tương tác trơn tru với nhau, cho dù chúng là mô hình Tuần tự, mô hình Chức năng hay mô hình phân lớp được viết từ đầu. Tất cả đều là một phần của cùng một quy trình công việc. Chẳng hạn, bạn có thể sử dụng lớp hoặc mô hình được phân lớp con trong mô hình Chức năng.

```python
class Classifier(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = "sigmoid"
        else:
            num_units = num_classes
            activation = "softmax"
        self.dense = layers.Dense(num_units, activation=activation)

    def call(self, inputs):
        return self.dense(inputs)

inputs = keras.Input(shape=(3,))
features = layers.Dense(64, activation="relu")(inputs)
outputs = Classifier(num_classes=10)(features)
model = keras.Model(inputs=inputs, outputs=outputs)
```

[Liệt kê 7.15](#listing-7-15): Tạo một mô hình Chức năng bao gồm một mô hình được phân lớp con

Ngược lại, bạn có thể sử dụng Mô hình chức năng như một phần của lớp hoặc mô hình được phân lớp con.

```python
inputs = keras.Input(shape=(64,))
outputs = layers.Dense(1, activation="sigmoid")(inputs)
binary_classifier = keras.Model(inputs=inputs, outputs=outputs)

class MyModel(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        self.dense = layers.Dense(64, activation="relu")
        self.classifier = binary_classifier

    def call(self, inputs):
        features = self.dense(inputs)
        return self.classifier(features)

model = MyModel()
```

[Liệt kê 7.16](#listing-7-16): Tạo một mô hình được phân lớp con bao gồm một mô hình Chức năng

### Hãy nhớ: Sử dụng đúng công cụ cho công việc

Bạn đã tìm hiểu về các quy trình công việc đa dạng để xây dựng mô hình Keras, từ quy trình công việc đơn giản nhất — mô hình `Tuần tự` — đến mô hình nâng cao nhất, phân lớp mô hình. Khi nào bạn nên sử dụng cái này hơn cái kia? Mỗi cái đều có ưu và nhược điểm - hãy chọn cái phù hợp nhất cho công việc hiện tại.

Nói chung, API chức năng cung cấp cho bạn sự cân bằng khá tốt giữa tính dễ sử dụng và tính linh hoạt. Nó cũng cung cấp cho bạn quyền truy cập trực tiếp vào kết nối lớp, tính năng này rất mạnh mẽ cho các trường hợp sử dụng như vẽ sơ đồ mô hình hoặc trích xuất tính năng. Nếu bạn *có thể* sử dụng API chức năng — tức là, nếu mô hình của bạn có thể được biểu thị dưới dạng biểu đồ tuần hoàn có hướng của các lớp — thì chúng tôi khuyên bạn nên sử dụng API này thay vì phân lớp mô hình.

Trong tương lai, tất cả các ví dụ trong cuốn sách này sẽ sử dụng API chức năng - đơn giản vì tất cả các mô hình mà chúng tôi sẽ làm việc đều có thể biểu thị dưới dạng biểu đồ các lớp. Tuy nhiên, chúng ta sẽ sử dụng thường xuyên các lớp con. Nói chung, việc sử dụng các mô hình Chức năng bao gồm các lớp được phân lớp con mang lại lợi ích tốt nhất cho cả hai thế giới: tính linh hoạt phát triển cao trong khi vẫn giữ được các ưu điểm của API Chức năng.

## Sử dụng các vòng đào tạo và đánh giá tích hợp

Nguyên tắc tiết lộ dần dần về độ phức tạp - khả năng tiếp cận nhiều quy trình công việc từ dễ dàng đến linh hoạt tùy ý, từng bước một - cũng áp dụng cho đào tạo mô hình. Keras cung cấp cho bạn các quy trình công việc khác nhau cho các mô hình đào tạo — có thể đơn giản như gọi `fit()` trên dữ liệu của bạn hoặc nâng cao như viết thuật toán đào tạo mới từ đầu.

Bạn đã quen với quy trình làm việc `compile()`, `fit()`, `evaluate()`, `predict()`. Xin nhắc lại, nó trông giống như danh sách sau đây.

```python
from keras.datasets import mnist

# Creates a model. (We factor this into a separate function so as to
# reuse it later.)
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

# Loads your data, reserving some for validation
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()
# Compiles the model by specifying its optimizer, the loss function to
# minimize, and metrics to monitor
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
# Uses `fit()` to train the model, optionally providing validation data
# to monitor performance on unseen data
model.fit(
    train_images,
    train_labels,
    epochs=3,
    validation_data=(val_images, val_labels),
)
# Uses `evaluate()` to compute the loss and metrics on new data
test_metrics = model.evaluate(test_images, test_labels)
# Uses `predict()` to compute classification probabilities on new data
predictions = model.predict(test_images)
```

[Liệt kê 7.17](#listing-7-17): Quy trình công việc tiêu chuẩn: `compile()`, `fit()`, `evaluate()`, `predict()`

Có một số cách bạn có thể tùy chỉnh quy trình làm việc đơn giản này:

* Bằng cách cung cấp số liệu tùy chỉnh của riêng bạn
* Bằng cách chuyển *callbacks* đến phương thức `fit()` để lên lịch hành động
được thực hiện tại các điểm cụ thể trong quá trình đào tạo

Chúng ta hãy nhìn vào những điều này.

### Viết số liệu của riêng bạn

Số liệu là chìa khóa để đo lường hiệu suất mô hình của bạn — cụ thể là để đo lường sự khác biệt giữa hiệu suất của mô hình trên dữ liệu huấn luyện và hiệu suất của mô hình trên dữ liệu thử nghiệm. Các số liệu thường được sử dụng để phân loại và hồi quy đã là một phần của mô-đun `keras.metrics` tích hợp sẵn — hầu hết thời gian, đó là những gì bạn sẽ sử dụng. Nhưng nếu bạn đang làm bất cứ điều gì khác thường, bạn sẽ cần có khả năng viết các số liệu của riêng mình. Thật đơn giản!

Số liệu Keras là một lớp con của lớp `keras.metrics.Metric`. Tương tự như các lớp, một số liệu có trạng thái bên trong được lưu trữ trong các biến Keras. Không giống như các lớp, các biến này không được cập nhật thông qua lan truyền ngược, vì vậy bạn phải tự viết logic cập nhật trạng thái - điều này xảy ra trong phương thức `update_state()`. Ví dụ: đây là một số liệu tùy chỉnh đơn giản để đo lỗi bình phương trung bình gốc (RMSE).

```python
from keras import ops

# Subclasses the Metric class
class RootMeanSquaredError(keras.metrics.Metric):
    # Defines the state variables in the constructor. Like for layers,
    # you have access to the add_weight() method.
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(
            name="total_samples", initializer="zeros"
        )

    # Implements the state update logic in update_state(). The y_true
    # argument is the targets (or labels) for one batch, while y_pred
    # represents the corresponding predictions from the model. To match
    # our MNIST model, we expect categorical predictions and integer
    # labels. You can ignore the sample_weight argument; we won't use
    # it here.
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.one_hot(y_true, num_classes=ops.shape(y_pred)[1])
        mse = ops.sum(ops.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = ops.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)
```

[Liệt kê 7.18](#listing-7-18): Triển khai một số liệu tùy chỉnh bằng cách phân lớp lớp `Số liệu`

Bạn sử dụng phương thức `result()` để trả về giá trị hiện tại của số liệu:

```python
    def result(self):
        return ops.sqrt(self.mse_sum / self.total_samples)
```

Đồng thời, bạn cũng cần tìm ra cách đặt lại trạng thái số liệu mà không cần phải khởi tạo lại nó — điều này cho phép sử dụng cùng một đối tượng số liệu trong các giai đoạn đào tạo khác nhau hoặc trong cả quá trình đào tạo và đánh giá. Bạn thực hiện việc này trong phương thức `reset_state()`:

```python
    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0.)
```

Số liệu tùy chỉnh có thể được sử dụng giống như số liệu tích hợp. Hãy thử nghiệm số liệu của riêng chúng tôi:

```python
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", RootMeanSquaredError()],
)
model.fit(
    train_images,
    train_labels,
    epochs=3,
    validation_data=(val_images, val_labels),
)
test_metrics = model.evaluate(test_images, test_labels)
```

Bây giờ bạn có thể thấy thanh tiến trình `fit()` hiển thị RMSE của mô hình của bạn.

### Sử dụng lệnh gọi lại

Việc khởi chạy quá trình huấn luyện trên một tập dữ liệu lớn trong hàng chục kỷ nguyên bằng cách sử dụng `model.fit()` có thể hơi giống việc phóng một chiếc máy bay giấy: vượt qua xung động ban đầu, bạn không có bất kỳ quyền kiểm soát nào đối với quỹ đạo hoặc điểm hạ cánh của nó. Nếu bạn muốn tránh những kết quả xấu (và do đó lãng phí máy bay giấy), tốt hơn hết bạn nên sử dụng, không phải máy bay giấy mà là một máy bay không người lái có thể cảm nhận được môi trường, gửi dữ liệu trở lại người điều khiển và tự động đưa ra quyết định điều khiển dựa trên trạng thái hiện tại của nó. API Keras *callbacks* sẽ giúp bạn chuyển đổi lệnh gọi của mình thành `model.fit()` từ máy bay giấy thành máy bay không người lái thông minh, tự động có thể tự xem xét nội tâm và thực hiện hành động một cách chủ động.

*Gọi lại* là một đối tượng (một phiên bản lớp triển khai các phương thức cụ thể) được chuyển đến mô hình trong lệnh gọi `fit()` và được mô hình gọi tại nhiều điểm khác nhau trong quá trình đào tạo. Nó có quyền truy cập vào tất cả dữ liệu có sẵn về trạng thái và hiệu suất của mô hình, đồng thời có thể thực hiện hành động: làm gián đoạn quá trình đào tạo, lưu mô hình, tải một tập trọng lượng khác hoặc thay đổi trạng thái của mô hình.

Dưới đây là một số ví dụ về cách bạn có thể sử dụng lệnh gọi lại:

* *Điểm kiểm tra mô hình* — Lưu trạng thái hiện tại của mô hình
ở những thời điểm khác nhau trong quá trình đào tạo.

* *Dừng sớm* — Gián đoạn đào tạo khi mất xác nhận
không còn cải thiện nữa (và tất nhiên, lưu lại mô hình tốt nhất thu được trong quá trình đào tạo).

* *Tự động điều chỉnh giá trị của các tham số nhất định trong quá trình đào tạo* —
Chẳng hạn như tốc độ học tập của trình tối ưu hóa.

* *Ghi nhật ký các số liệu đào tạo và xác thực trong quá trình đào tạo hoặc trực quan hóa các cách trình bày mà mô hình đã học khi chúng được cập nhật* —
Thanh tiến trình `fit()` mà bạn quen thuộc thực chất là một lệnh gọi lại!

Mô-đun `keras.callbacks` bao gồm một số lệnh gọi lại tích hợp (đây không phải là danh sách đầy đủ):

```python
keras.callbacks.ModelCheckpoint
keras.callbacks.EarlyStopping
keras.callbacks.LearningRateScheduler
keras.callbacks.ReduceLROnPlateau
keras.callbacks.CSVLogger
```

Hãy xem xét hai trong số chúng để bạn biết cách sử dụng chúng: `EarlyStopping` và `ModelCheckpoint`.

#### Lệnh gọi lại EarlyStopping và ModelCheckpoint

Khi đào tạo một người mẫu, có rất nhiều điều bạn không thể đoán trước được ngay từ đầu. Đặc biệt, bạn không thể biết cần bao nhiêu kỷ nguyên để đạt được mức mất xác thực tối ưu. Các ví dụ của chúng tôi cho đến nay đã áp dụng chiến lược đào tạo cho đủ kỷ nguyên mà bạn bắt đầu trang bị quá mức, sử dụng lần chạy đầu tiên để tìm ra số kỷ nguyên tối ưu và cuối cùng khởi chạy một đợt đào tạo mới từ đầu bằng cách sử dụng số lượng tối ưu này. Tất nhiên, cách tiếp cận này là lãng phí. Cách tốt hơn nhiều để xử lý vấn đề này là ngừng đào tạo khi bạn đo lường được rằng mức độ mất xác thực không còn được cải thiện nữa. Điều này có thể đạt được bằng cách sử dụng lệnh gọi lại `EarlyStopping`.

Lệnh gọi lại `EarlyStopping` làm gián đoạn quá trình đào tạo khi chỉ số mục tiêu đang được theo dõi đã ngừng cải thiện trong một số kỷ nguyên cố định. Ví dụ: lệnh gọi lại này cho phép bạn tạm dừng quá trình đào tạo ngay khi bạn bắt đầu trang bị quá mức, do đó tránh phải đào tạo lại mô hình của mình cho số lượng kỷ nguyên nhỏ hơn. Lệnh gọi lại này thường được sử dụng kết hợp với `ModelCheckpoint`, cho phép bạn liên tục lưu mô hình trong quá trình đào tạo (và, tùy ý, chỉ lưu mô hình tốt nhất hiện tại cho đến nay: phiên bản của mô hình đã đạt được hiệu suất tốt nhất khi kết thúc một kỷ nguyên).

```python
# Callbacks are passed to the model via the callbacks argument in
# fit(), which takes a list of callbacks. You can pass any number of
# callbacks.
callbacks_list = [
    # Interrupts training when improvement stops
    keras.callbacks.EarlyStopping(
        # Monitors the model's validation accuracy
        monitor="accuracy",
        # Interrupts training when accuracy has stopped improving for
        # more than one epoch (that is, two epochs)
        patience=1,
    ),
    # Saves the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath="checkpoint_path.keras",
        # These two arguments mean you won't overwrite the model file
        # unless val_loss has improved, which allows you to keep the
        # best model seen during training.
        monitor="val_loss",
        save_best_only=True,
    ),
]
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    # You monitor accuracy, so it should be part of the model's
    # metrics.
    metrics=["accuracy"],
)
# Because the callback will monitor validation loss and validation
# accuracy, you need to pass validation_data to the call to fit().
model.fit(
    train_images,
    train_labels,
    epochs=10,
    callbacks=callbacks_list,
    validation_data=(val_images, val_labels),
)
```

[Liệt kê 7.19](#listing-7-19): Sử dụng đối số `callbacks` trong phương thức `fit()`

Lưu ý rằng bạn luôn có thể lưu mô hình theo cách thủ công sau khi đào tạo - chỉ cần gọi `model.save("my_checkpoint_path.keras")`. Để tải lại mô hình bạn đã lưu, hãy sử dụng

```python
model = keras.models.load_model("checkpoint_path.keras")
```

### Viết cuộc gọi lại của riêng bạn

Nếu bạn cần thực hiện một hành động cụ thể trong quá trình đào tạo mà một trong các lệnh gọi lại tích hợp không có trong đó, bạn có thể viết lệnh gọi lại của riêng mình. Lệnh gọi lại được triển khai bằng cách phân lớp con `keras.callbacks.Callback`. Sau đó, bạn có thể triển khai bất kỳ số phương thức nào được đặt tên rõ ràng sau đây, được gọi ở nhiều thời điểm khác nhau trong quá trình đào tạo:

```python
# Called at the start of every epoch
on_epoch_begin(epoch, logs)
# Called at the end of every epoch
on_epoch_end(epoch, logs)
# Called right before processing each batch
on_batch_begin(batch, logs)
# Called right after processing each batch
on_batch_end(batch, logs)
# Called at the start of training
on_train_begin(logs)
# Called at the end of training
on_train_end(logs)
```

Tất cả các phương thức này đều được gọi với đối số `logs`, là một từ điển chứa thông tin về đợt, kỷ nguyên hoặc lần chạy đào tạo trước đó: số liệu đào tạo và xác thực, v.v. Các phương thức `on_epoch_*` và `on_batch_*` cũng lấy chỉ mục epoch hoặc batch làm đối số đầu tiên (một số nguyên).

Dưới đây là một ví dụ gọi lại đơn giản giúp lưu danh sách các giá trị tổn thất trên mỗi đợt trong quá trình đào tạo và vẽ đồ thị các giá trị này vào cuối mỗi kỷ nguyên.

```python
from matplotlib import pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(
            range(len(self.per_batch_losses)),
            self.per_batch_losses,
            label="Training loss for each batch",
        )
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_at_epoch_{epoch}", dpi=300)
        self.per_batch_losses = []
```

[Liệt kê 7.20](#listing-7-20): Tạo một lệnh gọi lại tùy chỉnh bằng cách phân lớp con lớp `Callback`

Hãy lái thử nó:

```python
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_images,
    train_labels,
    epochs=10,
    callbacks=[LossHistory()],
    validation_data=(val_images, val_labels),
)
```

Chúng ta có được đồ thị giống như hình 7.5.

![](../images/ch07/loss_history_callback_example.1e42f6b2.png)

[Figure 7.5](#figure-7-5): The output of our custom history-plotting callback

### Giám sát và trực quan hóa với TensorBoard

Để thực hiện nghiên cứu tốt hoặc phát triển các mô hình tốt, bạn cần có phản hồi phong phú và thường xuyên về những gì đang diễn ra bên trong các mô hình trong quá trình thử nghiệm. Đó là mục đích của việc chạy thử nghiệm: để có được thông tin về mức độ hoạt động của một mô hình - càng nhiều thông tin càng tốt. Tiến bộ là một quá trình lặp đi lặp lại, một vòng lặp: bạn bắt đầu với một ý tưởng và thể hiện nó như một thử nghiệm, cố gắng xác nhận hoặc vô hiệu hóa ý tưởng của mình. Bạn chạy thử nghiệm này và xử lý thông tin mà nó tạo ra, như minh họa trong hình 7.6. Điều này truyền cảm hứng cho ý tưởng tiếp theo của bạn. Bạn càng thực hiện được nhiều vòng lặp này thì ý tưởng của bạn càng trở nên tinh tế và mạnh mẽ hơn. Keras giúp bạn đi từ ý tưởng đến thử nghiệm trong thời gian ít nhất có thể và GPU nhanh có thể giúp bạn đi từ thử nghiệm đến kết quả nhanh nhất có thể. Nhưng còn việc xử lý kết quả thí nghiệm thì sao? Đó là lúc TensorBoard xuất hiện.

![](../images/ch07/the_loop_of_progress.df126e89.png)

[Figure 7.6](#figure-7-6): The loop of progress

TensorBoard là một ứng dụng dựa trên trình duyệt mà bạn có thể chạy cục bộ. Đó là cách tốt nhất để giám sát mọi thứ diễn ra bên trong mô hình của bạn trong quá trình đào tạo. Với TensorBoard, bạn có thể

* Theo dõi trực quan các số liệu trong quá trình đào tạo
* Trực quan hóa kiến ​​trúc mô hình của bạn
* Trực quan hóa biểu đồ kích hoạt và độ dốc
* Khám phá các phần nhúng trong 3D

Nếu bạn đang theo dõi nhiều thông tin hơn là chỉ tổn thất cuối cùng của mô hình, bạn có thể phát triển tầm nhìn rõ ràng hơn về những gì mô hình làm và không làm, đồng thời bạn có thể đạt được tiến bộ nhanh hơn.

Cách dễ nhất để sử dụng TensorBoard với mô hình Keras và phương thức `fit()` là lệnh gọi lại `keras.callbacks.TensorBoard`. Trong trường hợp đơn giản nhất, chỉ cần chỉ định nơi bạn muốn lệnh gọi lại ghi nhật ký và bạn đã sẵn sàng:

```python
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

tensorboard = keras.callbacks.TensorBoard(
    log_dir="/full_path_to_your_log_dir",
)
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(val_images, val_labels),
    callbacks=[tensorboard],
)
```

Khi mô hình bắt đầu chạy, nó sẽ ghi nhật ký tại vị trí đích. Nếu bạn đang chạy tập lệnh Python trên máy cục bộ, thì bạn có thể khởi chạy máy chủ TensorBoard cục bộ bằng lệnh sau (lưu ý rằng tệp thực thi `tensorboard` sẽ có sẵn nếu bạn đã cài đặt TensorFlow qua `pip`; nếu không, bạn có thể cài đặt TensorBoard theo cách thủ công thông qua `pip install tensorboard`):

```python
tensorboard --logdir /full_path_to_your_log_dir
```

Sau đó, bạn có thể điều hướng đến URL mà lệnh trả về để truy cập vào giao diện TensorBoard.

Nếu đang chạy tập lệnh trong sổ ghi chép Colab, thì bạn có thể chạy phiên bản TensorBoard được nhúng như một phần của sổ ghi chép bằng cách sử dụng các lệnh sau:

```python
%load_ext tensorboard
%tensorboard --logdir /full_path_to_your_log_dir
```

Trong giao diện TensorBoard, bạn sẽ có thể theo dõi biểu đồ trực tiếp về các số liệu đánh giá và đào tạo của mình, như minh họa trong hình 7.7.

![](../images/ch07/tensorboard.aec6cc75.png)

[Figure 7.7](#figure-7-7): TensorBoard can be used for easy monitoring of training and evaluation metrics.

## Viết vòng đào tạo và đánh giá của riêng bạn

Quy trình làm việc `fit()` tạo ra sự cân bằng tuyệt vời giữa tính dễ sử dụng và tính linh hoạt. Đó là những gì bạn sẽ sử dụng hầu hết thời gian. Tuy nhiên, nó không nhằm hỗ trợ mọi thứ mà một nhà nghiên cứu deep learning có thể muốn thực hiện — ngay cả với các số liệu tùy chỉnh, tổn thất tùy chỉnh và lệnh gọi lại tùy chỉnh.

Xét cho cùng, quy trình làm việc `fit()` tích hợp chỉ tập trung vào *học có giám sát*: một thiết lập trong đó có các *mục tiêu* đã biết (còn được gọi là *nhãn* hoặc *chú thích*) được liên kết với dữ liệu đầu vào của bạn và nơi bạn tính toán tổn thất của mình dưới dạng hàm của các mục tiêu này và dự đoán của mô hình. Tuy nhiên, không phải mọi hình thức học máy đều thuộc loại này. Có những thiết lập khác không có mục tiêu rõ ràng, chẳng hạn như *học tập sáng tạo* (mà chúng tôi sẽ giới thiệu trong chương 16), *học tập tự giám sát* (trong đó các mục tiêu được lấy từ đầu vào) hoặc *học tập tăng cường* (trong đó việc học được thúc đẩy bởi những “phần thưởng” không thường xuyên - giống như huấn luyện một con chó). Và ngay cả khi bạn đang thực hiện học tập có giám sát thường xuyên, với tư cách là một nhà nghiên cứu, bạn có thể muốn thêm một số tính năng mới lạ đòi hỏi tính linh hoạt ở mức độ thấp.

Bất cứ khi nào bạn thấy mình ở trong tình huống mà `fit()` tích hợp sẵn là không đủ, bạn sẽ cần phải viết logic đào tạo tùy chỉnh của riêng mình. Bạn đã thấy các ví dụ đơn giản về vòng lặp đào tạo cấp thấp trong chương 2 và 3. Xin nhắc lại, nội dung của một vòng đào tạo điển hình trông như thế này:

1. Chạy “chuyển tiếp” (tính toán đầu ra của mô hình) để lấy giá trị tổn thất cho lô dữ liệu hiện tại. 2. Truy xuất độ dốc của tổn thất liên quan đến trọng số của mô hình. 3. Cập nhật trọng số của mô hình để giảm giá trị tổn thất trên lô dữ liệu hiện tại.

Các bước này được lặp lại cho nhiều đợt nếu cần thiết. Về cơ bản đây là những gì `fit()` thực hiện. Trong phần này, bạn sẽ học cách triển khai lại `fit()` từ đầu, điều này sẽ cung cấp cho bạn tất cả kiến ​​thức cần thiết để viết bất kỳ thuật toán đào tạo nào mà bạn có thể nghĩ ra.

Chúng ta hãy đi qua các chi tiết. Trong một số phần tiếp theo, bạn sẽ nỗ lực viết một vòng đào tạo tùy chỉnh đầy đủ tính năng trong TensorFlow, PyTorch và JAX.

### Đào tạo so với suy luận

Trong các ví dụ về vòng lặp đào tạo cấp thấp mà bạn đã thấy cho đến nay, bước 1 (chuyển tiếp) được thực hiện thông qua `predictions = model(inputs)` và bước 2 (truy xuất độ dốc được tính toán bằng băng chuyển màu) được thực hiện thông qua API dành riêng cho chương trình phụ trợ, chẳng hạn như

* `gradient = tape.gradient(loss, model.weights)` trong TensorFlow
* `loss.backward()` trong PyTorch
* `jax.value_and_grad()` trong JAX

Trong trường hợp chung, thực tế có hai điểm tinh tế bạn cần tính đến.

Một số lớp Keras, chẳng hạn như lớp `Dropout`, có các hành vi khác nhau trong quá trình *huấn luyện* và trong khi *suy luận* (khi bạn sử dụng chúng để tạo dự đoán). Các lớp như vậy hiển thị một đối số Boolean `huấn luyện` trong phương thức `call()` của chúng. Việc gọi `dropout(inputs, Training=True)` sẽ loại bỏ một số mục kích hoạt, trong khi gọi `dropout(inputs, Training=False)` không làm gì cả. Nói rộng ra, các Mô hình chức năng và Mô hình tuần tự cũng hiển thị đối số `đào tạo` này trong các phương thức `call()` của chúng. Hãy nhớ chuyển `training=True` khi bạn gọi mô hình Keras trong quá trình chuyển tiếp! Do đó, chuyển tiếp của chúng tôi trở thành `dự đoán = mô hình (đầu vào, đào tạo = Đúng)`.

Ngoài ra, hãy lưu ý rằng khi truy xuất độ dốc của trọng số của mô hình, bạn không nên sử dụng `model.weights` mà thay vào đó là `model.trainable_weights`. Thật vậy, các lớp và mô hình có hai loại trọng số:

* *Trọng số có thể huấn luyện*, nghĩa là được cập nhật thông qua lan truyền ngược để giảm thiểu
mất mô hình, chẳng hạn như hạt nhân và độ lệch của lớp `Dense`.
* *Trọng lượng không thể huấn luyện được*, có nghĩa là sẽ được cập nhật trong quá trình chuyển tiếp
bởi các lớp sở hữu chúng. Ví dụ: nếu bạn muốn có một lớp tùy chỉnh để giữ bộ đếm
cho đến nay nó đã xử lý bao nhiêu lô, thông tin đó sẽ được lưu trữ
với trọng lượng không thể huấn luyện được và ở mỗi đợt, lớp của bạn sẽ tăng
đếm từng cái một.

Trong số các lớp tích hợp sẵn của Keras, lớp duy nhất có các trọng số không thể huấn luyện được là lớp `BatchNormalization`, lớp này chúng tôi sẽ giới thiệu trong chương 9. Lớp `BatchNormalization` cần các trọng số không thể huấn luyện được để theo dõi thông tin về giá trị trung bình và độ lệch chuẩn của dữ liệu đi qua nó, để thực hiện phép tính gần đúng trực tuyến của *chuẩn hóa tính năng* (một khái niệm bạn đã học trong chương 4 và 6).

### Viết các hàm bước đào tạo tùy chỉnh

Khi tính đến hai chi tiết này, bước đào tạo học có giám sát sẽ trông như thế này trong mã giả:

```python
def train_step(inputs, targets):
    # Runs the forward pass
    predictions = model(inputs, training=True)
    # Computes the loss for the current batch
    loss = loss_fn(targets, predictions)
    # Retrieves the gradients of the loss with regard to the model's
    # trainable weights This function doesn't actually exist!
    gradients = get_gradients_of(loss, wrt=model.trainable_weights)
    # Updates the model's trainable weights based on the gradients
    optimizer.apply(gradients, model.trainable_weights)
```

Đoạn mã này là mã giả chứ không phải mã thực vì nó bao gồm một hàm tưởng tượng, `get_gradients_of()`. Trên thực tế, việc truy xuất độ dốc được thực hiện theo cách dành riêng cho chương trình phụ trợ hiện tại của bạn — JAX, TensorFlow hoặc PyTorch.

Hãy sử dụng những gì bạn đã học về từng khung trong chương 3 để triển khai phiên bản thực của hàm `train_step()` này. Chúng ta sẽ bắt đầu với TensorFlow và PyTorch vì hai công cụ này giúp công việc tương đối dễ dàng nên đây là nơi tốt để bắt đầu. Chúng ta sẽ kết thúc bằng JAX, nó phức tạp hơn một chút.

#### Chức năng bước đào tạo TensorFlow

TensorFlow cho phép bạn viết mã trông khá giống đoạn mã giả của chúng tôi. Sự khác biệt duy nhất là việc chuyển tiếp của bạn sẽ diễn ra bên trong phạm vi `GradientTape`. Sau đó, bạn có thể sử dụng đối tượng `tape` để truy xuất độ dốc:

```python
import tensorflow as tf

model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

def train_step(inputs, targets):
    # Opens a GradientTape
    with tf.GradientTape() as tape:
        # Runs the forward pass
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    # Retrieves the gradients from the tape
    gradients = tape.gradient(loss, model.trainable_weights)
    # Updates the model's trainable weights based on the gradients
    optimizer.apply(gradients, model.trainable_weights)
    return loss
```

Hãy chạy nó trong một bước duy nhất:

```python
batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = train_step(inputs, targets)
```

Đủ dễ dàng! Tiếp theo hãy làm PyTorch.

#### Chức năng bước đào tạo PyTorch

Khi bạn sử dụng chương trình phụ trợ PyTorch, tất cả các lớp và mô hình Keras của bạn sẽ kế thừa từ lớp `torch.nn.Module` của PyTorch và hiển thị API `Module` gốc. Do đó, mô hình của bạn, các trọng số có thể huấn luyện và tensor mất mát của bạn đều nhận biết lẫn nhau và tương tác thông qua ba phương thức: `loss.backward()`, `weight.value.grad` và `model.zero_grad()`.

Xin nhắc lại ở chương 3, mô hình tinh thần mà bạn cần ghi nhớ là:

* Với mỗi lần chuyển tiếp, PyTorch sẽ xây dựng một biểu đồ tính toán một lần để duy trì
theo dõi phép tính vừa xảy ra.
* Gọi `.backward()` trên bất kỳ nút vô hướng đã cho nào của biểu đồ này (như mất mát của bạn) sẽ khiến biểu đồ chạy lùi
bắt đầu từ nút đó, tự động điền thuộc tính `tensor.grad` trên tất cả các tensor có liên quan (nếu chúng đáp ứng `requires_grad=True`),
chứa gradient của nút đầu ra đối với tensor đó. Đặc biệt, nó sẽ điền vào thuộc tính `grad`
các thông số có thể đào tạo của bạn.
* Để xóa nội dung của thuộc tính `tensor.grad` đó, bạn nên gọi `tensor.grad = None` trên tất cả
tensor của bạn. Bởi vì sẽ hơi cồng kềnh khi thực hiện việc này trên tất cả các biến mô hình riêng lẻ,
bạn chỉ có thể thực hiện điều đó ở cấp độ mô hình thông qua `model.zero_grad()` — lệnh gọi `zero_grad()` sẽ truyền tới tất cả các biến
được theo dõi bởi mô hình. Việc xóa độ dốc là rất quan trọng vì các lệnh gọi tới `backward()` có tính chất bổ sung: nếu bạn không
xóa độ dốc ở mỗi bước, các giá trị độ dốc sẽ tích lũy và việc đào tạo sẽ không tiếp tục.

Hãy xâu chuỗi các bước sau:

```python
import torch

model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

def train_step(inputs, targets):
    # Runs the forward pass
    predictions = model(inputs, training=True)
    loss = loss_fn(targets, predictions)
    # Runs the backward pass, populating gradient values
    loss.backward()
    # Recovers the gradient associated with each trainable variable.
    # That weight.value is the PyTorch tensor that contains the
    # variable's value.
    gradients = [weight.value.grad for weight in model.trainable_weights]
    # Updates the model's trainable weights based on the gradients.
    # This must be done in a no_grad() scope.
    with torch.no_grad():
        optimizer.apply(gradients, model.trainable_weights)
    # Don't forget to clear the gradients!
    model.zero_grad()
    return loss
```

Hãy chạy nó trong một bước duy nhất:

```python
batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = train_step(inputs, targets)
```

Điều đó không quá khó! Bây giờ, hãy chuyển sang JAX.

#### Hàm bước huấn luyện JAX

Khi nói đến mã đào tạo cấp thấp, JAX có xu hướng phức tạp nhất trong ba phần phụ trợ vì tính chất hoàn toàn không trạng thái của nó. Tính không trạng thái làm cho JAX có hiệu suất cao và có khả năng mở rộng, giúp nó có thể tuân thủ quá trình biên dịch và tối ưu hóa hiệu suất tự động. Tuy nhiên, việc viết mã phi trạng thái yêu cầu bạn phải vượt qua một số vòng.

Vì hàm gradient có được thông qua lập trình meta, trước tiên bạn cần xác định hàm trả về tổn thất của mình. Hơn nữa, hàm này cần phải không trạng thái, vì vậy nó cần lấy làm đối số tất cả các biến mà nó sẽ sử dụng và nó cần trả về giá trị của bất kỳ biến nào mà nó đã cập nhật. Bạn có nhớ những trọng lượng không thể huấn luyện được có thể được sửa đổi trong quá trình chuyển tiếp không? Đó là những biến chúng ta cần trả về.

Để làm việc với mô hình lập trình không trạng thái của JAX dễ dàng hơn, các mô hình Keras cung cấp một phương thức chuyển tiếp không trạng thái: phương thức `stateless_call()`. Nó hoạt động giống như `__call__`, ngoại trừ điều đó

* Ngoài ra, nó lấy đầu vào là trọng số có thể huấn luyện và trọng số không thể huấn luyện của mô hình.
đến các đối số `đầu vào` và `đào tạo`.
* Nó trả về các trọng số không thể huấn luyện được cập nhật của mô hình, bên cạnh các kết quả đầu ra của mô hình.

Nó hoạt động như thế này:

```python
outputs, non_trainable_weights = model.stateless_call(
    trainable_weights, non_trainable_weights, inputs
)
```

Chúng ta có thể sử dụng `stateless_call()` để triển khai hàm mất JAX của mình. Vì hàm loss cũng tính toán các bản cập nhật cho tất cả các biến không thể huấn luyện được nên chúng tôi đặt tên nó là `compute_loss_and_updates()`:

```python
model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# Gradients are computed for the entries in the first argument
# (trainable_variables here)
def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, inputs, targets
):
    # Calls stateless_call
    outputs, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, inputs, training=True
    )
    loss = loss_fn(targets, outputs)
    # Returns the scalar loss value and the updated non-trainable
    # weights
    return loss, non_trainable_variables
```

Khi có hàm `compute_loss_and_updates()` này, chúng ta có thể chuyển nó đến `jax.value_and_grad` để tính toán độ dốc:

```python
import jax

grad_fn = jax.value_and_grad(fn)
loss, gradients = grad_fn(...)
```

Bây giờ, chỉ có một vấn đề nhỏ. Cả `jax.grad()` và `jax.value_and_grad()` đều yêu cầu `fn` chỉ trả về giá trị vô hướng. Hàm `compute_loss_and_updates()` của chúng tôi trả về giá trị vô hướng làm đầu ra đầu tiên, nhưng nó cũng trả về giá trị mới cho các trọng số không thể huấn luyện được. Bạn có nhớ những gì bạn đã học ở chương 3 không? Giải pháp là chuyển đối số `has_aux` cho `grad()` hoặc `value_and_grad()`, như thế này:

```python
import jax

grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
```

Bạn sẽ sử dụng nó như thế này:

```python
(loss, non_trainable_weights), gradients = grad_fn(
    trainable_variables, non_trainable_variables, inputs, targets
)
```

Được rồi, đó là rất nhiều JAXiness. Nhưng bây giờ chúng ta đã có hầu hết mọi thứ cần thiết để tập hợp bước đào tạo JAX của mình. Chúng ta chỉ cần mảnh ghép cuối cùng: `optimizer.apply()`.

Khi viết bước đào tạo cơ bản đầu tiên trong TensorFlow ở đầu chương 2, bạn đã viết một hàm bước cập nhật trông như thế này:

```python
learning_rate = 1e-3

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign(w - g * learning_rate)
```

Điều này tương ứng với những gì trình tối ưu hóa `keras.optimizers.SGD` sẽ làm. Tuy nhiên, mọi trình tối ưu hóa khác trong API Keras phức tạp hơn thế một chút và theo dõi các biến phụ trợ giúp tăng tốc quá trình đào tạo — đặc biệt, hầu hết các trình tối ưu hóa đều sử dụng một số dạng *động lượng* mà bạn đã học ở chương 2. Các biến bổ sung này được cập nhật ở mỗi bước đào tạo và trong thế giới JAX, điều đó có nghĩa là bạn cần sử dụng một hàm không trạng thái lấy các biến này làm đối số và trả về giá trị mới của chúng.

Để thực hiện việc này dễ dàng, Keras cung cấp phương thức `stateless_apply()` trên tất cả các trình tối ưu hóa. Nó hoạt động như thế này:

```python
trainable_variables, optimizer_variables = optimizer.stateless_apply(
    optimizer_variables, grads, trainable_variables
)
```

Bây giờ, chúng ta đã có đủ để tập hợp một bước đào tạo từ đầu đến cuối:

```python
optimizer = keras.optimizers.Adam()
optimizer.build(model.trainable_variables)

# The state is part of the function arguments.
def train_step(state, inputs, targets):
    # Unpacks the state
    (trainable_variables, non_trainable_variables, optimizer_variables) = state
    # Computes gradients and updates to non-trainable variables
    (loss, non_trainable_variables), grads = grad_fn(
        trainable_variables, non_trainable_variables, inputs, targets
    )
    # Updates trainable variables and optimizer variables
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    return loss, (
        # Returns the updated state alongside the loss
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )
```

Hãy chạy nó trong một bước duy nhất:

```python
batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]

trainable_variables = [v.value for v in model.trainable_variables]
non_trainable_variables = [v.value for v in model.non_trainable_variables]
optimizer_variables = [v.value for v in optimizer.variables]

state = (trainable_variables, non_trainable_variables, optimizer_variables)
loss, state = train_step(state, inputs, targets)
```

Nó chắc chắn hiệu quả hơn TensorFlow và PyTorch một chút, nhưng lợi ích về tốc độ và khả năng mở rộng của JAX nhiều hơn là bù đắp cho điều đó.

Tiếp theo, chúng ta hãy xem xét một yếu tố quan trọng khác của vòng đào tạo tùy chỉnh: *số liệu*.

### Việc sử dụng số liệu ở mức độ thấp

Trong vòng đào tạo cấp thấp, bạn có thể sẽ muốn sử dụng các số liệu Keras (cho dù là số liệu tùy chỉnh hay số liệu tích hợp sẵn). Bạn đã tìm hiểu về API chỉ số: chỉ cần gọi `update_state(y_true, y_pred)` cho từng nhóm mục tiêu và dự đoán, sau đó sử dụng `result()` để truy vấn giá trị chỉ số hiện tại:

```python
from keras import ops

metric = keras.metrics.SparseCategoricalAccuracy()
targets = ops.array([0, 1, 2])
predictions = ops.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
metric.update_state(targets, predictions)
current_result = metric.result()
print(f"result: {current_result:.2f}")
```

Bạn cũng có thể cần theo dõi giá trị trung bình của giá trị vô hướng, chẳng hạn như độ mất của mô hình. Bạn có thể thực hiện việc này thông qua số liệu `keras.metrics.Mean`:

```python
values = ops.array([0, 1, 2, 3, 4])
mean_tracker = keras.metrics.Mean()
for value in values:
    mean_tracker.update_state(value)
print(f"Mean of values: {mean_tracker.result():.2f}")
```

Hãy nhớ sử dụng `metric.reset_state()` khi bạn muốn đặt lại kết quả hiện tại (khi bắt đầu giai đoạn huấn luyện hoặc khi bắt đầu đánh giá).

Bây giờ, nếu bạn đang sử dụng JAX, bạn không thể sử dụng các phương thức sửa đổi trạng thái như `update_state()` hoặc `reset()` bên trong một hàm không trạng thái. Thay vào đó, bạn có thể sử dụng API số liệu không trạng thái, tương tự như các phương thức `model.stateless_call()` và `optimizer.stateless_apply()` mà bạn đã tìm hiểu. Đây là cách nó hoạt động:

```python
metric = keras.metrics.SparseCategoricalAccuracy()
targets = ops.array([0, 1, 2])
predictions = ops.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Gets the metric's state variables
metric_variables = metric.variables
# Gets updated values for the metric's state
metric_variables = metric.stateless_update_state(
    metric_variables, targets, predictions
)
# Computes the metric value corresponding to the current state
current_result = metric.stateless_result(metric_variables)
print(f"result: {current_result:.2f}")

# Gets blank variable values for the metric
metric_variables = metric.stateless_reset_state()
```

### Sử dụng fit() với vòng lặp đào tạo tùy chỉnh

Trong các phần trước, chúng ta đã viết logic huấn luyện của riêng mình hoàn toàn từ đầu. Làm như vậy sẽ mang lại cho bạn sự linh hoạt nhất, nhưng cuối cùng bạn sẽ viết rất nhiều mã, đồng thời bỏ lỡ nhiều tính năng tiện lợi của `fit()`, chẳng hạn như lệnh gọi lại, tối ưu hóa hiệu suất hoặc hỗ trợ tích hợp cho đào tạo phân tán.

Điều gì sẽ xảy ra nếu bạn cần một thuật toán đào tạo tùy chỉnh nhưng vẫn muốn sử dụng sức mạnh của vòng đào tạo Keras tích hợp sẵn? Thực sự có một điểm trung gian giữa `fit()` và vòng lặp đào tạo được viết từ đầu: bạn có thể cung cấp chức năng bước đào tạo tùy chỉnh và để khung thực hiện phần còn lại.

Bạn có thể thực hiện việc này bằng cách ghi đè phương thức `train_step()` của lớp `Model`. Đây là hàm được gọi bởi `fit()` cho mỗi lô dữ liệu. Sau đó, bạn sẽ có thể gọi `fit()` như bình thường — và nó sẽ chạy thuật toán học tập của riêng bạn.

Đây là cách nó hoạt động:

* Tạo một lớp mới có các lớp con `keras.Model`.
* Ghi đè phương thức `train_step()`. Nội dung của nó là
gần giống với những gì chúng tôi đã sử dụng trong phần trước.
* Trả về tên số liệu ánh xạ từ điển (bao gồm cả phần mất) về giá trị hiện tại của chúng
giá trị.

Lưu ý những điều sau:

* Mẫu này không ngăn cản bạn xây dựng các mô hình bằng Chức năng
API. Bạn có thể làm điều này cho dù bạn đang xây dựng các mô hình `Tuần tự`, API chức năng
mô hình hoặc mô hình phân lớp.
* Bạn không cần sử dụng trang trí `@tf.function` hoặc `@jax.jit`
khi bạn ghi đè `train_step()` — framework sẽ thực hiện việc đó cho bạn.

#### Tùy chỉnh fit() với TensorFlow

Hãy bắt đầu bằng cách mã hóa bước huấn luyện TensorFlow tùy chỉnh:

```python
import keras
from keras import layers

loss_fn = keras.losses.SparseCategoricalCrossentropy()
# This metric object will be used to track the average of per-batch
# losses during training and evaluation.
loss_tracker = keras.metrics.Mean(name="loss")

class CustomModel(keras.Model):
    # Overrides the train_step() method
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            # We use self(inputs, training=True) instead of
            # model(inputs, training=True) since our model is the class
            # itself.
            predictions = self(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply(gradients, self.trainable_weights)

        # Updates the loss tracker metric that tracks the average of
        # the loss
        loss_tracker.update_state(loss)
        # Returns the average loss so far by querying the loss tracker
        # metric
        return {"loss": loss_tracker.result()}

    # Listing the loss tracker metric in the model.metrics property
    # enables the model to automatically call reset_state() on it at
    # the start of each epoch and at the start of a call to evaluate()
    # — so you don't have to do it by hand. Any metric you would like
    # to reset across epochs should be listed here.
    @property
    def metrics(self):
        return [loss_tracker]
```

[Liệt kê 7.21](#listing-7-21): Tùy chỉnh `fit()`: Phiên bản TensorFlow

Bây giờ chúng ta có thể khởi tạo mô hình tùy chỉnh của mình, biên dịch nó (chúng ta chỉ chuyển trình tối ưu hóa, vì phần mất mát đã được xác định bên ngoài mô hình) và huấn luyện nó bằng cách sử dụng `fit()` như bình thường.

Hãy đặt định nghĩa mô hình vào hàm có thể tái sử dụng của chính nó:

```python
def get_custom_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam())
    return model
```

Hãy thử xem:

```python
model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

#### Tùy chỉnh fit() với PyTorch

Tiếp theo, phiên bản PyTorch:

```python
import keras
from keras import layers

loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")

class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        # Runs the forward pass
        predictions = self(inputs, training=True)
        loss = loss_fn(targets, predictions)

        # Retrieves the gradients
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            # Updates weights
            self.optimizer.apply(gradients, trainable_weights)

        # Updates loss tracker metric
        loss_tracker.update_state(loss)
        # Returns the average loss so far by querying the loss tracker
        # metric
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        return [loss_tracker]
```

Hãy thử nó:

```python
model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

#### Tùy chỉnh fit() với JAX

Cuối cùng, hãy viết phiên bản JAX. Trước tiên, chúng ta cần xác định phương thức `compute_loss_and_updates()`, tương tự như hàm `compute_loss_and_updates()` mà chúng tôi đã sử dụng trong ví dụ về bước đào tạo tùy chỉnh của mình:

```python
import keras
from keras import layers

loss_fn = keras.losses.SparseCategoricalCrossentropy()

class CustomModel(keras.Model):
    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        inputs,
        targets,
        training=False,
    ):
        predictions, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            inputs,
            training=training,
        )
        loss = loss_fn(targets, predictions)
        # Returns both the loss and the updated non-trainable variables
        return loss, non_trainable_variables
```

Lưu ý rằng chúng tôi không tính toán mức trung bình động của tổn thất như chúng tôi đã làm đối với hai phần phụ trợ còn lại. Thay vào đó, chúng tôi chỉ trả về giá trị tổn thất trên mỗi lô, điều này ít hữu ích hơn. Chúng tôi làm điều này để đơn giản hóa việc quản lý trạng thái chỉ số trong ví dụ: mã sẽ rất dài dòng nếu chúng tôi đưa nó vào (bạn sẽ tìm hiểu về quản lý chỉ số trong phần tiếp theo):

```python
    def train_step(self, state, data):
        # Unpacks the state. metrics_variables are part of it, although
        # we won't use them here.
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        inputs, targets = data

        # Gets the gradient function
        grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates, has_aux=True
        )

        # Computes gradients and updates to non-trainable variables
        (loss, non_trainable_variables), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            inputs,
            targets,
            training=True,
        )

        # Updates trainable variables and optimizer variables
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        # We aren't computing a moving average of the loss, instead
        # returning the per-batch value.
        logs = {"loss": loss}
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )
        # Returns metric logs and updated state variables
        return logs, state
```

Hãy thử nó:

```python
model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

### Xử lý số liệu trong một chuyến tàu tùy chỉnh\_step()

Cuối cùng, còn `loss` và `số liệu` mà bạn có thể chuyển đến `compile()` thì sao? Sau khi gọi `compile()`, bạn sẽ có quyền truy cập vào

* `self.compute_loss` — Điều này kết hợp hàm mất mát mà bạn đã truyền cho `compile()`
cùng với các tổn thất chính quy có thể được thêm vào bởi các lớp nhất định.
* `self.metrics` — Danh sách các số liệu bạn đã chuyển đến `compile()`. Ghi chú
rằng nó cũng bao gồm một số liệu theo dõi sự mất mát.

#### xử lý số liệu train\_step() với TensorFlow

Đây là giao diện của TensorFlow:

```python
import keras
from keras import layers

class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            # Computes the loss via self.compute_loss
            loss = self.compute_loss(y=targets, y_pred=predictions)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply(gradients, self.trainable_weights)

        # Updates the model's metrics, including the one that tracks
        # the loss
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(targets, predictions)

        # Returns a dict mapping metric names to their current value
        return {m.name: m.result() for m in self.metrics}
```

Hãy thử nó:

```python
def get_custom_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

#### xử lý số liệu train\_step() với PyTorch

Và đây là giao diện của PyTorch - đó chính xác là sự thay đổi mã giống nhau!

```python
import keras
from keras import layers

class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        predictions = self(inputs, training=True)
        loss = self.compute_loss(y=targets, y_pred=predictions)

        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}
```

Hãy xem nó chạy như thế nào:

```python
def get_custom_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

#### xử lý số liệu train\_step() bằng JAX

Cuối cùng, đây là giao diện của JAX. Để bắt đầu, bạn có thể sử dụng `compute_loss()` trong phương thức `compute_loss_and_updates()` của mình để tính phần mất được truyền cho `compile()`:

```python
import keras
from keras import layers

class CustomModel(keras.Model):
    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        inputs,
        targets,
        training=False,
    ):
        predictions, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            inputs,
            training=training,
        )
        loss = self.compute_loss(y=targets, y_pred=predictions)
        return loss, (predictions, non_trainable_variables)
```

Tiếp theo: quản lý số liệu. Như thường lệ, nó phức tạp hơn một chút do các yêu cầu về trạng thái không trạng thái của JAX:

```python
    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            # Metric variables are part of the state.
            metrics_variables,
        ) = state
        inputs, targets = data

        grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates, has_aux=True
        )

        (loss, (predictions, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            inputs,
            targets,
            training=True,
        )
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        new_metrics_vars = []
        logs = {}
        # Iterates over metrics
        for metric in self.metrics:
            num_prev = len(new_metrics_vars)
            num_current = len(metric.variables)
            # Grabs the variables of the current metrics
            current_vars = metrics_variables[num_prev : num_prev + num_current]
            # Updates the metric's state
            if metric.name == "loss":
                current_vars = metric.stateless_update_state(current_vars, loss)
            else:
                current_vars = metric.stateless_update_state(
                    current_vars, targets, predictions
                )
            # Stores the results in the logs dict
            logs[metric.name] = metric.stateless_result(current_vars)
            new_metrics_vars += current_vars

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            # Returns the new metrics variables as part of the state
            new_metrics_vars,
        )
        return logs, state
```

Đó là rất nhiều thông tin, nhưng đến giờ bạn đã biết đủ để sử dụng Keras để làm hầu hết mọi việc!

## Bản tóm tắt

* Keras cung cấp nhiều quy trình công việc khác nhau, dựa trên nguyên tắc
*tiết lộ dần dần về độ phức tạp*. Tất cả đều tương tác trơn tru.
* Bạn có thể xây dựng mô hình thông qua lớp `Sequential`, thông qua API chức năng,
hoặc bằng cách phân lớp lớp `Model`. Hầu hết thời gian, bạn sẽ sử dụng API chức năng.
* Cách đơn giản nhất để huấn luyện và đánh giá một mô hình là thông qua các phương thức `fit()` và `evaluate()` mặc định.
* Lệnh gọi lại Keras cung cấp một cách đơn giản để giám sát các mô hình trong khi bạn gọi tới `fit()`
và tự động thực hiện hành động dựa trên trạng thái của mô hình.
* Bạn cũng có thể kiểm soát hoàn toàn những gì `fit()` thực hiện bằng cách ghi đè phương thức `train_step()`,
sử dụng API từ chương trình phụ trợ bạn chọn — JAX, TensorFlow hoặc PyTorch.
* Ngoài `fit()`, bạn cũng có thể viết các vòng đào tạo của riêng mình hoàn toàn từ đầu,
theo cách phụ trợ gốc. Điều này rất hữu ích cho các nhà nghiên cứu triển khai các thuật toán đào tạo hoàn toàn mới.

#### **Slide**

<div class="pdf-container" style="margin-bottom: 20px; width: 100%; height: 85vh;">
  <iframe src="TaiLieu/slideDL/Chapter07.pdf#view=FitH" width="100%" height="100%" style="border: none;"></iframe>
</div>

#### ** 💻 Luyện tập **

<div class="practice-container" style="background: #f8faff; border: 1px solid #cce0ff; border-radius: 8px; padding: 20px; margin-top: 15px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 15px;">
    <h3 style="margin:0; color: #1a73e8; display:flex; align-items:center; gap:8px;">🚀 Bài tập Thực hành Jupyter Notebook</h3>
    <div class="lang-toggle" style="display:flex; gap: 5px;">
      <button id="btn-vn" onclick="togglePracticeLang('VN')" style="opacity: 1; cursor:pointer; background:white; border:1px solid #cce0ff; border-radius:4px; padding:5px 10px; font-weight:600; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">🇻🇳 VN</button>
      <button id="btn-en" onclick="togglePracticeLang('EN')" style="opacity: 0.5; cursor:pointer; background:white; border:1px solid #cce0ff; border-radius:4px; padding:5px 10px; font-weight:600; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">🇬🇧 EN</button>
    </div>
  </div>
  <p style="margin-bottom: 15px;">Dưới đây là sổ tay (notebook) chứa mã nguồn Python thực hành cho chương này. Bạn có thể mở trực tiếp trên Google Colab để chạy thử nghiệm, hoặc tải file về máy.</p>
  <ul id="notebook-list-VN" style="display:block; list-style-type: none; padding-left: 0;">
    <li style="margin-bottom: 15px; padding: 15px; background: white; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
      <strong style="font-size:16px;">Chapter07 Deep Dive Keras</strong><br>
      <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
        <a href="https://colab.research.google.com/github/tranthanhthangbmt/WebDeepLearning_2026/blob/main/codeJupyterNotebook/deep-learning-with-python-notebooks-master/chapter07_deep_dive_keras_VN.ipynb" target="_blank" style="background: #fbbc04; color: #fff; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(251,188,4,0.3);">🔥 Mở trên Google Colab</a>
        <a href="codeJupyterNotebook/deep-learning-with-python-notebooks-master/chapter07_deep_dive_keras_VN.ipynb" download style="background: #1a73e8; color: white; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(26,115,232,0.3);">💾 Tải file .ipynb về máy</a>
      </div>
    </li>
  </ul>
  <ul id="notebook-list-EN" style="display:none; list-style-type: none; padding-left: 0;">
    <li style="margin-bottom: 15px; padding: 15px; background: white; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
      <strong style="font-size:16px;">Chapter07 Deep Dive Keras</strong><br>
      <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
        <a href="https://colab.research.google.com/github/tranthanhthangbmt/WebDeepLearning_2026/blob/main/codeJupyterNotebook/deep-learning-with-python-notebooks-master/chapter07_deep-dive-keras_EN.ipynb" target="_blank" style="background: #fbbc04; color: #fff; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(251,188,4,0.3);">🔥 Mở trên Google Colab</a>
        <a href="codeJupyterNotebook/deep-learning-with-python-notebooks-master/chapter07_deep-dive-keras_EN.ipynb" download style="background: #1a73e8; color: white; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(26,115,232,0.3);">💾 Tải file .ipynb về máy</a>
      </div>
    </li>
  </ul>
</div>


#### ** 🎥 Video **

<iframe src="TaiLieu/Video/Chapter_07/index.html" width="100%" height="600px" style="border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" allowfullscreen></iframe>

#### ** 📝 Bài tập Trắc nghiệm **

<iframe src="quizzes/Chapter07/index.html" style="width: 100%; min-height: 700px; border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></iframe>

<!-- tabs:end -->
