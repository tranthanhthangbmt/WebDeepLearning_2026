<!-- tabs:start -->

#### **Tiếng Anh (English)**

# Chapter 20: Conclusions

This chapter covers

* Important takeaways from this book
* Resources for learning further and applying your skills in practice

We’ll start with a bird’s-eye view of what you should take away from this book.
This should refresh your memory regarding some of the concepts you’ve learned.
Then we’ll give you a short list of
resources and strategies for learning further about machine learning
and staying up to date with new advances.

Becoming an effective AI practitioner
is a journey, and finishing this book is merely your first step on it. I
want to make sure you realize this and are properly equipped to take the next
steps of this journey on your own.

## Key concepts in review

This section briefly synthesizes key takeaways from this book. If you ever
need a quick refresher to help you recall what you’ve learned, you can read
these few pages.

### Various approaches to artificial intelligence

First, deep learning isn’t synonymous with artificial intelligence (AI), or even with machine
learning:

* *Artificial intelligence* (AI) is an ancient, broad field that can
  generally be understood as “all attempts to automate human cognitive processes.”
  This can range from the very basic,
  such as an Excel spreadsheet, to the very advanced, like a humanoid robot that
  can walk and talk.

* *Machine learning* is a specific subfield of AI that aims at automatically
  developing programs (called *models*) purely from exposure to training data.
  This process of turning data into a program is called *learning*. Although
  machine learning has been around for a long time, it only started to take off
  in the 1990s, before becoming the dominant form of AI in the 2000s.

* *Deep learning* is one of many branches of machine learning, where the
  models are long chains of geometric transformations, applied one after the other.
  These operations are structured into modules called *layers*: deep learning
  models are typically stacks of layers — or, more generally, graphs of layers.
  These layers are parameterized by *weights*, which are the parameters learned
  during training. The *knowledge* of a model is stored in its weights, and the
  process of learning consists of finding “good values” for these weights —
  values that minimize a *loss function*. Because the chain of geometric transformations
  considered is differentiable, updating the weights to minimize the loss function
  is done efficiently via *gradient descent*.

* *Generative AI* is a specific subset of deep learning, where models
  are capable of generating text, images, videos, or sound. These models tend to
  be very large — billions of parameters. They’re trained in a self-supervised
  manner; that is, they’re trained to reconstruct artificially
  missing or corrupted parts of an input —
  for instance, denoising images, predicting the next word in a sentence, and so on.
  This learning process enables the models to learn sophisticated “maps” (embedding manifolds)
  of their input space, which can be used for sampling new inputs. These models have launched
  AI into its “consumer” era with the rise of products like ChatGPT or Midjourney.

Even though deep learning is just one among many approaches to machine
learning, it isn’t on an equal footing with the others. Deep learning is a
breakout success. Here’s why.

### What makes deep learning special within the field of machine learning

In the span of only a few years, deep
learning has achieved tremendous breakthroughs across a wide range of tasks
that have been historically perceived as extremely difficult for computers,
especially in the area of machine perception: extracting useful information
from images, videos, sound, and more. Given sufficient training data (in
particular, training data appropriately labeled by humans), deep learning makes it
possible to extract from perceptual data almost anything a human could.
Hence, it’s sometimes said that deep learning has “solved perception” —
although that’s true only for a fairly narrow definition of perception.

Due to its unprecedented technical successes, deep learning has singlehandedly
brought about the third and by far the largest *AI summer*: a period of
intense interest, investment, and hype in the field of AI. As this book is
being written, we’re in the middle of it. Whether this period will end in the
near future and what happens after it ends are topics of debate. One thing
is certain: in stark contrast with previous AI summers, deep learning has
provided enormous business value to both large and small technology companies
and has become a huge consumer success,
enabling human-level speech recognition, chatbot assistants, photorealistic image
generation, human-level machine translation, and more. The hype may
(and likely will) recede, but the sustained economic and technological impact
of deep learning will remain. In that sense, deep learning could be analogous
to the internet: it may be overly hyped up for a few years, but in the longer
term, it will still be a major revolution that will transform our economy and
our lives.

One reason I’m particularly optimistic about deep learning is that
even if we were to make no further technological progress in the next decade,
deploying existing algorithms to every applicable problem would be a game changer for most
industries. Deep learning is nothing short of a revolution, and progress is
currently happening at an incredibly fast rate due to an exponential
investment in resources and headcount. From where we stand, the future looks
bright, although short-term expectations are somewhat overoptimistic;
deploying deep learning to the full extent of its potential will likely
take multiple decades.

### How to think about deep learning

The most surprising thing about deep learning is
how simple it is. Fifteen years ago, no one expected that we would achieve such
amazing results on machine-perception and natural language processing
problems by using simple parametric models trained with gradient descent.
Now it turns out that all you need is
sufficiently large parametric models trained with gradient descent on
sufficiently many examples. As Feynman once said about the universe, “It’s not
complicated,
it’s just a lot of it.”[[1]](#footnote-1)

In deep learning, everything is a vector; that is, everything is
a *point* in a *geometric space*. Model inputs (text, images, and so on)
and targets are first *vectorized* — turned into an initial input vector space
and target vector space. Each layer in a deep learning model operates one
simple geometric transformation on the data that goes through it. Together,
the chain of layers in the model forms one complex geometric transformation,
broken down into a series of simple ones. This complex transformation attempts
to map the input space to the target space, one point at a time. This
transformation is parameterized by the weights of the layers, which are
iteratively updated based on how well the model is currently performing. A key
characteristic of this geometric transformation is that it must be
*differentiable*, which is required for us to be able to learn its
parameters via gradient descent. Intuitively, this means the geometric
morphing from inputs to outputs must be smooth and continuous — a significant
constraint.

The entire process of applying this complex geometric transformation to the
input data can be visualized in 3D by imagining a person trying to uncrumple a
paper ball: the crumpled paper ball is the manifold of the input data that the
model starts with. Each movement operated by the person on the paper ball is
similar to a simple geometric transformation operated by one layer. The full
uncrumpling gesture sequence is the complex transformation of the entire
model. Deep learning models are mathematical machines for uncrumpling
complicated manifolds of high-dimensional data.

That’s the magic of deep learning — turning meaning into vectors, into geometric
spaces, and then incrementally learning complex geometric transformations that
map one space to another. All you need are spaces of sufficiently high
dimensionality to capture the full scope of the relationships found
in the original data.

The whole thing hinges on two core ideas:
that *meaning is derived from the pairwise relationship between things*
(between words in a language, between pixels in an image, and so on) and that
*these relationships can be captured by a distance function*.
But note that whether the brain implements meaning
via geometric spaces is an entirely separate question. Vector spaces are
efficient to work with from a computational standpoint, but different data
structures for intelligence can easily be envisioned — in particular, graphs.
Neural networks initially emerged from the idea of using graphs as a way to
encode meaning, which is why they’re named *neural networks*; the surrounding
field of research used to be called *connectionism*. Nowadays, the name *neural
network* exists purely for historical reasons — it’s an extremely misleading name
because they’re neither neural nor networks. In particular, neural networks
have hardly anything to do with the brain. A more appropriate name would have
been *layered representations learning* or *hierarchical representations learning*,
or maybe even deep *differentiable models* or *chained geometric transforms*,
to emphasize the fact that continuous geometric space manipulation is at their core.

### Key enabling technologies

The technological revolution that’s
currently unfolding didn’t start with any single breakthrough invention.
Rather, like any other revolution, it’s the product of a vast accumulation of
enabling factors — slowly at first, and then suddenly. In the case of deep
learning, we can point out the following key factors:

* Incremental algorithmic innovations, first spread over two decades (starting
  with backpropagation) and then happening increasingly faster as more research
  effort was poured into deep learning after 2012. One major such
  breakthrough was the Transformer architecture in 2017.

* The availability of large amounts of image, video, and text data, which is a requirement
  to realize that sufficiently large models trained on sufficiently large
  data are all we need. This is, in turn, a by-product of the rise of the consumer
  internet and Moore’s law applied to storage media. Today, state-of-the-art language models
  are trained on a large fraction of the entire internet.

* The availability of fast, highly parallel computation hardware at a low price,
  especially the GPUs produced by NVIDIA — first gaming GPUs and then chips
  designed from the ground up for deep learning. Early on, NVIDIA CEO Jensen
  Huang took note of the deep learning boom and decided to bet the company’s
  future on it, which paid off in a big way.

* A complex stack of software layers that makes this computational power
  available to humans: the CUDA language, frameworks like TensorFlow, JAX, and PyTorch that do
  automatic differentiation, and Keras, which makes deep learning accessible to
  most people.

In the future, deep learning will not be used only by specialists such as researchers,
graduate students, and engineers with an academic profile; it will be a
tool in the toolbox of every developer, much like web technology today.
Everyone needs to build intelligent apps: just as every business today needs a
website, every product will need to intelligently make sense of user-generated
data. Bringing about this future will require us to build tools that make deep
learning radically easy to use and accessible to anyone with basic coding
abilities. Keras has been the first major step in that direction.

### The universal machine learning workflow

Having access to an extremely powerful tool
for creating models that map any input space to any target space is great, but
the difficult part of the machine learning workflow is often everything that
comes before designing and training such models (and, for production models,
what comes after, as well). Understanding the problem domain to be able
to determine what to attempt to predict, given what data, and how to measure
success is a prerequisite for any successful application of machine learning,
and it isn’t something that advanced tools like Keras and TensorFlow can help
you with. As a reminder, here’s a quick summary of the typical
machine learning workflow as described in chapter 6:

* *Define the problem.* What data is available, and what are you trying to
  predict? Will you need to collect more data or hire people to manually label a
  dataset?
* *Identify a way to reliably measure success on your goal.* For simple tasks,
  this may be prediction accuracy, but in many cases, it will require
  sophisticated, domain-specific metrics.
* *Prepare the validation process that you’ll use to evaluate your models.* In
  particular, you should define a training set, a validation set, and a test
  set. The validation-set and test-set labels shouldn’t leak into the training
  data: for instance, with temporal prediction, the validation and test data
  should be posterior to the training data.
* *Vectorize the data by turning it into vectors and preprocessing it in a way that makes it more easily approachable by a neural network (normalization and so on).*
* *Develop a first model that beats a trivial common-sense baseline, thus demonstrating that machine learning can work on your problem.* This may not
  always be the case!
* *Gradually refine your model architecture by tuning hyperparameters and adding regularization.* Make changes based on performance on the validation data only,
  not the test data or the training data. Remember that you should get your
  model to overfit (thus identifying a model capacity level that’s greater than
  you need) and only then begin to add regularization or downsize your model.
  Beware of validation-set overfitting when tuning hyperparameters — the fact
  that your hyperparameters may end up being overspecialized to the validation
  set. Avoiding this is the purpose of having a separate test set!
* *Deploy your final model in production — as a web API, as part of a JavaScript or C++ application, on an embedded device, etc.* Keep monitoring its performance
  on real-world data and use your findings to refine the next iteration of the model!

### Key network architectures

The families of network architectures that you should be familiar with after
reading this book are *densely connected networks*, *convolutional networks*,
*recurrent networks*, *Diffusion Models*, and *Transformers*. Each type of model
is meant for specific data modalities: a network architecture encodes
*assumptions* about the structure of the data — a *hypothesis space* within which
the search for a good model will proceed. Whether a given architecture will work
on a given problem depends entirely on the match between the structure of the
data and the assumptions of the network architecture.

These different network types can easily be combined to achieve larger
multimodal models, much as you combine LEGO bricks. In a way, deep learning
layers are LEGO bricks for information processing. Table 20.1 shows a quick
overview of the mapping between input and output modalities and the appropriate
network architectures.

| Input | Output | Model |
| --- | --- | --- |
| Vector data | Class probability, Regression value | Densely connected network |
| Timeseries data | Class probability, Regression value | RNN, Transformer |
| Images | Class probability, Regression value | ConvNet |
| Text | Class probability, Regression value | Transformer |
| Text, Images | Text | Transformer |
| Text, Images | Images | VAE, Diffusion Model |

[Table 20.1](#table-20-1): Model architectures for different data types

Now let’s quickly review the specificities of each network architecture.

#### Densely connected networks

A densely connected network is a stack of
`Dense` layers, meant to process vector data (where each sample is a vector
of numerical or categorical attributes).
Such networks assume no specific structure in the input features: they’re called
*densely connected* because the units of a `Dense` layer are connected to
every other unit. The layer attempts to map relationships between any two
input features; this is unlike a 2D convolution layer, for instance, which
only looks at *local* relationships.

Densely connected networks are most commonly used for categorical data (for
example, where the input features are lists of attributes), such as the Boston
Housing Price dataset used in chapter 4. They’re also used as the final
classification or regression stage of most networks. For instance, the
ConvNets covered in chapter 8 typically end with one or two `Dense` layers,
and so do the recurrent networks in chapter 13.

Remember, to perform *binary classification*, end your stack of layers with a
`Dense` layer with a single unit and a `sigmoid` activation, and use
`binary_crossentropy` as the loss. Your targets should be either 0 or 1:

```python
import keras
from keras import layers

inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
```

To perform *single-label categorical classification* (where each
sample has exactly one class, no
more), end your stack of layers with a `Dense` layer with a number of units
equal to the number of classes and a `softmax` activation.
If your targets are one-hot encoded, use `categorical_crossentropy`
as the loss; if they’re integers, use `sparse_categorical_ crossentropy`:

```python
inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
```

To perform *multilabel categorical classification* (where
each sample can have several classes), end your stack of
layers with a `Dense` layer with a number of units equal to the number of
classes and a `sigmoid` activation, and use
`binary_crossentropy` as the loss. Your targets should be k-hot encoded:

```python
inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
```

To perform *regression* toward a vector of continuous values,
end your stack of layers with a `Dense` layer with a number of units equal to
the number of values you’re trying to predict (often a single one, such as the
price of a house) and no activation. Various losses can be used for
regression — most commonly `mean_squared_error` (MSE):

```python
inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_values)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="mse")
```

#### ConvNets

Convolution layers look at
spatially local patterns by applying the same geometric transformation to
different spatial locations (*patches*) in an input tensor. This results in
representations that are *translation invariant*, making convolution layers
highly data efficient and modular. This idea is applicable to spaces
of any dimensionality: 1D (continuous sequences), 2D (images), 3D (volumes), and so on.
You can use the `Conv1D` layer to process
sequences, the `Conv2D` layer to process images, and the `Conv3D` layer to
process volumes. As a leaner, more efficient alternative to convolution layers,
you can also use *depthwise separable convolution* layers, such as `SeparableConv2D`.

*ConvNets*, or *convolutional networks*, consist of stacks of convolution and
max-pooling layers. The pooling layers let you spatially downsample the data,
which is required to keep feature maps to a reasonable size as the number of
features grows and to allow subsequent convolution layers to “see” a greater
spatial extent of the inputs. ConvNets are often ended with either a `Flatten`
operation or a global pooling layer, turning spatial feature maps into
vectors, followed by `Dense` layers to achieve
classification or regression.

Here’s a typical image-classification network (categorical classification, in
this case) using `SeparableConv2D` layers:

```python
inputs = keras.Input(shape=(height, width, channels))
x = layers.SeparableConv2D(32, 3, activation="relu")(inputs)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.SeparableConv2D(128, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.SeparableConv2D(128, 3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
```

When building a very deep ConvNet, it’s common to add
*batch normalization* layers as well as *residual connections*
— two architecture patterns that help
gradient information flow smoothly through the network.

#### Transformers

A Transformer looks at a set of vectors (such as word vectors)
and uses *neural attention*
to transform each vector into a representation that is aware of the *context*
provided by the other vectors in the set. When the set in question is an ordered
sequence, you can also use *positional encoding* to create Transformers
that can take into account both global context and word order, capable of
processing long text paragraphs much more effectively than RNNs or 1D ConvNets.

Transformers can be used for any set-processing or sequence-processing task,
including text classification, but they excel especially at
*sequence-to-sequence learning*, such as translating
paragraphs in a source language into a target language.

A sequence-to-sequence Transformer is made of two parts:

* A `TransformerEncoder` that turns
  an input vector sequence into a context-aware, order-aware output vector sequence
* A `TransformerDecoder` that takes the output of the `TransformerEncoder`, as
  well as a target sequence, and predicts what should come next in the target sequence

If you’re only processing a single sequence (or set) of vectors, you’d only
use the `TransformerEncoder`.

Following is a sequence-to-sequence Transformer for mapping
a source sequence to a target sequence (this setup could be used for machine translation
or question-answering, for instance):

```python
from keras_hub.layers import TokenAndPositionEmbedding
from keras_hub.layers import TransformerDecoder, TransformerEncoder

# Source sequence
encoder_inputs = keras.Input(shape=(src_seq_length,), dtype="int64")
x = TokenAndPositionEmbedding(vocab_size, src_seq_length, embed_dim)(
    encoder_inputs
)
encoder_outputs = TransformerEncoder(intermediate_dim=256, num_heads=8)(x)
# Target sequence so far
decoder_inputs = keras.Input(shape=(dst_seq_length,), dtype="int64")
x = TokenAndPositionEmbedding(vocab_size, dst_seq_length, embed_dim)(
    decoder_inputs
)
x = TransformerDecoder(intermediate_dim=256, num_heads=8)(x, encoder_outputs)
# Predictions for target sequence one step in the future
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
transformer.compile(optimizer="adamw", loss="categorical_crossentropy")
```

And this is a lone `TransformerEncoder` for binary classification of integer sequences:

```python
inputs = keras.Input(shape=(seq_length,), dtype="int64")
x = TokenAndPositionEmbedding(vocab_size, seq_length, embed_dim)(inputs)
x = TransformerEncoder(intermediate_dim=256, num_heads=8)(x)
x = layers.GlobalMaxPooling1D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="adamw", loss="binary_crossentropy")
```

#### Recurrent neural networks

*Recurrent neural networks* (RNNs) work by
processing sequences of inputs one timestep at a time and maintaining a state
throughout (a state is typically a vector or set of vectors).
They should be used preferentially over 1D
ConvNets in the case of sequences where patterns of interest aren’t invariant
by temporal translation (for instance, timeseries data where the recent past
is more important than the distant past).

Three RNN layers are available in Keras: `SimpleRNN`,
`GRU`, and `LSTM`. For most practical purposes, you should use either `GRU` or
`LSTM`. `LSTM` is the more powerful of the two but is also more expensive; you
can think of `GRU` as a simpler, cheaper alternative to it.

To stack multiple RNN layers on top of each other, each layer prior to
the last layer in the stack should return the full sequence of its outputs
(each input timestep will correspond to an output timestep); if you aren’t
stacking any further RNN layers, then it’s common to return only the last
output, which contains information about the entire sequence.

Following is a single RNN layer for binary classification of vector sequences:

```python
inputs = keras.Input(shape=(num_timesteps, num_features))
x = layers.LSTM(32)(inputs)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
```

And this is a stacked RNN layer for binary classification of vector sequences:

```python
inputs = keras.Input(shape=(num_timesteps, num_features))
x = layers.LSTM(32, return_sequences=True)(inputs)
x = layers.LSTM(32, return_sequences=True)(x)
x = layers.LSTM(32)(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
```

## Limitations of deep learning

Building deep learning models is like playing with LEGO bricks:
layers can be plugged together to map essentially anything to anything, given
that you have appropriate training data available and that the mapping is
achievable via a continuous geometric transformation of reasonable complexity.

Here’s the catch, though — this mapping is often not learnable in a way
that will generalize. Deep learning models operate like vast, interpolative databases of patterns.
Their pattern-matching strength is also their core weakness:

* *They fundamentally struggle to adapt to novelty.*
  Because their parameters are fixed after training, they can only retrieve or replicate patterns similar to their training data.
  Faced with inputs significantly outside this familiar distribution — no matter how simple the underlying task —
  their performance degrades drastically, as they lack mechanisms for fluid generalization beyond their memorized experience.
  This explains why even large models fail on novel tasks or simple variations of familiar problems — like ARC-AGI tasks.

* *They’re very sensitive to phrasing and other distractors.* Deep learning models exhibit high sensitivity to superficial
  variations in input presentation, such as minor phrasing changes (consider prompt sensitivity in LLMs)
  or imperceptible perturbations (consider adversarial examples in vision), indicating a lack of robust, human-like understanding.

* *They often can’t learn generalizable algorithms.* The continuous, geometric nature of deep learning models makes them fundamentally
  ill-suited for learning exact, discrete, step-by-step algorithms, such as those that are the bread and butter of classical computer science.
  Models approximate such processes through interpolation rather than implementing robust, generalizable procedures.

You should always resist the temptation to anthropomorphize deep learning models. Their performance is
built on pointwise statistical patterns rather than human-like experiential grounding,
making it brittle when encountering deviations from training data.

The narrative that simply scaling up model size and training data would lead to general intelligence
has proven insufficient. While scaling enhances performance on benchmarks that amount to memorization tests,
it fails to address the fundamental limitations of deep learning, which stem from the core paradigm of fitting static,
interpolative curves to data. Five years of exponential scaling of base LLMs haven’t overcome these constraints because the underlying approach remains unchanged.

By 2024, this realization spurred a transition toward test-time adaptation (TTA),
where models perform search or fine-tuning during the inference phase to adapt to novel problems.
While TTA methods have yielded major breakthroughs, such as OpenAI’s o3 surpassing human baseline on ARC-AGI-1 in late 2024,
this performance has come at an extreme computational cost.
Efficient, human-like adaptation is still a completely open problem,
and the slightly harder ARC-AGI-2 benchmark remains completely unsolved as of today.
We still need further conceptual advances beyond mere scaling or brute-force search.

## What might lie ahead

Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches.
While deep learning excels at *value-centric abstraction*, which enables pattern recognition and intuition,
it fundamentally lacks capabilities for *program-centric abstraction*, which underpins discrete reasoning, planning, and causal understanding.
Human intelligence seamlessly integrates both — future AI must do the same.

Future key developments may include

* *Hybrid models* — Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation)
  with deep learning modules (providing perception and intuition).
  These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.

* *Deep-learning guided program search* — Program synthesis — automatically discovering executable code that meets specifications —
  offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck.
  A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, combinatorial spaces of programs
  efficiently, much like human developers use experience and intuition to narrow down their choices.

* *Modular recombination and lifelong learning* — We’ll move away from monolithic, end-to-end models trained from scratch.
  Instead, future AI systems will use massive libraries of reusable, modular components that can be repurposed across many problems, acquired from experience.
  These libraries will feature both “geometric” (deep learning based) and “algorithmic” modules.
  When faced with a new problem, such AI systems will fetch relevant modules and dynamically recombine them into a new model adapted
  to the situation at hand. Whenever the system ends up developing a reusable component as a by-product of this problem-solving
  loop, the new component would get added to the library, becoming available for every future task the system might encounter.

Ultimately, developing AI that mirrors human-like fluid intelligence will require
blending continuous pattern recognition together with discrete, symbolic programs
and fully embracing the paradigm of on-the-fly adaptation.

## Staying up to date in a fast-moving field

As final parting words, I want to give you some pointers about how to keep
learning and updating your knowledge and skills after you’ve turned the last
page of this book. The field of modern deep learning, as we know it today, is
only a few years old, despite a long, slow prehistory stretching back decades.
With an exponential increase in financial resources and research headcount
since 2013, the field as a whole is now moving at a frenetic pace. What you’ve
learned in this book won’t stay relevant forever, and it isn’t all you’ll need
for the rest of your career.

Fortunately, there are plenty of free online resources that you can use to stay
up to date and expand your horizons. Here are a few.

### Practice on real-world problems using Kaggle

An effective way to acquire real-world experience is to try your hand at machine learning
competitions on Kaggle (<https://kaggle.com>). The only real way to learn is
through practice and actual coding — that’s the philosophy of this book, and
Kaggle competitions are the natural continuation of this. On Kaggle, you’ll
find an array of constantly renewed data science competitions, many of which
involve deep learning, prepared by companies interested in obtaining novel
solutions to some of their most challenging machine learning problems. Fairly
large monetary prizes are offered to top entrants.

By participating in a few competitions, maybe as part of a team, you’ll become
more familiar with the practical side of some of the advanced best practices
described in this book, especially hyperparameter tuning, avoiding
validation-set overfitting, and model ensembling.

### Read about the latest developments on arXiv

Deep learning research, in contrast with some other
scientific fields, takes place completely in the open. Papers are made
publicly and freely accessible as soon as they’re finalized, and a lot of
related software is open source. arXiv (<https://arxiv.org>) — pronounced
“archive” (the X stands for the Greek *chi*) — is an open access preprint server
for physics, mathematics, and computer science research papers. It has become
the de facto way to stay up to date on the cutting edge of machine learning
and deep learning. The large majority of deep learning researchers upload any
paper they write to arXiv shortly after completion. This allows them to plant
a flag and claim a specific finding without waiting for a conference
acceptance (which takes months), which is necessary given the fast pace of
research and the intense competition in the field. It also allows the field to
move extremely fast: all new findings are immediately available for all to see
and to build on.

An important downside is that the sheer quantity of new papers posted every day
on arXiv makes it impossible to even skim them all, and the fact that they
aren’t peer-reviewed makes it difficult to identify those that are both
important and high quality. It’s challenging, and becoming increasingly more so,
to find the signal in the noise. But some tools can help: in particular,
you can use Google Scholar (<https://scholar.google.com>) to keep track of
publications by your favorite authors.

### Explore the Keras ecosystem

With over 2.5 million users as of early 2025 and still growing,
Keras has a large ecosystem of tutorials, guides, and related open source projects:

* Your main reference for working with Keras is the online documentation
  at <https://keras.io>. In particular, you’ll find extensive
  developer guides at <https://keras.io/guides>, and you’ll find dozens
  of high-quality Keras code examples at <https://keras.io/examples>.
  Make sure to check them out!

* The Keras source code can be found at <https://github.com/keras-team/keras>,
  and Keras Hub can be found at <https://github.com/keras-team/keras-hub>.

* You can follow François (@fchollet) and Matt (@mattdangerw) on X (formerly Twitter).

## Final words

This is the end of *Deep Learning with Python*! I hope you’ve learned a thing
or two about machine learning, deep learning, Keras, and maybe even cognition
in general. Learning is a lifelong journey, especially in the field of AI,
where we have far more unknowns on our hands than certitudes. So please go on
learning, questioning, and researching. Never stop. Because even given the
progress made so far, most of the fundamental questions in AI remain
unanswered. Many haven’t even been properly asked yet.

#### **Tiếng Việt (Vietnamese)**

# Chương 20: Kết luận

Chương này bao gồm

* Những điều quan trọng rút ra từ cuốn sách này
* Tài nguyên để học thêm và áp dụng các kỹ năng của bạn vào thực tế

Chúng ta sẽ bắt đầu với cái nhìn toàn cảnh về những điều bạn nên rút ra từ cuốn sách này. Điều này sẽ làm mới trí nhớ của bạn về một số khái niệm bạn đã học. Sau đó, chúng tôi sẽ cung cấp cho bạn danh sách ngắn các tài nguyên và chiến lược để tìm hiểu thêm về học máy và cập nhật những tiến bộ mới.

Trở thành một nhà thực hành AI hiệu quả là một hành trình và việc hoàn thành cuốn sách này chỉ là bước đầu tiên của bạn trên đó. Tôi muốn đảm bảo rằng bạn nhận ra điều này và được trang bị phù hợp để tự mình thực hiện các bước tiếp theo của hành trình này.

## Các khái niệm chính đang được xem xét

Phần này tổng hợp ngắn gọn những điểm chính rút ra từ cuốn sách này. Nếu bạn cần ôn lại nhanh để giúp bạn nhớ lại những gì đã học, bạn có thể đọc vài trang này.

### Các cách tiếp cận khác nhau về trí tuệ nhân tạo

Đầu tiên, deep learning không đồng nghĩa với trí tuệ nhân tạo (AI) hay thậm chí với machine learning:

* *Trí tuệ nhân tạo* (AI) là một lĩnh vực rộng lớn, lâu đời có thể
thường được hiểu là “tất cả những nỗ lực nhằm tự động hóa quá trình nhận thức của con người”.
Điều này có thể bao gồm từ những điều rất cơ bản,
chẳng hạn như bảng tính Excel, cho đến những thứ rất tiên tiến, như robot hình người
có thể đi lại và nói chuyện.

* *Học máy* là một trường con cụ thể của AI nhằm mục đích tự động
phát triển chương trình (được gọi là *mô hình*) hoàn toàn từ việc tiếp xúc với dữ liệu đào tạo.
Quá trình biến dữ liệu thành chương trình này được gọi là *học*. Mặc dù
học máy đã có từ lâu, nó chỉ mới bắt đầu phát triển
vào những năm 1990, trước khi trở thành dạng AI thống trị vào những năm 2000.

* *Học sâu* là một trong nhiều nhánh của học máy, trong đó
mô hình là chuỗi dài các phép biến đổi hình học, được áp dụng lần lượt.
Các hoạt động này được cấu trúc thành các mô-đun gọi là *lớp*: học sâu
các mô hình thường là các chồng lớp - hay nói chung hơn là đồ thị của các lớp.
Các lớp này được tham số hóa bởi *weights*, là các tham số đã học
trong quá trình đào tạo. *Kiến thức* của một mô hình được lưu trữ dưới dạng trọng số của nó và
quá trình học tập bao gồm việc tìm kiếm “giá trị tốt” cho các trọng số này -
các giá trị giảm thiểu *hàm mất mát*. Bởi vì chuỗi biến đổi hình học
được coi là khả vi, cập nhật các trọng số để giảm thiểu hàm mất mát
được thực hiện một cách hiệu quả thông qua *giảm độ dốc*.

* *Generative AI* là một tập hợp con cụ thể của deep learning, trong đó các mô hình
có khả năng tạo ra văn bản, hình ảnh, video hoặc âm thanh. Những mô hình này có xu hướng
rất lớn - hàng tỷ tham số. Họ được đào tạo trong môi trường tự giám sát
thái độ; nghĩa là họ được đào tạo để tái tạo lại một cách nhân tạo
các phần bị thiếu hoặc bị hỏng của đầu vào -
ví dụ: khử nhiễu hình ảnh, dự đoán từ tiếp theo trong câu, v.v.
Quá trình học tập này cho phép các mô hình học các “bản đồ” phức tạp (nhúng đa tạp)
không gian đầu vào của chúng, có thể được sử dụng để lấy mẫu đầu vào mới. Các mẫu này đã ra mắt
AI bước vào kỷ nguyên “tiêu dùng” với sự trỗi dậy của các sản phẩm như ChatGPT hay Midjourney.

Mặc dù học sâu chỉ là một trong nhiều cách tiếp cận học máy, nhưng nó không ngang bằng với các phương pháp khác. Học sâu là một thành công đột phá. Đây là lý do tại sao.

### Điều gì khiến deep learning trở nên đặc biệt trong lĩnh vực machine learning

Chỉ trong vòng vài năm, deep learning đã đạt được những bước đột phá to lớn trên nhiều nhiệm vụ mà trước đây được coi là cực kỳ khó khăn đối với máy tính, đặc biệt là trong lĩnh vực nhận thức của máy: trích xuất thông tin hữu ích từ hình ảnh, video, âm thanh, v.v. Với đầy đủ dữ liệu đào tạo (đặc biệt là dữ liệu đào tạo được con người gắn nhãn phù hợp), học sâu giúp có thể trích xuất hầu hết mọi thứ mà con người có thể làm được từ dữ liệu nhận thức. Do đó, đôi khi người ta nói rằng deep learning đã “giải quyết được nhận thức” - mặc dù điều đó chỉ đúng với một định nghĩa khá hẹp về nhận thức.

Do những thành công chưa từng có về mặt kỹ thuật, học sâu đã tự mình mang lại *Mùa hè AI* thứ ba và lớn nhất cho đến nay: một thời kỳ thu hút sự quan tâm, đầu tư và cường điệu mạnh mẽ trong lĩnh vực AI. Khi cuốn sách này đang được viết, chúng ta đang ở giữa quá trình đó. Liệu giai đoạn này có kết thúc trong tương lai gần hay không và điều gì xảy ra sau khi nó kết thúc vẫn là chủ đề tranh luận. Có một điều chắc chắn: trái ngược hoàn toàn với các mùa hè AI trước đây, học sâu đã mang lại giá trị kinh doanh to lớn cho cả các công ty công nghệ lớn và nhỏ, đồng thời trở thành một thành công lớn của người tiêu dùng, cho phép nhận dạng giọng nói ở cấp độ con người, trợ lý chatbot, tạo hình ảnh chân thực, dịch máy ở cấp độ con người, v.v. Sự cường điệu có thể (và có thể sẽ) giảm đi, nhưng tác động kinh tế và công nghệ bền vững của học sâu sẽ vẫn còn. Theo nghĩa đó, deep learning có thể tương tự như internet: nó có thể được quảng bá quá mức trong một vài năm, nhưng về lâu dài, nó vẫn sẽ là một cuộc cách mạng lớn giúp thay đổi nền kinh tế và cuộc sống của chúng ta.

Một lý do khiến tôi đặc biệt lạc quan về học sâu là ngay cả khi chúng ta không đạt được tiến bộ công nghệ nào nữa trong thập kỷ tới, thì việc triển khai các thuật toán hiện có cho mọi vấn đề có thể áp dụng sẽ là yếu tố thay đổi cuộc chơi cho hầu hết các ngành. Học sâu không khác gì một cuộc cách mạng và sự tiến bộ hiện đang diễn ra với tốc độ cực kỳ nhanh do sự đầu tư theo cấp số nhân vào nguồn lực và số lượng nhân viên. From where we stand, the future looks bright, although short-term expectations are somewhat overoptimistic; deploying deep learning to the full extent of its potential will likely take multiple decades.

### Cách nghĩ về học sâu

Điều đáng ngạc nhiên nhất về deep learning là nó đơn giản đến mức nào. Mười lăm năm trước, không ai ngờ rằng chúng ta sẽ đạt được kết quả đáng kinh ngạc như vậy đối với các vấn đề về nhận thức máy và xử lý ngôn ngữ tự nhiên bằng cách sử dụng các mô hình tham số đơn giản được đào tạo với độ dốc giảm dần. Bây giờ, hóa ra tất cả những gì bạn cần là các mô hình tham số đủ lớn được đào tạo với độ dốc giảm dần trên đủ nhiều ví dụ. Như Feynman đã từng nói về vũ trụ, “Nó không phức tạp, chỉ là rất nhiều thôi.”[[1]](#footnote-1)

Trong học sâu, mọi thứ đều là vectơ; nghĩa là mọi thứ đều là một *điểm* trong một *không gian hình học*. Đầu vào của mô hình (văn bản, hình ảnh, v.v.) và mục tiêu trước tiên được *vector hóa* — được chuyển thành không gian vectơ đầu vào ban đầu và không gian vectơ mục tiêu. Mỗi lớp trong mô hình học sâu thực hiện một phép biến đổi hình học đơn giản trên dữ liệu đi qua nó. Cùng với nhau, chuỗi các lớp trong mô hình tạo thành một phép biến đổi hình học phức tạp, được chia thành một loạt các lớp đơn giản. Phép biến đổi phức tạp này cố gắng ánh xạ không gian đầu vào vào không gian đích, mỗi lần một điểm. Phép biến đổi này được tham số hóa bằng trọng số của các lớp, được cập nhật lặp đi lặp lại dựa trên mức độ hoạt động của mô hình hiện tại. Đặc điểm chính của phép biến đổi hình học này là nó phải *có khả vi*, điều này bắt buộc chúng ta có thể tìm hiểu các tham số của nó thông qua việc giảm độ dốc. Theo trực giác, điều này có nghĩa là sự biến đổi hình học từ đầu vào đến đầu ra phải trơn tru và liên tục - một hạn chế đáng kể.

Toàn bộ quá trình áp dụng phép biến đổi hình học phức tạp này cho dữ liệu đầu vào có thể được hiển thị dưới dạng 3D bằng cách tưởng tượng một người đang cố gắng gỡ một quả bóng giấy: quả bóng giấy bị nhàu nát là tập hợp dữ liệu đầu vào mà mô hình bắt đầu. Mỗi chuyển động do người thực hiện trên quả bóng giấy tương tự như một phép biến đổi hình học đơn giản được thực hiện bởi một lớp. Chuỗi cử chỉ hoàn chỉnh không bị nhàu nát là sự biến đổi phức tạp của toàn bộ mô hình. Các mô hình học sâu là những cỗ máy toán học để giải mã các tập hợp dữ liệu nhiều chiều phức tạp.

Đó là điều kỳ diệu của học sâu - biến ý nghĩa thành vectơ, thành không gian hình học, sau đó học dần dần các phép biến đổi hình học phức tạp để ánh xạ không gian này sang không gian khác. Tất cả những gì bạn cần là không gian có chiều đủ cao để nắm bắt được phạm vi đầy đủ của các mối quan hệ được tìm thấy trong dữ liệu gốc.

Toàn bộ vấn đề xoay quanh hai ý tưởng cốt lõi: *ý nghĩa bắt nguồn từ mối quan hệ cặp đôi giữa các sự vật* (giữa các từ trong ngôn ngữ, giữa các pixel trong một hình ảnh, v.v.) và *những mối quan hệ này có thể được nắm bắt bằng hàm khoảng cách*. Nhưng lưu ý rằng liệu bộ não có thực hiện được ý nghĩa thông qua các không gian hình học hay không là một câu hỏi hoàn toàn riêng biệt. Không gian vectơ hoạt động hiệu quả từ quan điểm tính toán, nhưng có thể dễ dàng hình dung các cấu trúc dữ liệu khác nhau dành cho trí thông minh - đặc biệt là đồ thị. Mạng lưới thần kinh ban đầu xuất hiện từ ý tưởng sử dụng đồ thị như một cách để mã hóa ý nghĩa, đó là lý do tại sao chúng được đặt tên là *mạng lưới thần kinh*; lĩnh vực nghiên cứu xung quanh từng được gọi là *chủ nghĩa kết nối*. Ngày nay, cái tên *mạng lưới thần kinh* tồn tại hoàn toàn vì lý do lịch sử - đó là một cái tên cực kỳ dễ gây hiểu lầm vì chúng không phải là mạng lưới thần kinh hay mạng lưới. Đặc biệt, mạng lưới thần kinh hầu như không liên quan gì đến bộ não. Một cái tên thích hợp hơn sẽ là *học biểu diễn theo lớp* hoặc *học biểu diễn theo cấp bậc* hoặc thậm chí có thể sâu *các mô hình vi phân* hoặc *các phép biến đổi hình học theo chuỗi*, để nhấn mạnh thực tế rằng cốt lõi của chúng là thao tác không gian hình học liên tục.

### Các công nghệ hỗ trợ chính

Cuộc cách mạng công nghệ hiện đang diễn ra không bắt đầu từ bất kỳ phát minh mang tính đột phá nào. Đúng hơn, giống như bất kỳ cuộc cách mạng nào khác, nó là sản phẩm của sự tích lũy rất lớn các yếu tố tạo điều kiện - lúc đầu chậm rãi, sau đó đột ngột. Trong trường hợp học sâu, chúng ta có thể chỉ ra các yếu tố chính sau:

* Những đổi mới về thuật toán tăng dần, lần đầu tiên lan rộng trong hai thập kỷ (bắt đầu
với lan truyền ngược) và sau đó diễn ra ngày càng nhanh hơn khi có nhiều nghiên cứu hơn
nỗ lực đã được đổ vào deep learning sau năm 2012. Một trong số đó là
đột phá là kiến ​​trúc Transformer năm 2017.

* Sự sẵn có của một lượng lớn dữ liệu hình ảnh, video và văn bản là một yêu cầu
để nhận ra rằng các mô hình đủ lớn được đào tạo trên
dữ liệu là tất cả những gì chúng ta cần. Đây lại là sản phẩm phụ của sự gia tăng của người tiêu dùng
internet và định luật Moore áp dụng cho phương tiện lưu trữ. Ngày nay, các mô hình ngôn ngữ tiên tiến
được đào tạo trên một phần lớn của toàn bộ internet.

* Sự sẵn có của phần cứng tính toán song song nhanh, cao ở mức giá thấp,
đặc biệt là GPU do NVIDIA sản xuất - GPU chơi game đầu tiên và sau đó là chip
được thiết kế từ đầu cho việc học sâu. Ngay từ đầu, Giám đốc điều hành NVIDIA Jensen
Huang nhận thấy sự bùng nổ của deep learning và quyết định đặt cược vào lợi thế của công ty.
tương lai của nó, điều đó đã được đền đáp xứng đáng.

* Một tập hợp các lớp phần mềm phức tạp tạo nên sức mạnh tính toán này
có sẵn cho con người: ngôn ngữ CUDA, các framework như TensorFlow, JAX và PyTorch.
phân biệt tự động và Keras, giúp cho việc học sâu có thể tiếp cận được
hầu hết mọi người.

Trong tương lai, deep learning sẽ không chỉ được sử dụng bởi các chuyên gia như nhà nghiên cứu, nghiên cứu sinh và kỹ sư có hồ sơ học thuật; nó sẽ là một công cụ trong hộp công cụ của mọi nhà phát triển, giống như công nghệ web ngày nay. Mọi người đều cần xây dựng các ứng dụng thông minh: giống như mọi doanh nghiệp ngày nay đều cần một trang web, mọi sản phẩm sẽ cần phải hiểu được dữ liệu do người dùng tạo một cách thông minh. Để đạt được tương lai này, chúng ta phải xây dựng các công cụ giúp việc học sâu hoàn toàn dễ sử dụng và có thể truy cập được đối với bất kỳ ai có khả năng viết mã cơ bản. Keras là bước quan trọng đầu tiên theo hướng đó.

### Quy trình làm việc của máy học phổ quát

Có quyền truy cập vào một công cụ cực kỳ mạnh mẽ để tạo các mô hình ánh xạ bất kỳ không gian đầu vào nào tới bất kỳ không gian mục tiêu nào là điều tuyệt vời, nhưng phần khó khăn của quy trình học máy thường là mọi thứ xảy ra trước khi thiết kế và đào tạo các mô hình đó (và đối với các mô hình sản xuất, cả những gì diễn ra sau đó). Hiểu miền vấn đề để có thể xác định nội dung cần dự đoán, dữ liệu được cung cấp và cách đo lường thành công là điều kiện tiên quyết cho bất kỳ ứng dụng học máy thành công nào và đó không phải là điều mà các công cụ nâng cao như Keras và TensorFlow có thể giúp bạn. Xin nhắc lại, đây là bản tóm tắt nhanh về quy trình học máy điển hình như được mô tả trong chương 6:

* *Xác định vấn đề.* Dữ liệu nào có sẵn và bạn đang cố gắng làm gì
dự đoán? Bạn sẽ cần thu thập thêm dữ liệu hay thuê người gắn nhãn thủ công cho sản phẩm?
tập dữ liệu?
* *Xác định cách đo lường mức độ thành công theo mục tiêu của bạn một cách đáng tin cậy.* Đối với các nhiệm vụ đơn giản,
đây có thể là độ chính xác của dự đoán, nhưng trong nhiều trường hợp, nó sẽ yêu cầu
các số liệu phức tạp, theo miền cụ thể.
* *Chuẩn bị quy trình xác thực mà bạn sẽ sử dụng để đánh giá mô hình của mình.* Trong
cụ thể, bạn nên xác định tập huấn luyện, tập xác thực và tập kiểm tra
bộ. Nhãn tập xác thực và tập kiểm tra không được rò rỉ vào quá trình đào tạo
dữ liệu: ví dụ: với dự đoán theo thời gian, dữ liệu xác thực và kiểm tra
nên ở phía sau dữ liệu huấn luyện.
* *Vector hóa dữ liệu bằng cách biến nó thành vectơ và xử lý trước theo cách giúp mạng thần kinh dễ dàng tiếp cận dữ liệu hơn (chuẩn hóa, v.v.).*
* *Phát triển mô hình đầu tiên vượt qua đường cơ sở thông thường tầm thường, từ đó chứng minh rằng học máy có thể giải quyết được vấn đề của bạn.* Điều này có thể không
luôn luôn là như vậy!
* *Dần dần tinh chỉnh kiến ​​trúc mô hình của bạn bằng cách điều chỉnh siêu tham số và thêm chính quy.* Thực hiện các thay đổi chỉ dựa trên hiệu suất trên dữ liệu xác thực,
không phải dữ liệu thử nghiệm hoặc dữ liệu huấn luyện. Hãy nhớ rằng bạn sẽ nhận được
mô hình quá phù hợp (do đó xác định mức công suất của mô hình lớn hơn
bạn cần) và chỉ sau đó mới bắt đầu thêm chính quy hóa hoặc thu nhỏ mô hình của bạn.
Hãy cẩn thận với việc thiết lập quá mức xác thực khi điều chỉnh siêu tham số - thực tế
rằng siêu tham số của bạn có thể trở nên quá chuyên biệt để xác thực
bộ. Tránh điều này là mục đích của việc có một bộ thử nghiệm riêng biệt!
* *Triển khai mô hình cuối cùng của bạn trong sản xuất — dưới dạng API web, như một phần của ứng dụng JavaScript hoặc C++, trên thiết bị nhúng, v.v.* Tiếp tục theo dõi hiệu suất của mô hình đó
trên dữ liệu trong thế giới thực và sử dụng những phát hiện của bạn để tinh chỉnh lần lặp tiếp theo của mô hình!

### Kiến trúc mạng chính

Các họ kiến ​​trúc mạng mà bạn nên làm quen sau khi đọc cuốn sách này là *mạng kết nối dày đặc*, *mạng tích chập*, *mạng tái phát*, *Mô hình khuếch tán* và *Máy biến áp*. Mỗi loại mô hình dành cho các phương thức dữ liệu cụ thể: kiến ​​trúc mạng mã hóa *giả định* về cấu trúc của dữ liệu — một *không gian giả thuyết* trong đó việc tìm kiếm một mô hình tốt sẽ tiến hành. Liệu một kiến ​​trúc nhất định có hoạt động được trên một vấn đề nhất định hay không phụ thuộc hoàn toàn vào sự phù hợp giữa cấu trúc dữ liệu và các giả định của kiến ​​trúc mạng.

Các loại mạng khác nhau này có thể dễ dàng được kết hợp để đạt được các mô hình đa phương thức lớn hơn, giống như bạn kết hợp các viên gạch LEGO. Theo một cách nào đó, các lớp học sâu là những viên gạch LEGO để xử lý thông tin. Bảng 20.1 trình bày tổng quan nhanh về ánh xạ giữa các phương thức đầu vào và đầu ra cũng như các kiến ​​trúc mạng thích hợp.

| Đầu vào | Đầu ra | Người mẫu | | --- | --- | --- | | Dữ liệu vectơ | Xác suất lớp, Giá trị hồi quy | Mạng kết nối dày đặc | | Dữ liệu chuỗi thời gian | Xác suất lớp, Giá trị hồi quy | RNN, Máy biến áp | | Hình ảnh | Xác suất lớp, Giá trị hồi quy | ConvNet | | văn bản | Xác suất lớp, Giá trị hồi quy | Máy biến áp | | Văn bản, Hình ảnh | văn bản | Máy biến áp | | Văn bản, Hình ảnh | Hình ảnh | VAE, Mô hình khuếch tán |

[Bảng 20.1](#table-20-1): Kiến trúc mô hình cho các loại dữ liệu khác nhau

Bây giờ hãy nhanh chóng xem xét các đặc điểm cụ thể của từng kiến ​​trúc mạng.

#### Mạng kết nối dày đặc

Mạng được kết nối dày đặc là một tập hợp các lớp `Dense`, dùng để xử lý dữ liệu vectơ (trong đó mỗi mẫu là một vectơ thuộc tính số hoặc phân loại). Các mạng như vậy giả định không có cấu trúc cụ thể trong các tính năng đầu vào: chúng được gọi là *kết nối dày đặc* vì các đơn vị của lớp `Dense` được kết nối với mọi đơn vị khác. Lớp này cố gắng ánh xạ các mối quan hệ giữa hai tính năng đầu vào bất kỳ; chẳng hạn, điều này không giống như lớp tích chập 2D, chỉ xem xét các mối quan hệ *cục bộ*.

Các mạng được kết nối dày đặc được sử dụng phổ biến nhất cho dữ liệu phân loại (ví dụ: trong đó các tính năng đầu vào là danh sách các thuộc tính), chẳng hạn như tập dữ liệu Giá Nhà ở Boston được sử dụng trong chương 4. Chúng cũng được sử dụng làm giai đoạn phân loại hoặc hồi quy cuối cùng của hầu hết các mạng. Ví dụ: các ConvNet được đề cập trong chương 8 thường kết thúc bằng một hoặc hai lớp `Dense` và các mạng lặp lại trong chương 13 cũng vậy.

Hãy nhớ rằng, để thực hiện *phân loại nhị phân*, hãy kết thúc chồng lớp của bạn bằng lớp `Dense` với một đơn vị duy nhất và kích hoạt `sigmoid`, đồng thời sử dụng `binary_crossentropy` làm phần mất. Mục tiêu của bạn phải là 0 hoặc 1:

```python
import keras
from keras import layers

inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
```

Để thực hiện *phân loại phân loại nhãn đơn* (trong đó mỗi mẫu có chính xác một lớp, không nhiều hơn), hãy kết thúc chồng lớp của bạn bằng lớp `Dense` với số đơn vị bằng với số lượng lớp và kích hoạt `softmax`. Nếu mục tiêu của bạn được mã hóa một lần, hãy sử dụng `categorical_crossentropy` làm phần mất; nếu chúng là số nguyên, hãy sử dụng `sparse_categorical_crossentropy`:

```python
inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
```

Để thực hiện *phân loại phân loại nhiều nhãn* (trong đó mỗi mẫu có thể có một số lớp), hãy kết thúc chồng lớp của bạn bằng lớp `Dense` với số đơn vị bằng với số lượng lớp và kích hoạt `sigmoid`, đồng thời sử dụng `binary_crossentropy` làm phần mất. Mục tiêu của bạn phải được mã hóa k-hot:

```python
inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
```

Để thực hiện *hồi quy* đối với vectơ có các giá trị liên tục, hãy kết thúc chồng lớp của bạn bằng lớp `Dense` với số đơn vị bằng với số lượng giá trị bạn đang cố gắng dự đoán (thường là một giá trị duy nhất, chẳng hạn như giá một ngôi nhà) và không kích hoạt. Có thể sử dụng nhiều tổn thất khác nhau để hồi quy - phổ biến nhất là `mean_squared_error` (MSE):

```python
inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_values)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="mse")
```

#### ConvNet

Các lớp tích chập xem xét các mẫu không gian cục bộ bằng cách áp dụng cùng một phép biến đổi hình học cho các vị trí không gian khác nhau (*bản vá*) trong một tenxơ đầu vào. Điều này dẫn đến các biểu diễn *bất biến dịch*, làm cho các lớp tích chập có hiệu quả dữ liệu cao và mang tính mô-đun. Ý tưởng này có thể áp dụng cho các không gian có chiều bất kỳ: 1D (các chuỗi liên tục), 2D (hình ảnh), 3D (khối), v.v. Bạn có thể sử dụng lớp `Conv1D` để xử lý chuỗi, lớp `Conv2D` để xử lý hình ảnh và lớp `Conv3D` để xử lý khối. Là một giải pháp thay thế gọn gàng hơn, hiệu quả hơn cho các lớp tích chập, bạn cũng có thể sử dụng các lớp *tích chập có thể phân tách theo chiều sâu*, chẳng hạn như `SeparableConv2D`.

*ConvNets* hoặc *mạng tích chập*, bao gồm các lớp tích chập và tổng hợp tối đa. Các lớp tổng hợp cho phép bạn lấy mẫu dữ liệu theo không gian, cần thiết để giữ bản đồ đối tượng ở kích thước hợp lý khi số lượng đối tượng tăng lên và cho phép các lớp tích chập tiếp theo “nhìn thấy” phạm vi không gian lớn hơn của đầu vào. ConvNet thường được kết thúc bằng thao tác `Làm phẳng` hoặc lớp tổng hợp toàn cầu, biến bản đồ đặc điểm không gian thành vectơ, theo sau là các lớp `Dense` để đạt được phân loại hoặc hồi quy.

Đây là mạng phân loại hình ảnh điển hình (trong trường hợp này là phân loại theo danh mục) sử dụng các lớp `SeparableConv2D`:

```python
inputs = keras.Input(shape=(height, width, channels))
x = layers.SeparableConv2D(32, 3, activation="relu")(inputs)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.SeparableConv2D(128, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.SeparableConv2D(128, 3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
```

Khi xây dựng một ConvNet rất chuyên sâu, người ta thường thêm các lớp *chuẩn hóa hàng loạt* cũng như *các kết nối dư* — hai mẫu kiến ​​trúc giúp thông tin chuyển màu lưu chuyển suôn sẻ qua mạng.

#### Máy biến áp

Máy biến áp xem xét một tập hợp các vectơ (chẳng hạn như vectơ từ) và sử dụng *sự chú ý thần kinh* để biến đổi mỗi vectơ thành một biểu diễn có thể nhận biết được *ngữ cảnh* do các vectơ khác trong tập hợp đó cung cấp. Khi tập hợp được đề cập là một chuỗi có thứ tự, bạn cũng có thể sử dụng *mã hóa vị trí* để tạo Máy biến áp có thể tính đến cả ngữ cảnh chung và thứ tự từ, có khả năng xử lý các đoạn văn bản dài hiệu quả hơn nhiều so với RNN hoặc ConvNet 1D.

Máy biến áp có thể được sử dụng cho bất kỳ tác vụ xử lý tập hợp hoặc xử lý trình tự nào, bao gồm cả phân loại văn bản, nhưng chúng đặc biệt vượt trội ở *học theo trình tự*, chẳng hạn như dịch các đoạn văn trong ngôn ngữ nguồn sang ngôn ngữ đích.

Máy biến áp tuần tự được tạo thành từ hai phần:

* Một `TransformerEncode` quay
một chuỗi vectơ đầu vào thành một chuỗi vectơ đầu ra nhận biết theo ngữ cảnh, nhận biết thứ tự
* Một `TransformerDecode` lấy đầu ra của `TransformerEncode`, dưới dạng
cũng như trình tự mục tiêu và dự đoán điều gì sẽ xảy ra tiếp theo trong trình tự mục tiêu

Nếu bạn chỉ xử lý một chuỗi (hoặc tập hợp) vectơ duy nhất, bạn sẽ chỉ sử dụng `TransformerEncode`.

Sau đây là Trình biến đổi trình tự thành trình tự để ánh xạ trình tự nguồn sang trình tự đích (ví dụ: thiết lập này có thể được sử dụng để dịch máy hoặc trả lời câu hỏi):

```python
from keras_hub.layers import TokenAndPositionEmbedding
from keras_hub.layers import TransformerDecoder, TransformerEncoder

# Source sequence
encoder_inputs = keras.Input(shape=(src_seq_length,), dtype="int64")
x = TokenAndPositionEmbedding(vocab_size, src_seq_length, embed_dim)(
    encoder_inputs
)
encoder_outputs = TransformerEncoder(intermediate_dim=256, num_heads=8)(x)
# Target sequence so far
decoder_inputs = keras.Input(shape=(dst_seq_length,), dtype="int64")
x = TokenAndPositionEmbedding(vocab_size, dst_seq_length, embed_dim)(
    decoder_inputs
)
x = TransformerDecoder(intermediate_dim=256, num_heads=8)(x, encoder_outputs)
# Predictions for target sequence one step in the future
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
transformer.compile(optimizer="adamw", loss="categorical_crossentropy")
```

Và đây là một `TransformerEncode` duy nhất để phân loại nhị phân các chuỗi số nguyên:

```python
inputs = keras.Input(shape=(seq_length,), dtype="int64")
x = TokenAndPositionEmbedding(vocab_size, seq_length, embed_dim)(inputs)
x = TransformerEncoder(intermediate_dim=256, num_heads=8)(x)
x = layers.GlobalMaxPooling1D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="adamw", loss="binary_crossentropy")
```

#### Mạng lưới thần kinh tái phát

*Mạng thần kinh tái phát* (RNN) hoạt động bằng cách xử lý chuỗi đầu vào theo từng dấu thời gian và duy trì trạng thái xuyên suốt (trạng thái thường là một vectơ hoặc tập hợp vectơ). Chúng nên được sử dụng tốt hơn so với ConvNet 1D trong trường hợp các chuỗi trong đó các mẫu quan tâm không bất biến bởi sự dịch chuyển theo thời gian (ví dụ: dữ liệu chuỗi thời gian trong đó quá khứ gần đây quan trọng hơn quá khứ xa xôi).

Ba lớp RNN có sẵn trong Keras: `SimpleRNN`, `GRU` và `LSTM`. Đối với hầu hết các mục đích thực tế, bạn nên sử dụng `GRU` hoặc `LSTM`. `LSTM` mạnh hơn trong hai loại nhưng cũng đắt hơn; bạn có thể nghĩ về `GRU` như một giải pháp thay thế đơn giản hơn, rẻ hơn cho nó.

Để xếp chồng nhiều lớp RNN lên nhau, mỗi lớp trước lớp cuối cùng trong ngăn xếp phải trả về chuỗi đầy đủ các đầu ra của nó (mỗi dấu thời gian đầu vào sẽ tương ứng với một dấu thời gian đầu ra); nếu bạn không xếp chồng thêm bất kỳ lớp RNN nào nữa thì thông thường chỉ trả về kết quả đầu ra cuối cùng chứa thông tin về toàn bộ chuỗi.

Sau đây là một lớp RNN duy nhất để phân loại nhị phân các chuỗi vectơ:

```python
inputs = keras.Input(shape=(num_timesteps, num_features))
x = layers.LSTM(32)(inputs)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
```

Và đây là lớp RNN xếp chồng lên nhau để phân loại nhị phân các chuỗi vectơ:

```python
inputs = keras.Input(shape=(num_timesteps, num_features))
x = layers.LSTM(32, return_sequences=True)(inputs)
x = layers.LSTM(32, return_sequences=True)(x)
x = layers.LSTM(32)(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
```

## Hạn chế của học sâu

Xây dựng các mô hình học sâu cũng giống như chơi với những viên gạch LEGO: các lớp có thể được kết nối với nhau để ánh xạ về cơ bản mọi thứ với mọi thứ, miễn là bạn có sẵn dữ liệu đào tạo phù hợp và có thể đạt được ánh xạ thông qua chuyển đổi hình học liên tục với độ phức tạp hợp lý.

Tuy nhiên, đây là điểm đáng chú ý - việc lập bản đồ này thường không thể học được theo cách khái quát hóa. Các mô hình học sâu hoạt động giống như cơ sở dữ liệu nội suy rộng lớn về các mẫu. Điểm mạnh phù hợp với khuôn mẫu cũng là điểm yếu cốt lõi của họ:

* *Về cơ bản, họ gặp khó khăn trong việc thích nghi với sự mới lạ.*
Vì các tham số của chúng được cố định sau khi huấn luyện nên chúng chỉ có thể truy xuất hoặc sao chép các mẫu tương tự với dữ liệu huấn luyện của chúng.
Đối mặt với những đầu vào nằm ngoài phạm vi phân phối quen thuộc này — cho dù nhiệm vụ cơ bản có đơn giản đến đâu —
hiệu suất của họ giảm sút nghiêm trọng vì họ thiếu cơ chế khái quát hóa linh hoạt ngoài kinh nghiệm đã ghi nhớ của họ.
Điều này giải thích tại sao ngay cả những mô hình lớn cũng thất bại trong các nhiệm vụ mới hoặc các biến thể đơn giản của các vấn đề quen thuộc — như nhiệm vụ ARC-AGI.

* *Họ rất nhạy cảm với cách diễn đạt và những yếu tố gây xao nhãng khác.* Các mô hình học sâu thể hiện độ nhạy cao với những điều hời hợt
các biến thể trong cách trình bày đầu vào, chẳng hạn như các thay đổi nhỏ về cụm từ (xem xét độ nhạy của lời nhắc trong LLM)
hoặc những xáo trộn không thể nhận thấy (xem xét các ví dụ đối lập trong tầm nhìn), cho thấy sự thiếu hiểu biết mạnh mẽ, giống con người.

* *Họ thường không thể học các thuật toán có thể khái quát hóa.* Bản chất hình học, liên tục của các mô hình học sâu khiến chúng trở nên cơ bản
không phù hợp để học các thuật toán chính xác, rời rạc, từng bước, chẳng hạn như những thuật toán cốt lõi của khoa học máy tính cổ điển.
Các mô hình gần đúng các quy trình như vậy thông qua phép nội suy thay vì thực hiện các quy trình mạnh mẽ, có tính khái quát hóa.

Bạn phải luôn chống lại sự cám dỗ của việc nhân cách hóa các mô hình học sâu. Hiệu suất của chúng được xây dựng dựa trên các mẫu thống kê theo từng điểm chứ không phải dựa trên nền tảng kinh nghiệm giống con người, khiến nó trở nên dễ vỡ khi gặp phải những sai lệch so với dữ liệu huấn luyện.

Câu chuyện cho rằng chỉ cần tăng quy mô mô hình và dữ liệu huấn luyện sẽ dẫn đến trí thông minh chung đã được chứng minh là không đủ. Mặc dù việc mở rộng quy mô giúp nâng cao hiệu suất trên các điểm chuẩn tương đương với các bài kiểm tra khả năng ghi nhớ, nhưng nó không giải quyết được các hạn chế cơ bản của học sâu, xuất phát từ mô hình cốt lõi là khớp các đường cong nội suy tĩnh với dữ liệu. Năm năm mở rộng quy mô theo cấp số nhân của LLM cơ sở vẫn chưa khắc phục được những hạn chế này vì cách tiếp cận cơ bản vẫn không thay đổi.

Đến năm 2024, nhận thức này đã thúc đẩy quá trình chuyển đổi sang thích ứng với thời gian thử nghiệm (TTA), trong đó các mô hình thực hiện tìm kiếm hoặc tinh chỉnh trong giai đoạn suy luận để thích ứng với các vấn đề mới. Mặc dù các phương pháp TTA đã mang lại những bước đột phá lớn, chẳng hạn như O3 của OpenAI vượt qua mức cơ bản của con người trên ARC-AGI-1 vào cuối năm 2024, nhưng hiệu suất này lại phải trả một chi phí tính toán cực kỳ cao. Sự thích ứng hiệu quả, giống con người vẫn là một vấn đề hoàn toàn mở và tiêu chuẩn ARC-AGI-2 khó hơn một chút vẫn hoàn toàn chưa được giải quyết cho đến ngày nay. Chúng tôi vẫn cần những tiến bộ về mặt khái niệm hơn nữa ngoài việc mở rộng quy mô hoặc tìm kiếm vũ phu.

## Điều gì có thể ở phía trước

Việc giải quyết trí thông minh chất lỏng giống con người (và ARC-AGI-2) đòi hỏi phải vượt qua những giới hạn cố hữu trong các phương pháp tiếp cận hiện tại. Mặc dù học sâu vượt trội ở *sự trừu tượng lấy giá trị làm trung tâm*, cho phép nhận dạng mẫu và trực giác, nhưng về cơ bản, nó thiếu các khả năng cho *sự trừu tượng lấy chương trình làm trung tâm*, làm nền tảng cho lý luận, lập kế hoạch và hiểu biết nhân quả rời rạc. Trí thông minh của con người tích hợp liền mạch cả hai - AI trong tương lai cũng phải làm được điều tương tự.

Những phát triển quan trọng trong tương lai có thể bao gồm

* *Mô hình kết hợp* — Các mô hình trong tương lai có thể sẽ tích hợp các mô-đun thuật toán đã học (cung cấp khả năng suy luận và thao tác ký hiệu)
với các mô-đun học sâu (cung cấp nhận thức và trực giác).
Các hệ thống này có thể học cách sử dụng các nguyên tắc lập trình như luồng điều khiển, biến, đệ quy và cấu trúc dữ liệu phức tạp một cách linh hoạt.

* *Tìm kiếm chương trình có hướng dẫn học sâu* — Tổng hợp chương trình — tự động phát hiện mã thực thi đáp ứng thông số kỹ thuật —
cung cấp một lộ trình đến sự trừu tượng hóa lấy chương trình làm trung tâm. Tuy nhiên, sự phụ thuộc vào tìm kiếm rời rạc không hiệu quả là một trở ngại lớn.
Một tiến bộ quan trọng sẽ là sử dụng học sâu để hướng dẫn tìm kiếm này, sử dụng trực giác đã học được về cấu trúc chương trình để điều hướng các không gian tổ hợp rộng lớn của chương trình.
hiệu quả, giống như việc các nhà phát triển con người sử dụng kinh nghiệm và trực giác để thu hẹp các lựa chọn của họ.

* *Tái kết hợp mô-đun và học tập suốt đời* — Chúng tôi sẽ loại bỏ các mô hình nguyên khối, toàn diện được đào tạo từ đầu.
Thay vào đó, các hệ thống AI trong tương lai sẽ sử dụng các thư viện khổng lồ gồm các thành phần mô-đun, có thể tái sử dụng, có thể được sử dụng lại để giải quyết nhiều vấn đề, thu được từ kinh nghiệm.
Các thư viện này sẽ có cả mô-đun “hình học” (dựa trên học tập sâu) và “thuật toán”.
Khi gặp một vấn đề mới, các hệ thống AI như vậy sẽ tìm nạp các mô-đun có liên quan và tự động kết hợp lại chúng thành một mô hình mới được điều chỉnh phù hợp.
đến tình huống hiện tại. Bất cứ khi nào hệ thống kết thúc việc phát triển một thành phần có thể tái sử dụng như một sản phẩm phụ của quá trình giải quyết vấn đề này
loop, thành phần mới sẽ được thêm vào thư viện, sẵn sàng cho mọi tác vụ trong tương lai mà hệ thống có thể gặp phải.

Cuối cùng, việc phát triển AI phản ánh trí thông minh chất lỏng giống con người sẽ yêu cầu kết hợp nhận dạng mẫu liên tục cùng với các chương trình mang tính biểu tượng, rời rạc và hoàn toàn chấp nhận mô hình thích ứng nhanh chóng.

## Luôn cập nhật trong một lĩnh vực chuyển động nhanh

Như lời chia tay cuối cùng, tôi muốn đưa ra cho bạn một số gợi ý về cách tiếp tục học tập và cập nhật kiến ​​thức cũng như kỹ năng của bạn sau khi bạn lật trang cuối cùng của cuốn sách này. Lĩnh vực học sâu hiện đại, như chúng ta biết ngày nay, chỉ mới hình thành được vài năm, mặc dù có lịch sử lâu dài và chậm chạp kéo dài hàng thập kỷ. Với sự gia tăng theo cấp số nhân về nguồn tài chính và số lượng nhân viên nghiên cứu kể từ năm 2013, toàn bộ lĩnh vực này hiện đang phát triển với tốc độ chóng mặt. Những gì bạn học được trong cuốn sách này sẽ không còn phù hợp mãi mãi và đó không phải là tất cả những gì bạn cần cho phần còn lại của sự nghiệp.

May mắn thay, có rất nhiều tài nguyên trực tuyến miễn phí mà bạn có thể sử dụng để cập nhật và mở rộng tầm nhìn của mình. Dưới đây là một vài.

### Thực hành giải quyết các vấn đề thực tế bằng Kaggle

Một cách hiệu quả để có được trải nghiệm thực tế là thử sức mình trong các cuộc thi học máy trên Kaggle (<https://kaggle.com>). Cách thực sự duy nhất để học là thông qua thực hành và viết mã thực tế - đó là triết lý của cuốn sách này và các cuộc thi Kaggle là sự tiếp nối tự nhiên của điều này. Trên Kaggle, bạn sẽ tìm thấy một loạt các cuộc thi khoa học dữ liệu được đổi mới liên tục, nhiều cuộc thi trong số đó liên quan đến học sâu, do các công ty quan tâm đến việc đạt được các giải pháp mới cho một số vấn đề học máy thách thức nhất của họ chuẩn bị. Giải thưởng tiền tệ khá lớn được trao cho những người tham gia hàng đầu.

Bằng cách tham gia một vài cuộc thi, có thể với tư cách là thành viên của một nhóm, bạn sẽ trở nên quen thuộc hơn với khía cạnh thực tế của một số phương pháp thực hành nâng cao tốt nhất được mô tả trong cuốn sách này, đặc biệt là điều chỉnh siêu tham số, tránh việc trang bị quá mức bộ xác thực và tập hợp mô hình.

### Đọc về những phát triển mới nhất trên arXiv

Nghiên cứu deep learning, trái ngược với một số lĩnh vực khoa học khác, diễn ra hoàn toàn mở. Các bài viết được công khai và có thể truy cập miễn phí ngay sau khi chúng được hoàn thiện và rất nhiều phần mềm liên quan là nguồn mở. arXiv (<https://arxiv.org>) — phát âm là “kho lưu trữ” (chữ X là viết tắt của *chi* trong tiếng Hy Lạp) — là một máy chủ in sẵn truy cập mở dành cho các tài liệu nghiên cứu vật lý, toán học và khoa học máy tính. Trên thực tế, nó đã trở thành cách để cập nhật những thông tin tiên tiến nhất về học máy và học sâu. Phần lớn các nhà nghiên cứu deep learning tải bất kỳ bài viết nào họ viết lên arXiv ngay sau khi hoàn thành. Điều này cho phép họ cắm cờ và tuyên bố một phát hiện cụ thể mà không cần chờ hội nghị chấp nhận (mất nhiều tháng), điều này cần thiết do tốc độ nghiên cứu nhanh chóng và sự cạnh tranh khốc liệt trong lĩnh vực này. Nó cũng cho phép lĩnh vực này phát triển cực kỳ nhanh chóng: tất cả những phát hiện mới đều có sẵn ngay lập tức cho tất cả mọi người xem và tiếp tục phát triển.

Một nhược điểm quan trọng là số lượng lớn các bài báo mới được đăng mỗi ngày trên arXiv khiến cho việc đọc lướt qua tất cả chúng là không thể và thực tế là chúng không được bình duyệt khiến việc xác định những bài báo vừa quan trọng vừa có chất lượng cao trở nên khó khăn. Việc tìm ra tín hiệu trong tiếng ồn là một thách thức và ngày càng trở nên khó khăn hơn. Tuy nhiên, một số công cụ có thể trợ giúp: cụ thể là bạn có thể sử dụng Google Scholar (<https://scholar.google.com>) để theo dõi các ấn phẩm của các tác giả bạn yêu thích.

### Khám phá hệ sinh thái Keras

Với hơn 2,5 triệu người dùng tính đến đầu năm 2025 và vẫn đang phát triển, Keras có một hệ sinh thái lớn gồm các hướng dẫn, hướng dẫn và các dự án nguồn mở liên quan:

* Tài liệu tham khảo chính của bạn khi làm việc với Keras là tài liệu trực tuyến
tại <https://keras.io>. Đặc biệt, bạn sẽ tìm thấy rộng rãi
hướng dẫn dành cho nhà phát triển tại <https://keras.io/guides> và bạn sẽ tìm thấy hàng tá
các ví dụ về mã Keras chất lượng cao tại <https://keras.io/examples>.
Hãy chắc chắn kiểm tra chúng!

* Bạn có thể tìm thấy mã nguồn Keras tại <https://github.com/keras-team/keras>,
và Keras Hub có thể được tìm thấy tại <https://github.com/keras-team/keras-hub>.

* Bạn có thể theo dõi François (@fchollet) và Matt (@mattdangerw) trên X (trước đây là Twitter).

## Lời cuối cùng

Đến đây là kết thúc *Học sâu với Python*! Tôi hy vọng bạn đã học được một vài điều về học máy, học sâu, Keras và thậm chí có thể là nhận thức nói chung. Học tập là một hành trình suốt đời, đặc biệt là trong lĩnh vực AI, nơi chúng ta có nhiều điều chưa biết hơn là những bằng chứng xác thực. Vì vậy hãy tiếp tục học hỏi, đặt câu hỏi và nghiên cứu. Không bao giờ dừng lại. Bởi vì ngay cả với những tiến bộ đã đạt được cho đến nay, hầu hết các câu hỏi cơ bản về AI vẫn chưa được giải đáp. Nhiều người thậm chí còn chưa được hỏi đúng cách.

#### **Slide**

<div class="pdf-container" style="margin-bottom: 20px; width: 100%; height: 85vh;">
  <iframe src="TaiLieu/slideDL/Chapter20.pdf" width="100%" height="100%" style="border: none;"></iframe>
</div>

<!-- tabs:end -->
