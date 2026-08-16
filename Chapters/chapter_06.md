<!-- tabs:start -->

#### **Tiếng Anh (English)**

# Chapter 6: The universal workflow of machine learning

This chapter covers

* Framing a machine learning problem
* Developing a working model
* Deploying your model in production and maintaining it

Our previous examples have assumed that we already had a labeled dataset
to start from, and that we could immediately start training a model.
In the real world, this is often not the case. You don’t start from a dataset;
you start from a problem.

Imagine that you’re launching your own machine learning consulting shop. You
incorporate, you put up a fancy website, you notify your network.
The projects start rolling in:

* A personalized photo search engine for a picture-sharing social
  network — type in “wedding” and retrieve all the pictures you took at
  weddings, without any manual tagging needed.
* Flagging spam and offensive text content among the posts of a budding chat
  app.
* Building a music recommendation system for users of an online radio.
* Detecting credit card fraud for an e-commerce website.
* Predicting display ad click-through rate to decide which ad to serve to a
  given user at a given time.
* Flagging anomalous cookies on the conveyor belt of a cookie-manufacturing line.
* Using satellite images to predict the location of as-yet unknown
  archaeological sites.

It would be very convenient if you could import the correct dataset
from `keras.datasets` and start fitting some deep learning models.
Unfortunately, in the real world, you’ll have to start from scratch.

In this chapter, you’ll learn about the universal step-by-step
blueprint that you can use to approach and solve any machine learning problem,
like those previously listed. This template will bring together and
consolidate everything you’ve learned in chapters 4 and 5 and give you
the wider context that should anchor what you will learn in the next chapters.

The universal workflow of machine learning is broadly structured in three parts:

* *Define the task* — Understand the problem domain and the business logic
  underlying what the customer asked. Collect a dataset, understand
  what the data represents, and choose how you will measure success on the task.
* *Develop a model* — Prepare your data so that it can be processed by a machine
  learning model, select a model evaluation protocol and a simple baseline to
  beat, train a first model that has generalization power that can overfit, and then
  regularize and tune your model until you achieve the best possible
  generalization performance.
* *Deploy the model* — Present your work to stakeholders, ship the model
  to a web server, a mobile app, a web page, or an embedded device,
  monitor the model’s performance in the wild, and start collecting
  the data you’ll need to build the next model generation.

Let’s dive in.

## Defining the task

You can’t do good work without a deep understanding of the context of
what you’re doing. Why is your customer trying to solve this particular problem?
What value will they derive from the solution? How will your model be used?
How will it fit into your customer’s business processes?
What kind of data is available or could be collected?
What kind of machine learning task can be mapped to the business problem?

### Framing the problem

Framing a machine learning problem usually involves many detailed
discussions with stakeholders. Here are the questions that should be on top
of your min:

* What will your input data be? What are you trying to predict? You can only
  learn to predict something if you have available training data: for example,
  you can only learn to classify the sentiment of movie reviews if you have both
  movie reviews and sentiment annotations available. As such, data availability
  is usually the limiting factor at this stage. In many cases, you will have
  to resort to collecting and annotating new datasets yourself
  (which we cover in the next section).

* What type of machine learning task are you facing?
  Is it binary classification? Multiclass classification? Scalar regression?
  Vector regression? Multiclass, multilabel classification? Image segmentation?
  Ranking? Something else, like clustering, generation, or reinforcement
  learning? In some cases, it may be that machine
  learning isn’t even the best way to make sense of your data, and you should use
  something else, such as plain old-school statistical analysis:
  + The photo search engine project is a multiclass, multilabel
    classification task.
  + The spam detection project is a binary classification task.
    If you set “offensive content” as a separate class,
    it’s a three-way classification task.
  + The music recommendation engine turns out to be better handled not via
    deep learning, but via matrix factorization (collaborative filtering).
  + The credit card fraud detection project is a binary classification task.
  + The click-through rate prediction project is a scalar regression task.
  + Anomalous cookie detection is a binary classification task, but it
    will also require an object detection model as a first stage
    to correctly crop out the cookies in raw images. Note that
    the set of machine learning techniques known as “anomaly detection”
    would not be a good fit in this setting!
  + The project about finding new archaeological sites from satellite images
    is an image similarity ranking task: you need to retrieve new images
    that look the most like known archaeological sites.

* What do existing solutions look like? Perhaps your customer already has a
  hand-crafted algorithm that handles spam filtering or credit card fraud
  detection — with lots of nested `if` statements. Perhaps a human is currently
  in charge of manually handling the process considered — monitoring
  the conveyor belt at the cookie plant and manually removing the bad cookies,
  or crafting playlists of song recommendations to be sent out to users who
  liked a specific artist. You should make sure to understand what systems
  are already in place, and how they work.

* Are there particular constraints you will need to deal with? For example,
  you could find out that the app for which you’re building a spam detection
  system is strictly end-to-end encrypted, so that the spam detection model
  will have to live on the end-user’s phone,
  and must be trained on an external dataset.
  Perhaps the cookie-filtering model has such latency constraints that it
  will need to run on an embedded device at the factory rather than on a remote
  server. You should understand the full context in which your work will fit.

Once you’ve done your research, you should know what your inputs will be,
what your targets will be, and what broad type of machine learning task
the problem maps to. Be aware of the hypotheses you’re making at this stage:

* You hypothesize that your targets can be predicted given your inputs.
* You hypothesize that the data that’s available (or that you will soon collect)
  is sufficiently informative to learn the relationship between inputs
  and targets.

Until you have a working model, these are merely hypotheses, waiting to be
validated or invalidated. Not all problems can be solved with machine learning;
just because you’ve assembled examples of inputs X and targets Y
doesn’t mean X contains enough information to predict Y.
For instance, if you’re trying to predict the
movements of a stock on the stock market given its recent price history,
you’re unlikely to succeed, because price history doesn’t contain much
predictive information.

Note on ethics

You may sometimes be offered ethically dubious projects, such as
“building an AI that rates the trustworthiness of someone from a picture
of their face.” First of all, the validity of the project is in doubt:
it isn’t clear why trustworthiness would be reflected on someone’s face.
Second, such a task opens the door to all kinds of ethical problems.
Collecting a dataset for this task would amount to recording the biases
and prejudices of the people who label the pictures. The models you
would train on such data would be merely encoding these same biases —
into a black box algorithm, which would give them a thin veneer of legitimacy.
In a largely tech-illiterate society like ours,
“The AI algorithm said this person cannot be trusted” strangely appears to
carry more weight and objectivity than
“John Smith said this person cannot be trusted” — despite the former being
a learned approximation of the latter.
Your model would be laundering and operationalizing at scale the worst
aspects of human judgement, with negative effects on the lives of real people.

Technology is never neutral. If your work has any impact on the world,
then this impact has a moral direction: technical choices are also ethical
choices. Always be deliberate about the values you want your
work to support.

### Collecting a dataset

Once you understand the nature of the task and you know what your inputs and
targets are going to be, it’s time for data collection —
the most arduous, time-consuming, and costly part
of most machine learning projects:

* The photo search engine project requires you to first select the set
  of labels you want to classify — you settle on 10,000 common image
  categories. Then, you need to manually tag hundreds of thousands of your past
  user-uploaded images with labels from this set.
* For the chat app spam detection project, because user chats are end-to-end
  encrypted, you cannot use their contents for training a model.
  You need to gain access to a separate dataset of tens of thousands
  of unfiltered social media posts, and manually tag them as spam, offensive,
  or acceptable.
* For the music recommendation engine, you can just use the “likes” of your
  users. No new data needs to be collected. Likewise, for the click-through rate
  prediction project, you have an extensive record of click-through rate for your past ads,
  going back years.
* For the cookie-flagging model, you will need to install cameras above the
  conveyor belts to collect tens of thousands of images,
  and then someone will need to manually label
  these images. The people who know how to do this
  currently work at the cookie factory — but it doesn’t seem too difficult,
  you should be able to train people to do it.
* The satellite imagery project will require a team of archaeologists to
  collect a database of existing sites of interest, and for each site,
  you will need to find existing satellite images taken in different
  weather conditions. To get a good model, you’re going to need thousands
  of different sites.

You’ve learned in chapter 5 that a model’s ability to generalize comes almost
entirely from the properties of the data it is trained on — the number of
data points you have, the reliability of your labels,
the quality of your features. A good dataset is an asset worthy
of care and investment. If you get an extra 50 hours to spend on a project,
chances are that the most effective way to allocate them is to collect more
data, rather than search for incremental modeling improvements.

The point that data matters more than algorithms was most famously made
in a 2009 paper by Google researchers titled “The Unreasonable
Effectiveness of Data” (the title is a riff on the well-known 1960 book
*The Unreasonable Effectiveness of Mathematics in the Natural Sciences* by
Eugene Wigner). This was before deep learning was popular, but remarkably,
the rise of deep learning has only increased the importance of data.

If you’re doing supervised learning, then once you’ve collected inputs
(such as images) you’re going to need *annotations* for them (such as tags
for those images): the targets you will train your model to predict.

Sometimes, annotations can be
retrieved automatically — for instance, in the case of our music
recommendation task or our click-through rate prediction task.
But often, you have to annotate your data by hand.
This is a labor-heavy process.

#### Investing in data annotation infrastructure

Your data annotation process will determine the quality of your targets,
which, in turn, determines the quality of your model.
Carefully consider the options you have available:

* Should you annotate the data yourself?
* Should you use a crowdsourcing platform like Mechanical Turk to collect labels?
* Should you use the services of a specialized data-labeling company?

Outsourcing can potentially save you time and money
but takes away control. Using something like Mechanical Turk is likely to be
inexpensive and to scale well, but your annotations may end up being
quite noisy.

To pick the best option, consider the constraints you’re working with:

* Do the data labelers need to be subject matter experts, or could anyone
  annotate the data? The labels for a cat-versus-dog image classification
  problem can be selected by anyone, but those for a dog breed classification
  task require specialized knowledge. Meanwhile, annotating CT scans of
  bone fractures pretty much requires a medical degree.
* If annotating the data requires specialized knowledge,
  can you train people to do it? If not, how can you get access to
  relevant experts?
* Do you, yourself, understand the way experts come up with
  the annotations? If you don’t, you will have to treat
  your dataset as a black box, and you won’t be able to perform manual feature
  engineering — this isn’t critical, but it can be limiting.

If you decide to label your data in-house,
ask yourself what software you will use
to record annotations. You may well need to develop that software yourself.
Productive data annotation software will save you a lot of
time, so it’s something worth investing in early in a project.

#### Beware of nonrepresentative data

Machine learning models can only make sense of inputs that are similar to what
they’ve seen before. As such, it’s critical that the data used for training
should be *representative* of the production data. This concern should be
the foundation of all of your data collection work.

Suppose you’re developing an app where users can take pictures of a dish
to find out its name. You train a model using pictures from an
image-sharing social network that’s popular with foodies. Come deployment time,
and feedback from angry users starts rolling in: your app gets the answer
wrong 8 times out of 10. What’s going on? Your accuracy on the test
set was well over 90%! A quick look at user-uploaded data reveals that
mobile picture uploads of random dishes from random restaurants taken with
random smartphones look nothing like the professional-quality, well-lit,
appetizing pictures you trained the model on:
*your training data wasn’t representative of the production data*.
That’s a cardinal sin — welcome to machine learning hell.

If possible, collect data directly from the environment where your model
will be used. A movie review sentiment classification model should be used
on new IMDB reviews, not on Yelp restaurant reviews, nor on Twitter status updates.
If you want to rate the sentiment of a tweet, start by collecting and annotating
actual tweets — from a similar set of users as those you’re expecting
in production. If it’s not possible to train on production data,
then make sure you fully understand how your training and production data differ,
and that you are actively correcting these differences.

A related phenomenon you should be aware of is *concept drift*. You’ll
encounter concept drift in almost all real-world problems, especially those
that deal with user-generated data. Concept drift occurs when the properties
of the production data change over time, causing model accuracy to gradually
decay. A music recommendation engine trained in the year 2013 may not be
very effective today. Likewise, the IMDB dataset you worked with was collected
in 2011, and a model trained on it would likely not perform as well on reviews
from 2020 compared to reviews from 2012, as vocabulary, expressions, and movie
genres evolve over time. Concept drift is particularly acute in
adversarial contexts like credit card fraud detection,
where fraud patterns change practically every day.
Dealing with fast concept drift requires
constant data collection, annotation, and model retraining.

Keep in mind that machine learning can only be used to memorize patterns that
are present in your training data. You can only recognize what you’ve seen
before. Using machine learning trained on past data to predict the future is
making the assumption that the future will behave like the past. That often
isn’t the case.

The problem of sampling bias

A particularly insidious and common case of nonrepresentative data is
*sampling bias*. Sampling bias occurs when your data collection process
interacts with what you are trying to predict, resulting in biased measurements.
A famous historical example occurred in the 1948 US presidential election.
On election night, the Chicago Tribune printed the headline
“DEWEY DEFEATS TRUMAN.” The next morning, Truman emerged as the winner.
The editor of the Tribune had trusted the results of a phone survey —
but phone users in 1948 were not a random, representative sample of the
voting population. They were more likely to be richer, conservative, and to
vote for Dewey, the Republican candidate.
Nowadays, every phone survey takes sampling bias into account. That doesn’t
mean that sampling bias is a thing of the past in political polling — far from
it. But unlike in 1948, pollsters are aware of it and take steps to correct it.

![](../images/ch06/dewey_truman.015b2e12.jpg)


[Figure 6.1](#figure-6-1): “DEWEY DEFEATS TRUMAN”: a famous example of sampling bias

### Understanding your data

It’s bad practice to treat a dataset as a black box. Before you start
training models, you should explore and visualize your data to gain insights
about what makes it predictive — which will inform feature engineering — and
screen for potential issues:

* If your data includes images or natural language text, take a look at a few
  samples (and their labels) directly.
* If your data contains numerical features, it’s a good idea to plot the
  histogram of feature values to get a feel for the range of values taken
  and the frequency of different values.
* If your data includes location information, plot it on a map. Do any
  clear patterns emerge?
* Are some samples missing values for some features? If so, you’ll need
  to deal with this when you prepare the data (we cover how to do this
  in the next section).
* If your task is a classification problem, print the number of instances
  of each class in your data. Are the classes roughly equally represented?
  If not, you will need to account for this imbalance.
* Check for *target leaking* — the presence of features in your data that
  provide information about the targets that may not be available in
  production. If you’re training a model on medical records to predict
  whether someone will be treated for cancer in the future, and the records
  include the feature “This person has been diagnosed with cancer,” then
  your targets are being artificially leaked into your data.
  Always ask yourself, is every feature in your data something that will be
  available in the same form in production?

### Choosing a measure of success

To control something, you need to be able to observe it.
To achieve success on a project, you must first define what you mean by success.
Accuracy? Precision and recall? Customer retention rate?
Your metric for success will guide all of the technical choices you
will make throughout the project. It should directly align with
your higher-level goals, such as the business success of your customer.

For balanced classification problems, where every class is equally likely,
accuracy and *area under curve* (AUC) of the *receiver operating characteristic*
(ROC) are common metrics. For class-imbalanced problems, ranking problems, or
multilabel classification, you can use precision
and recall or a metric that counts false positives, true positives,
false negatives, and true negatives.
And it isn’t uncommon to have to define your own
custom metric by which to measure success. To get a sense of the diversity of
machine learning success metrics and how they relate to different problem
domains, it’s helpful to browse the data science competitions on Kaggle
(<https://kaggle.com>); it showcases a wide range of problems and evaluation
metrics.

## Developing a model

Once you know how you will measure your progress, you can get started with
model development. Most tutorials and research projects assume that this
is the only step — skipping problem definition and dataset collection,
which are assumed to be already done, and skipping model deployment and maintenance,
which is assumed to be handled by someone else. In fact, model development
is only *one step* in the machine learning workflow, and if you ask me,
it’s not the most difficult one. The hardest things in machine learning
are framing problems and collecting, annotating, and cleaning data. So cheer
up, what comes next will be easy in comparison!

### Preparing the data

As you’ve learned before, deep learning models typically don’t ingest raw data.
Data preprocessing aims at making the raw data at hand more amenable
to neural networks. This includes vectorization, normalization,
or handling missing values. Many preprocessing techniques are domain specific
(for example, specific to text data or image data); we’ll
cover those in the following chapters as we encounter them in practical
examples. For now, we’ll review the basics that are common to all data domains.

#### Vectorization

All inputs and targets in a neural network must be typically tensors
of floating-point data (or, in specific cases, tensors of integers or strings).
Whatever data you need to process — sound, images, text —
you must first turn into tensors, a step called *data vectorization*.
For instance, in the two previous
text classification examples from chapter 4, we started from text represented
as lists of integers (standing for sequences of words),
and we used multi-hot encoding to turn them into a tensor of `float32` data.
In the examples of classifying digits and predicting house prices,
the data already came in vectorized form, so you were able to skip this step.

#### Value normalization

In the MNIST digit-classification example from chapter 2, you started from
image data encoded as integers in the 0–255 range, encoding grayscale values.
Before you fed this data into your network, you had to cast it to `float32`
and divide by 255 so you’d end up with floating-point values in
the 0–1 range. Similarly, when predicting house prices, you started from
features that took a variety of ranges — some features had small floating-point
values, others had fairly large integer values. Before you fed this data into
your network, you had to normalize each feature independently so that it had a
standard deviation of 1 and a mean of 0.

In general, it isn’t safe to feed into a neural network data that takes
relatively large values (for example, multi-digit integers, which are much
larger than the initial values taken by the weights of a network) or
data that is heterogeneous (for example, data where one
feature is in the range 0–1, and another is in the range 100–200). Doing so can
trigger large gradient updates that will prevent the
network from converging. To make learning easier for your network, your data
should have the following characteristics:

* *Take small values*  — Typically, most values should be in the 0–1 range.
* *Be homogeneous* — That is, all features should take values in roughly
  the same range.

Additionally, the following stricter normalization practice is common and can
help, although it isn’t always necessary (for example, you didn’t do this in
the digit-classification example):

* Normalize each feature independently to have a mean of 0.
* Normalize each feature independently to have a standard deviation of 1.

This is easy to do with NumPy arrays:

```python
# Assuming x is a 2D data matrix of shape (samples, features)
x -= x.mean(axis=0)
x /= x.std(axis=0)
```

#### Handling missing values

You may sometimes have missing values in your data. For instance, in the
house price example, the second feature was the median age of houses in the
district. What if this feature wasn’t available for all samples? You’d then have
missing values in the training or test data.

You could just discard the feature entirely, but you don’t necessarily have to:

* If the feature is categorical, it’s safe to create a new category that means
  “the value is missing.” The model will automatically learn what this implies
  with respect to the targets.
* If the feature is numerical, avoid inputting an arbitrary value like 0
  because it may create a discontinuity in the latent space formed by
  your features, making it harder for a model trained on it to
  generalize. Instead, consider replacing the missing value with the
  average or median value for the feature in the dataset. You could also train
  a model to predict the feature value given the values of other features.

Note that if you’re expecting missing categorical features in the test data,
but the network was trained on data without any missing values,
the network won’t have learned to ignore missing values! In this situation,
you should artificially generate training samples with missing entries:
copy some training samples several times, and drop some of the categorical
features that you expect are likely to be missing in the test data.

### Choosing an evaluation protocol

As you’ve learned in the previous chapter, the purpose of a model is to achieve
generalization, and every modeling decision you will make throughout the
model development process will be guided by *validation metrics* that seek
to measure generalization performance. The goal of your validation protocol is
to accurately estimate what your success metric of choice (such as accuracy)
will be on actual production data. The reliability of that process
is critical to building a useful model.

In chapter 5, we’ve reviewed three common evaluation protocols:

* *Maintaining a hold-out validation set*  — The way to go when you have
  plenty of data

* *Doing K-fold cross-validation*  — The right choice when you have too few
  samples for hold-out validation to be reliable

* *Doing iterated K-fold validation*  — For performing highly accurate model
  evaluation when little data is available

Just pick one of these. In most cases, the first will work well enough. As
you’ve learned before, always be mindful of the *representativity*
of your validation set(s) and be careful not to have redundant samples between
your training set and your validation set(s).

### Beating a baseline

As you start working on the model itself, your initial goal is
to achieve *statistical power*, as you saw in chapter 5 — that is,
to develop a small model that is capable of beating a simple baseline.

At this stage, these are the three most important things you should focus on:

* *Feature engineering* — Filter out uninformative features (feature selection)
  and use your knowledge of the problem to develop new features
  that are likely to be useful.
* *Selecting the correct architecture priors* — What type of model architecture
  will you use? A densely connected network, a ConvNet, a recurrent neural
  network, a Transformer? Is deep learning even a good approach for the task,
  or should you use something else?
* *Selecting a good enough training configuration* — What loss function should
  you use? What batch size and learning rate?

Picking the right loss function

It’s often not possible to
directly optimize for the metric that measures success on a problem. Sometimes
there is no easy way to turn a metric into a loss function; loss functions,
after all, need to be computable given only a mini-batch of data (ideally, a
loss function should be computable for as little as a single data point) and
must be differentiable (otherwise, you can’t use backpropagation to train your
network). For instance, the widely used classification metric ROC AUC can’t be
directly optimized. Hence, in classification tasks, it’s common to optimize
for a proxy metric of ROC AUC, such as crossentropy. In general, you can hope
that the lower the crossentropy gets, the higher the ROC AUC will be.

Table 6.1 can help you choose a last-layer activation, a loss function, and metrics
for a few common problem types.

| Task | Last-layer activation | Loss function | Metrics |
| --- | --- | --- | --- |
| Binary classification | Sigmoid | Binary crossentropy | Binary accuracy, ROC AUC |
| Multiclass, single-label classification | Softmax | Categorical crossentropy | Categorical accuracy, top-k categorical accuracy, ROC AUC |
| Multiclass, multi-label classification | Sigmoid | Binary crossentropy | Binary accuracy, ROC AUC |
| Regression | None | Mean squared error | Mean absolute error |

[Table 6.1](#table-6-1): Which loss, last-layer activation, and metrics to use for different tasks

For most problems, there are existing templates you can start from. You’re
not the first person to try to build a spam detector, a music recommendation
engine, or an image classifier. Make sure to research prior art to identify
the feature engineering techniques and model architectures that are most
likely to perform well on your task.

Note that it’s not always possible to achieve statistical power. If you can’t
beat a simple baseline after trying multiple reasonable architectures, it may
be that the answer to the question you’re asking isn’t present in the input
data. Remember that you’re making two hypotheses:

* You hypothesize that your outputs can be predicted given your inputs.
* You hypothesize that the available data is sufficiently informative to learn
  the relationship between inputs and outputs.

It may well be that these hypotheses are false, in which case you must go back
to the drawing board.

### Scale up: developing a model that overfits

Once you’ve obtained a model that has statistical power,
the question becomes, is your model sufficiently powerful?
Does it have enough layers and parameters to properly
model the problem at hand? For instance, a logistic regression model has
statistical power on MNIST but wouldn’t be
sufficient to solve the problem well. Remember that the universal tension in
machine learning is between optimization and generalization; the ideal model
is one that stands right at the border between underfitting and overfitting,
between undercapacity and overcapacity. To figure out where this border lies,
first you must cross it.

To figure out how big a model you’ll need, you must develop a model that
overfits. This is fairly easy, as you learned in chapter 5:

* Add layers.
* Make the layers bigger.
* Train for more epochs.

Always monitor the training loss and validation loss, as well as the training
and validation values for any metrics you care about. When you see that the
model’s performance on the validation data begins to degrade, you’ve achieved
overfitting.

### Regularizing and tuning your model

Once you’ve achieved statistical power and you’re able to overfit, you know
you’re on the right path. At this point, your goal becomes to
maximize generalization performance.

This phase will take the most time: you’ll repeatedly modify your model,
train it, evaluate on your validation data (not the test data, at this point),
modify it again, and repeat, until the model is as good as it can get.
Here are some things you should try:

* Try different architectures; add or remove layers.
* Add dropout.
* If your model is small, add L1 or L2 regularization.
* Try different hyperparameters (such as the number of units per layer or the
  learning rate of the optimizer) to find the optimal configuration.
* Optionally, iterate on data curation or feature engineering:
  collect and annotate more data, develop better features,
  or remove features that don’t seem to be informative.

It’s possible to automate a large chunk of this work by using
*automated hyperparameter tuning software*, such as KerasTuner. We’ll cover
this in chapter 18.

Be mindful of the following: every time you use feedback from your validation
process to tune your model, you leak information about the validation process
into the model. Repeated just a few times, this is innocuous; however, done
systematically over many iterations, it will eventually cause your model to
overfit to the validation process (even though no model is directly trained on
any of the validation data). This makes the evaluation process less reliable.

Once you’ve developed a satisfactory model configuration, you can train your
final production model on all the available data (training and validation) and
evaluate it one last time on the test set. If it turns out that performance on
the test set is significantly worse than the performance measured on the
validation data, this may mean either that your validation procedure wasn’t
reliable after all, or that you began overfitting to the validation data while
tuning the parameters of the model. In this case, you may want to switch to a
more reliable evaluation protocol (such as iterated K-fold validation).

## Deploying your model

After your model has successfully cleared its final evaluation on the test set,
it’s ready to be deployed and to begin its productive life.

### Explaining your work to stakeholders and setting expectations

Success and customer trust are about consistently meeting or exceeding people’s
expectations; the actual system you deliver is only half of that picture.
The other half is setting appropriate expectations before launch.

The expectations of nonspecialists toward AI systems are often unrealistic.
For example, they might expect that the system “understands”
its task and is capable of exercising human-like common sense in the
context of the task. To address this, you should consider showing some examples
of the *failure modes* of your model
(for instance, show what incorrectly classified samples look like,
especially those for which the misclassification seems surprising).

They might also expect human-level performance, especially for processes
that were previously handled by people. Most machine learning models,
because they are (imperfectly) trained to approximate human-generated
labels, do not nearly get there. You should clearly convey model performance
expectations. Avoid using abstract statements like “The model has 98% accuracy”
(which most people mentally round up to 100%),
and prefer talking, for instance, about false-negative rates and false-
positive rates. You could say, “With these settings,
the fraud detection model would have a 5%
false-negative rate and a 2.5% false-positive rate. Every day, an average of
200 valid transactions would be flagged as fraudulent and sent for manual
review, and an average of 14 fraudulent transactions would be missed. An
average of 266 fraudulent transactions would be correctly caught.”
Clearly relate the model’s performance metrics to business goals.

You should also make sure to discuss with stakeholders the choice of key
launch parameters — for instance, the probability threshold at which a
transaction should be flagged (different thresholds will
produce different false-negative and false-positive rates).
Such decisions involve tradeoffs that can only be handled with a
deep understanding of the business context.

### Shipping an inference model

A machine learning project doesn’t end when you arrive at a Colab notebook
that can save a trained model. You rarely put into production the exact same
Python model object that you manipulated during training.

First, you may want to export your model to something other than Python:

* Your production environment may not support Python at all — for instance,
  if it’s a mobile app or an embedded system.
* If the rest of the app isn’t in Python (it could be in JavaScript,
  C++, etc.), the use of Python to serve a model may induce
  significant overhead.

Second, since your production model will only be used to output predictions
(a phase called *inference*), rather than for training,
you have room to perform various optimizations that can make the model
faster and reduce its memory footprint.

Let’s take a quick look at the different model deployment options you
have available.

#### Deploying a model as a REST API

Perhaps the easiest way to turn a model into a product is to serve it online via
a REST API. There are a number of libraries out there for making this happen. Keras
supports two of the most popular approaches out of the box — *TensorFlow
Serving* and *ONNX* (short for Open Neural Network Exchange). Both libraries
operate by lifting all model weights and a computation graph outside of the
Python program, so you can serve it from a number of different environments
(for example, a C++ server). If this sounds a lot like the compilation mechanism
discussed in chapter 3 you are spot-on. TensorFlow Serving is essentially a
library for serving `tf.function` computation graphs with a specific set of
saved weights.

Keras allows access to both TensorFlow Serving and ONNX via an easy-to-use
`export()` method available on all Keras models. Here’s a code snippet showing
how this works for TensorFlow Serving:

```python
# Exports the model as a TensorFlow SavedModel artifact
model.export("path/to/location", format="tf_saved_model")

# Loads the artifact in a different process, environment, or
# programming language
reloaded_artifact = tf.saved_model.load("path/to/location")
predictions = reloaded_artifact.serve(input_data)
```

A similar flow exists for ONNX:

```python
model.export("path/to/location", format="onnx")

ort_session = onnxruntime.InferenceSession("path/to/location")
predictions = ort_session.run(None, input_data)
```

You should use this deployment setup when

* The application that will consume the model’s prediction will have reliable
  access to the internet (obviously). For instance,
  if your application is a mobile app, serving predictions from a remote API
  means that the application won’t be usable in airplane mode or in a
  low-connectivity environment.
* The application does not have strict latency requirements: the request,
  inference, and answer round trip will typically take around 500 ms.
* The input data sent for inference is not highly sensitive: the data will need
  to be available on the server in a decrypted form, since it will need to be
  seen by the model (but note that you should use SSL encryption for
  the HTTP request and answer).

For instance, the image search engine project, the music recommender system,
the credit card fraud detection project, and the satellite imagery project
are all a good fit for serving via a REST API.

An important question when deploying a model as a REST API is whether you
want to host the code on your own or whether you want to use a fully managed
third-party cloud service. For instance, Cloud AI Platform, a Google product,
lets you simply upload your TensorFlow model to Google Cloud Storage (GCS)
and gives you an API endpoint to query it. It takes care of many practical
details such as batching predictions, load balancing, and scaling.

#### Deploying a model on a device

Sometimes, you may need your model to live on the same device that runs the
application that uses it — maybe a smartphone, an embedded ARM CPU on a robot,
or a microcontroller on a tiny device. For instance, perhaps you’ve already
seen a camera capable of automatically detecting people and faces in the scenes
you pointed it at: that was probably a small deep learning model running
directly on the camera.

You should use this setup when

* Your model has strict latency constraints or needs to run in a
  low-connectivity environment. If you’re building an immersive
  augmented-reality application, querying a remote server is not a viable
  option.
* Your model can be made sufficiently small that it can run under the
  memory and power constraints of the target device.
* Getting the highest possible accuracy isn’t mission critical for your task:
  there is always a tradeoff between runtime efficiency and accuracy, so
  memory and power constraints often require you to ship a model that isn’t
  quite as good as the best model you could run on a large GPU.
* The input data is strictly sensitive and thus shouldn’t be decryptable
  on a remote server.

For instance, our spam detection model will need to run on the end user’s
smartphone as part of the chat app, because messages are end-to-end encrypted
and thus cannot be read by a remotely hosted model at all. Likewise, the
bad-cookie-detection model has strict latency constraints and will need to
run at the factory. Thankfully, in this case, we don’t have any power or space
constraints, so we can actually run the model on a GPU.

To deploy a Keras model on a smartphone or embedded device, you can again use
the `export()` method to create a TensorFlow or ONNX save of your model
including the computation graph. TensorFlow Lite (<https://www.tensorflow.org/lite>)
is a framework for efficient on-device deep learning inference that runs on
Android and iOS smartphones, as well as ARM CPUs, Raspberry Pi, or
certain microcontrollers. It uses the same TensorFlow save model format as
TensorFlow Serving. The ONNX runtime can also run on mobile devices.

#### Deploying a model in the browser

Deep learning is often used in browser-based or desktop-based JavaScript
applications. While it is usually possible to have the application query a
remote model via a REST API, there can be key advantages in instead having
the model run directly in the browser, on the user’s computer
(utilizing GPU resources if available).

Use this setup when

* You want to offload compute to the end user, which can
  dramatically reduce server costs.
* The input data needs to stay on the end user’s computer or phone.
  For instance, in our spam detection project, the web version and the
  desktop version of the chat app (implemented as a cross-platform app
  written in JavaScript) should use a locally run model.
* Your application has strict latency constraints: while a model running on the
  end user’s laptop or smartphone is likely to be slower than one running on
  a large GPU on your own server, you don’t have the extra 100 ms of network
  round trip.
* You need your app to keep working without connectivity, after the model has
  been downloaded and cached.

Of course, you should only go with this option if your model is small enough
that it won’t hog the CPU, GPU, or RAM of your user’s laptop or smartphone. In
addition, since the entire model will be downloaded to the user’s device, you
should make sure that nothing about the model needs to stay confidential. Be
mindful of the fact that, given a trained deep learning model,
it is usually possible to recover some information about the training data:
better not to make your trained model public if it was trained on sensitive data.

To deploy a model in JavaScript, the TensorFlow ecosystem includes
TensorFlow.js (<https://www.tensorflow.org/js>), and ONNX supports a native
JavaScript runtime. TensorFlow.js even implements almost all of the
Keras API (it was originally developed under the working name WebKeras)
as well as many lower-level TensorFlow APIs. You can easily import a saved Keras
model into TensorFlow.js to query it as part of your browser-based JavaScript
app or your desktop Electron app.

#### Inference model optimization

Optimizing your model for inference is especially important when deploying
in an environment with strict constraints on available power and memory
(smartphones and embedded devices) or for applications with low-latency
requirements. You should always seek to optimize your model before
importing it into TensorFlow.js or exporting it to TensorFlow Lite.

There are two popular optimization techniques you can apply:

* *Weight pruning* — Not every coefficient in a weight tensor contributes
  equally to the predictions. It’s possible to considerably lower the
  number of parameters in the layers of your model by only keeping the
  most significant ones.
  This reduces the memory and compute footprint of your model
  at a small cost in performance metrics. By tuning how much pruning you
  want to apply, you are in control of the tradeoff between size and accuracy.
* *Weight quantization* — Deep learning models are trained with single-precision
  floating-point (`float32`) weights.
  However, it’s possible to *quantize* weights to 8-bit signed integers
  (`int8`) to get an inference-only
  model that’s four times smaller but remains near the accuracy
  of the original model. Keras models come with a built-in `quantize()` API
  that can help with this. Simply call `model.quantize("int8")` to compress
  each weight in your model to a single byte.

### Monitoring your model in the wild

You’ve exported an inference model, you’ve integrated it into your application,
and you’ve done a dry run on production data — the model behaved exactly
as you expected. You’ve written unit tests as well
as logging and status-monitoring code — perfect.
Now it’s time to press the big red button and deploy to production.

Even this is not the end. Once you’ve deployed a model, you need to keep
monitoring its behavior, its performance on new data, its interaction with
the rest of the application, and its eventual impact on business metrics:

* Is user engagement in your online radio up or down after deploying the new
  music recommender system? Has average ad click-through rate increased after switching to
  the new click-through rate prediction model? Consider using *randomized A/B testing*
  to isolate the impact of the model itself from other changes: a
  subset of cases should go through the new model, while another control subset
  should stick to the old process. Once sufficiently many cases have been
  processed, the difference in outcomes
  between the two is likely attributable to the model.
* If possible, do a regular manual audit of the model’s predictions
  on production data. It’s generally possible to reuse the same infrastructure
  as for data annotation: send some fraction of the production data to be
  manually annotated and compare the model’s predictions to the new annotations.
  For instance, you should definitely do this for the image search engine
  and the bad-cookie-flagging system.
* When manual audits are impossible, consider alternative evaluation avenues
  such as user surveys (for example, in the case of the spam and offensive
  content–flagging system).

### Maintaining your model

Lastly, no model lasts forever. You’ve already learned about *concept drift*:
over time, the characteristics of your production data will change, gradually
degrading the performance and relevance of your model.
The lifespan of your music recommender system will be counted in weeks. For the
credit card fraud detection system, it would be days; a couple of years in the
best case for the image search engine.

As soon as your model has launched, you should be getting ready to train
the next generation that will replace it:

* Watch out for changes in the production data. Are new features
  becoming available? Should you expand or otherwise edit the label set?
* Keep collecting and annotating data, and keep
  improving your annotation pipeline over time. In particular, you should
  pay special attention to collecting samples that seem to be difficult
  to classify for your current model — such samples are the most likely
  to help improve performance.

This concludes the universal workflow of machine learning — that’s a lot of
things to keep in mind. It takes time and experience to become an expert,
but don’t worry, you’re already a lot wiser than you were a few chapters ago.
You are now familiar with the big picture — the entire spectrum of what
machine learning projects entail.
While most of this book will focus on the model development part, you’re now
aware that it’s only one part of the entire workflow. Always keep in mind
the big picture!

## Summary

* When you take on a new machine learning project, first, define the problem at hand:
  + Understand the broader context of what you’re setting out to do —
    what’s the end goal and what are the constraints?
  + Collect and annotate a dataset; make sure you understand your data
    in depth.
  + Choose how you’ll measure success on your problem. What metrics will you
    monitor on your validation data?
* Once you understand the problem and you have an appropriate dataset, develop a model:
  + Prepare your data.
  + Pick your evaluation protocol. Hold-out validation? K-fold validation?
    Which portion of the data should you use for validation?
  + Achieve statistical power: beat a simple baseline.
  + Scale up: develop a model that can overfit.
  + Regularize your model and tune its hyperparameters, based on performance on
    the validation data. A lot of machine learning research tends to focus only
    on this step — but keep the big picture in mind.
* When your model is ready and yields good performance on the test data,
  it’s time for deployment:
  + First, make sure to set appropriate expectations with stakeholders.
  + Optimize a final model for inference, and ship the model to the deployment
    environment of choice — web server, mobile, browser, embedded device, etc.
  + Monitor your model’s performance in production and keep collecting data
    so you can develop the next generation of the model.

#### **Tiếng Việt (Vietnamese)**

# Chương 6: Quy trình làm việc phổ biến của học máy

Chương này bao gồm

* Đóng khung một vấn đề học máy
* Phát triển mô hình làm việc
* Triển khai mô hình của bạn trong sản xuất và duy trì nó

Các ví dụ trước đây của chúng tôi đã giả định rằng chúng tôi đã có một tập dữ liệu được gắn nhãn để bắt đầu và chúng tôi có thể bắt đầu đào tạo mô hình ngay lập tức. Trong thế giới thực, điều này thường không xảy ra. Bạn không bắt đầu từ một tập dữ liệu; bạn bắt đầu từ một vấn đề.

Hãy tưởng tượng rằng bạn đang khai trương cửa hàng tư vấn về máy học của riêng mình. Bạn kết hợp, bạn thiết lập một trang web ưa thích, bạn thông báo cho mạng lưới của mình. Các dự án bắt đầu triển khai:

* Công cụ tìm kiếm ảnh được cá nhân hóa dành cho mạng xã hội chia sẻ ảnh
mạng - nhập “đám cưới” và truy xuất tất cả các bức ảnh bạn đã chụp tại
đám cưới mà không cần gắn thẻ thủ công.
* Gắn cờ nội dung thư rác và văn bản xúc phạm trong số các bài đăng của cuộc trò chuyện mới chớm nở
ứng dụng.
* Xây dựng hệ thống gợi ý âm nhạc cho người dùng radio trực tuyến.
* Phát hiện gian lận thẻ tín dụng cho một trang web thương mại điện tử.
* Dự đoán tỷ lệ nhấp vào quảng cáo hiển thị hình ảnh để quyết định quảng cáo nào sẽ phân phát tới
người dùng nhất định tại một thời điểm nhất định.
* Gắn cờ các bánh quy dị thường trên băng chuyền của dây chuyền sản xuất bánh quy.
* Sử dụng hình ảnh vệ tinh để dự đoán vị trí chưa được biết
các địa điểm khảo cổ.

Sẽ rất thuận tiện nếu bạn có thể nhập đúng tập dữ liệu từ `keras.datasets` và bắt đầu điều chỉnh một số mô hình học sâu. Thật không may, trong thế giới thực, bạn sẽ phải bắt đầu lại từ đầu.

Trong chương này, bạn sẽ tìm hiểu về kế hoạch chi tiết từng bước phổ quát mà bạn có thể sử dụng để tiếp cận và giải quyết bất kỳ vấn đề nào về máy học, giống như những vấn đề được liệt kê trước đó. Mẫu này sẽ tập hợp và củng cố mọi thứ bạn đã học trong chương 4 và 5, đồng thời cung cấp cho bạn bối cảnh rộng hơn để củng cố những gì bạn sẽ học trong các chương tiếp theo.

Quy trình làm việc chung của học máy được cấu trúc rộng rãi thành ba phần:

* *Xác định nhiệm vụ* - Hiểu miền vấn đề và logic nghiệp vụ
cơ bản những gì khách hàng yêu cầu. Thu thập một tập dữ liệu, hiểu
dữ liệu thể hiện điều gì và chọn cách bạn sẽ đo lường mức độ thành công của nhiệm vụ.
* *Phát triển mô hình* — Chuẩn bị dữ liệu của bạn để máy có thể xử lý dữ liệu đó
mô hình học tập, chọn một giao thức đánh giá mô hình và một đường cơ sở đơn giản để
đánh bại, huấn luyện mô hình đầu tiên có khả năng khái quát hóa có thể phù hợp quá mức, sau đó
thường xuyên hóa và điều chỉnh mô hình của bạn cho đến khi bạn đạt được kết quả tốt nhất có thể
hiệu suất khái quát hóa.
* *Triển khai mô hình* — Trình bày công việc của bạn cho các bên liên quan, gửi mô hình
tới máy chủ web, ứng dụng dành cho thiết bị di động, trang web hoặc thiết bị nhúng,
theo dõi hiệu suất của mô hình trong thực tế và bắt đầu thu thập
dữ liệu bạn sẽ cần để xây dựng thế hệ mô hình tiếp theo.

Hãy đi sâu vào.

## Xác định nhiệm vụ

Bạn không thể làm tốt công việc nếu không hiểu biết sâu sắc về bối cảnh của việc bạn đang làm. Tại sao khách hàng của bạn lại cố gắng giải quyết vấn đề cụ thể này? Họ sẽ thu được giá trị gì từ giải pháp? Mô hình của bạn sẽ được sử dụng như thế nào? Nó sẽ phù hợp với quy trình kinh doanh của khách hàng như thế nào? Loại dữ liệu nào có sẵn hoặc có thể được thu thập? Loại nhiệm vụ học máy nào có thể được ánh xạ tới vấn đề kinh doanh?

### Đóng khung vấn đề

Việc định hình một vấn đề về học máy thường bao gồm nhiều cuộc thảo luận chi tiết với các bên liên quan. Dưới đây là những câu hỏi bạn nên đặt lên hàng đầu:

* Dữ liệu đầu vào của bạn sẽ là gì? Bạn đang cố gắng dự đoán điều gì? Bạn chỉ có thể
học cách dự đoán điều gì đó nếu bạn có sẵn dữ liệu đào tạo: ví dụ:
bạn chỉ có thể học cách phân loại cảm xúc của các bài đánh giá phim nếu bạn có cả hai
đánh giá phim và chú thích tình cảm có sẵn. Như vậy, tính sẵn có của dữ liệu
thường là yếu tố hạn chế ở giai đoạn này. Trong nhiều trường hợp, bạn sẽ có
phải tự mình thu thập và chú thích các tập dữ liệu mới
(mà chúng tôi sẽ trình bày trong phần tiếp theo).

* Bạn đang phải đối mặt với loại nhiệm vụ học máy nào?
Đây có phải là phân loại nhị phân? Phân loại nhiều lớp? Hồi quy vô hướng?
Hồi quy vectơ? Phân loại đa lớp, đa nhãn? Phân đoạn hình ảnh?
Xếp hạng? Một cái gì đó khác, như phân cụm, tạo hoặc củng cố
học hỏi? Trong một số trường hợp, có thể là máy đó
học thậm chí không phải là cách tốt nhất để hiểu dữ liệu của bạn và bạn nên sử dụng
một cái gì đó khác, chẳng hạn như phân tích thống kê kiểu cũ đơn giản:
+ Dự án công cụ tìm kiếm ảnh là một dự án đa lớp, đa nhãn
nhiệm vụ phân loại.
+ Dự án phát hiện thư rác là một nhiệm vụ phân loại nhị phân.
Nếu bạn đặt “nội dung xúc phạm” thành một lớp riêng biệt,
đó là một nhiệm vụ phân loại ba chiều.
+ Công cụ đề xuất âm nhạc được xử lý tốt hơn không thông qua
học sâu, nhưng thông qua hệ số hóa ma trận (lọc cộng tác).
+ Dự án phát hiện gian lận thẻ tín dụng là một nhiệm vụ phân loại nhị phân.
+ Dự án dự đoán tỷ lệ nhấp chuột là một nhiệm vụ hồi quy vô hướng.
+ Phát hiện cookie bất thường là một nhiệm vụ phân loại nhị phân, nhưng nó
cũng sẽ yêu cầu một mô hình phát hiện đối tượng ở giai đoạn đầu tiên
để cắt cookie một cách chính xác trong ảnh thô. Lưu ý rằng
tập hợp các kỹ thuật học máy được gọi là “phát hiện bất thường”
sẽ không phù hợp trong bối cảnh này!
+ Dự án tìm kiếm địa điểm khảo cổ mới từ ảnh vệ tinh
là một nhiệm vụ xếp hạng độ tương tự của hình ảnh: bạn cần truy xuất hình ảnh mới
trông giống những địa điểm khảo cổ được biết đến nhất.

* Các giải pháp hiện tại trông như thế nào? Có lẽ khách hàng của bạn đã có
thuật toán thủ công xử lý việc lọc thư rác hoặc gian lận thẻ tín dụng
phát hiện - với rất nhiều câu lệnh `if` lồng nhau. Có lẽ hiện nay con người đang
phụ trách xử lý thủ công quy trình được xem xét - giám sát
băng chuyền tại nhà máy sản xuất bánh quy và loại bỏ những chiếc bánh quy hỏng theo cách thủ công,
hoặc tạo danh sách đề xuất bài hát để gửi cho người dùng
thích một nghệ sĩ cụ thể. Bạn nên đảm bảo hiểu rõ hệ thống nào
đã có sẵn và cách chúng hoạt động.

* Có những hạn chế cụ thể nào bạn sẽ cần phải giải quyết không? Ví dụ,
bạn có thể phát hiện ra rằng ứng dụng mà bạn đang xây dựng tính năng phát hiện thư rác
hệ thống được mã hóa nghiêm ngặt từ đầu đến cuối, do đó mô hình phát hiện thư rác
sẽ phải tồn tại trên điện thoại của người dùng cuối,
và phải được đào tạo trên tập dữ liệu bên ngoài.
Có lẽ mô hình lọc cookie có những hạn chế về độ trễ đến mức nó
sẽ cần chạy trên một thiết bị nhúng tại nhà máy thay vì trên điều khiển từ xa
máy chủ. Bạn nên hiểu toàn bộ bối cảnh mà công việc của bạn sẽ phù hợp.

Sau khi thực hiện nghiên cứu của mình, bạn nên biết đầu vào của mình sẽ là gì, mục tiêu của bạn sẽ là gì và loại nhiệm vụ học máy rộng rãi mà vấn đề sẽ hướng tới. Hãy nhận biết các giả thuyết bạn đang đưa ra ở giai đoạn này:

* Bạn đưa ra giả thuyết rằng mục tiêu của bạn có thể được dự đoán dựa trên thông tin đầu vào của bạn.
* Bạn đưa ra giả thuyết rằng dữ liệu có sẵn (hoặc dữ liệu bạn sẽ sớm thu thập)
có đủ thông tin để tìm hiểu mối quan hệ giữa các yếu tố đầu vào
và các mục tiêu.

Cho đến khi bạn có một mô hình hoạt động được, đây chỉ là những giả thuyết, đang chờ được xác thực hoặc vô hiệu. Không phải tất cả các vấn đề đều có thể được giải quyết bằng học máy; chỉ vì bạn đã tập hợp các ví dụ về đầu vào X và mục tiêu Y không có nghĩa là X chứa đủ thông tin để dự đoán Y. Ví dụ: nếu bạn đang cố gắng dự đoán chuyển động của một cổ phiếu trên thị trường chứng khoán dựa trên lịch sử giá gần đây của nó, thì bạn khó có thể thành công vì lịch sử giá không chứa nhiều thông tin mang tính dự đoán.

Lưu ý về đạo đức

Đôi khi, bạn có thể được giao những dự án đáng ngờ về mặt đạo đức, chẳng hạn như “xây dựng một AI đánh giá độ tin cậy của ai đó từ hình ảnh khuôn mặt của họ”. Trước hết, tính hợp lệ của dự án đang bị nghi ngờ: không rõ tại sao sự đáng tin cậy lại được phản ánh trên khuôn mặt của một ai đó. Thứ hai, một nhiệm vụ như vậy sẽ mở ra cánh cửa cho mọi loại vấn đề đạo đức. Việc thu thập một tập dữ liệu cho nhiệm vụ này sẽ giống như việc ghi lại những thành kiến ​​và thành kiến ​​của những người dán nhãn cho các bức ảnh. Các mô hình mà bạn đào tạo dựa trên dữ liệu đó sẽ chỉ mã hóa những thành kiến ​​​​tương tự này - thành một thuật toán hộp đen, điều này sẽ mang lại cho chúng một lớp vỏ hợp pháp mỏng manh. Trong một xã hội phần lớn mù chữ về công nghệ như xã hội của chúng ta, “Thuật toán AI cho biết người này không thể tin cậy được” dường như có sức nặng và tính khách quan hơn một cách kỳ lạ so với “John Smith nói rằng người này không thể tin cậy được” - mặc dù thuật toán trước đây là một xấp xỉ gần đúng đã học được của cái sau. Mô hình của bạn sẽ rửa sạch và vận hành trên quy mô lớn những khía cạnh tồi tệ nhất trong cách đánh giá của con người, gây ra những tác động tiêu cực đến cuộc sống của con người thực.

Công nghệ không bao giờ trung tính. Nếu công việc của bạn có bất kỳ tác động nào đến thế giới thì tác động này mang tính đạo đức: những lựa chọn kỹ thuật cũng là những lựa chọn có tính đạo đức. Hãy luôn thận trọng về những giá trị mà bạn muốn công việc của mình hỗ trợ.

### Thu thập tập dữ liệu

Khi bạn hiểu bản chất của nhiệm vụ cũng như biết đầu vào và mục tiêu của mình sẽ là gì, đã đến lúc thu thập dữ liệu - phần khó khăn, tốn thời gian và tốn kém nhất trong hầu hết các dự án học máy:

* Dự án công cụ tìm kiếm ảnh yêu cầu bạn trước tiên phải chọn bộ
nhãn bạn muốn phân loại - bạn chọn 10.000 hình ảnh chung
Thể loại. Sau đó, bạn cần phải gắn thẻ thủ công hàng trăm ngàn quá khứ của mình
hình ảnh do người dùng tải lên có nhãn từ bộ này.
* Đối với dự án phát hiện thư rác của ứng dụng trò chuyện, vì cuộc trò chuyện của người dùng diễn ra từ đầu đến cuối
được mã hóa, bạn không thể sử dụng nội dung của chúng để đào tạo mô hình.
Bạn cần có quyền truy cập vào một tập dữ liệu riêng gồm hàng chục nghìn
các bài đăng trên mạng xã hội chưa được lọc và gắn thẻ chúng theo cách thủ công là spam, xúc phạm,
hoặc chấp nhận được.
* Đối với công cụ đề xuất âm nhạc, bạn chỉ có thể sử dụng lượt “thích” của
người dùng. Không có dữ liệu mới cần phải được thu thập. Tương tự, đối với tỷ lệ nhấp
dự án dự đoán, bạn có thành tích rộng rãi về tỷ lệ nhấp cho các quảng cáo trong quá khứ của mình,
quay trở lại nhiều năm.
* Đối với mô hình gắn cờ cookie, bạn sẽ cần cài đặt camera phía trên
băng chuyền để thu thập hàng chục ngàn hình ảnh,
và sau đó ai đó sẽ cần gắn nhãn theo cách thủ công
những hình ảnh này Những người biết cách làm điều này
hiện đang làm việc tại nhà máy sản xuất bánh quy - nhưng công việc đó có vẻ không quá khó khăn,
bạn sẽ có thể đào tạo mọi người làm điều đó.
* Dự án hình ảnh vệ tinh sẽ yêu cầu một nhóm các nhà khảo cổ học
thu thập cơ sở dữ liệu về các địa điểm quan tâm hiện có và đối với mỗi địa điểm,
bạn sẽ cần tìm những hình ảnh vệ tinh hiện có được chụp ở những nơi khác nhau
điều kiện thời tiết. Để có được một mô hình tốt, bạn sẽ cần hàng ngàn
của các trang web khác nhau.

Bạn đã học trong chương 5 rằng khả năng khái quát hóa của mô hình gần như hoàn toàn đến từ các thuộc tính của dữ liệu mà nó được đào tạo - số lượng điểm dữ liệu bạn có, độ tin cậy của nhãn, chất lượng các đặc điểm của bạn. Một bộ dữ liệu tốt là tài sản đáng được quan tâm và đầu tư. Nếu bạn có thêm 50 giờ để dành cho một dự án, rất có thể cách hiệu quả nhất để phân bổ chúng là thu thập thêm dữ liệu thay vì tìm kiếm các cải tiến mô hình hóa gia tăng.

Quan điểm cho rằng dữ liệu quan trọng hơn thuật toán đã được các nhà nghiên cứu của Google đưa ra một cách nổi tiếng nhất trong một bài báo năm 2009 có tựa đề “Tính hiệu quả phi lý của dữ liệu” (tiêu đề là đoạn trích từ cuốn sách nổi tiếng năm 1960 * Hiệu quả phi lý của toán học trong khoa học tự nhiên* của Eugene Wigner). Điều này xảy ra trước khi học sâu trở nên phổ biến, nhưng đáng chú ý là sự phát triển của học sâu chỉ làm tăng tầm quan trọng của dữ liệu.

Nếu bạn đang thực hiện học có giám sát thì sau khi đã thu thập thông tin đầu vào (chẳng hạn như hình ảnh), bạn sẽ cần *chú thích* cho chúng (chẳng hạn như thẻ cho những hình ảnh đó): mục tiêu mà bạn sẽ đào tạo mô hình của mình để dự đoán.

Đôi khi, chú thích có thể được truy xuất tự động — ví dụ: trong trường hợp nhiệm vụ đề xuất âm nhạc hoặc nhiệm vụ dự đoán tỷ lệ nhấp của chúng tôi. Nhưng thông thường, bạn phải chú thích dữ liệu của mình bằng tay. Đây là một quá trình tốn nhiều công sức.

#### Đầu tư vào cơ sở hạ tầng chú thích dữ liệu

Quá trình chú thích dữ liệu của bạn sẽ xác định chất lượng của các mục tiêu, từ đó xác định chất lượng mô hình của bạn. Hãy xem xét cẩn thận các lựa chọn bạn có sẵn:

* Bạn có nên tự mình chú thích dữ liệu?
* Bạn có nên sử dụng nền tảng cung cấp dịch vụ cộng đồng như Mechanical Turk để thu thập nhãn không?
* Bạn có nên sử dụng dịch vụ của một công ty ghi nhãn dữ liệu chuyên dụng không?

Gia công phần mềm có thể giúp bạn tiết kiệm thời gian và tiền bạc nhưng lại lấy đi quyền kiểm soát. Sử dụng thứ gì đó như Mechanical Turk có thể sẽ không tốn kém và có khả năng mở rộng quy mô tốt, nhưng chú thích của bạn có thể khá ồn ào.

Để chọn tùy chọn tốt nhất, hãy xem xét các ràng buộc bạn đang làm việc:

* Người gắn nhãn dữ liệu có cần phải là chuyên gia về chủ đề đó hay không hoặc bất kỳ ai cũng có thể
chú thích dữ liệu? Các nhãn để phân loại hình ảnh giữa mèo và chó
bất cứ ai cũng có thể chọn vấn đề, nhưng vấn đề phân loại giống chó
công việc đòi hỏi kiến ​​thức chuyên môn. Trong khi đó, chú thích ảnh chụp CT của
gãy xương khá nhiều đòi hỏi phải có bằng y khoa.
* Nếu chú thích dữ liệu đòi hỏi kiến ​​thức chuyên môn,
bạn có thể đào tạo mọi người làm điều đó không? Nếu không, làm thế nào bạn có thể truy cập vào
chuyên gia liên quan?
* Bản thân bạn có hiểu cách các chuyên gia đưa ra
các chú thích? Nếu không, bạn sẽ phải xử lý
tập dữ liệu của bạn dưới dạng hộp đen và bạn sẽ không thể thực hiện tính năng thủ công
kỹ thuật - điều này không quan trọng nhưng nó có thể bị hạn chế.

Nếu bạn quyết định gắn nhãn dữ liệu của mình nội bộ, hãy tự hỏi bạn sẽ sử dụng phần mềm nào để ghi lại chú thích. Bạn có thể cần phải tự mình phát triển phần mềm đó. Phần mềm chú thích dữ liệu hiệu quả sẽ giúp bạn tiết kiệm rất nhiều thời gian, vì vậy đây là thứ đáng để đầu tư sớm vào một dự án.

#### Cẩn thận với dữ liệu không mang tính đại diện

Các mô hình học máy chỉ có thể hiểu được những đầu vào tương tự với những gì chúng đã thấy trước đây. Do đó, điều quan trọng là dữ liệu được sử dụng cho quá trình đào tạo phải *đại diện* cho dữ liệu sản xuất. Mối quan tâm này phải là nền tảng cho tất cả công việc thu thập dữ liệu của bạn.

Giả sử bạn đang phát triển một ứng dụng trong đó người dùng có thể chụp ảnh một món ăn để tìm ra tên của nó. Bạn đào tạo một người mẫu bằng cách sử dụng hình ảnh từ mạng xã hội chia sẻ hình ảnh được những người sành ăn ưa chuộng. Đến thời điểm triển khai và phản hồi từ những người dùng tức giận bắt đầu xuất hiện: ứng dụng của bạn nhận được câu trả lời sai 8/10 lần. Chuyện gì đang xảy ra vậy? Độ chính xác của bạn trong tập kiểm tra là hơn 90%! Xem nhanh dữ liệu do người dùng tải lên cho thấy rằng ảnh tải lên trên thiết bị di động về các món ăn ngẫu nhiên từ các nhà hàng ngẫu nhiên được chụp bằng điện thoại thông minh ngẫu nhiên trông không giống những bức ảnh ngon miệng, đủ ánh sáng, chất lượng chuyên nghiệp mà bạn đã đào tạo mô hình: *dữ liệu đào tạo của bạn không đại diện cho dữ liệu sản xuất*. Đó là một tội lỗi nghiêm trọng - chào mừng bạn đến với địa ngục của máy học.

Nếu có thể, hãy thu thập dữ liệu trực tiếp từ môi trường nơi mô hình của bạn sẽ được sử dụng. Nên sử dụng mô hình phân loại cảm tính khi đánh giá phim cho các bài đánh giá IMDB mới, không phải trên các bài đánh giá về nhà hàng trên Yelp cũng như trên các cập nhật trạng thái trên Twitter. Nếu bạn muốn đánh giá cảm xúc của một tweet, hãy bắt đầu bằng cách thu thập và chú thích các tweet thực tế - từ một nhóm người dùng tương tự như những người bạn mong đợi trong quá trình sản xuất. Nếu không thể đào tạo trên dữ liệu sản xuất thì hãy đảm bảo bạn hiểu đầy đủ sự khác biệt giữa dữ liệu đào tạo và dữ liệu sản xuất và bạn đang tích cực khắc phục những khác biệt này.

Một hiện tượng liên quan mà bạn cần lưu ý là *trôi dạt khái niệm*. Bạn sẽ gặp phải tình trạng trôi dạt khái niệm trong hầu hết các vấn đề trong thế giới thực, đặc biệt là những vấn đề liên quan đến dữ liệu do người dùng tạo. Sự trôi dạt khái niệm xảy ra khi các thuộc tính của dữ liệu sản xuất thay đổi theo thời gian, khiến độ chính xác của mô hình giảm dần. Công cụ đề xuất âm nhạc được đào tạo vào năm 2013 có thể không hiệu quả lắm ở thời điểm hiện tại. Tương tự, tập dữ liệu IMDB mà bạn làm việc cùng đã được thu thập vào năm 2011 và một mô hình được đào tạo dựa trên đó có thể sẽ không hoạt động tốt trên các bài đánh giá từ năm 2020 so với các bài đánh giá từ năm 2012, vì từ vựng, cách diễn đạt và thể loại phim phát triển theo thời gian. Sự lệch lạc khái niệm đặc biệt nghiêm trọng trong các bối cảnh đối nghịch như phát hiện gian lận thẻ tín dụng, nơi các mô hình gian lận thực tế thay đổi hàng ngày. Xử lý tình trạng trôi dạt khái niệm nhanh đòi hỏi phải thu thập dữ liệu, chú thích và đào tạo lại mô hình liên tục.

Hãy nhớ rằng học máy chỉ có thể được sử dụng để ghi nhớ các mẫu có trong dữ liệu đào tạo của bạn. Bạn chỉ có thể nhận ra những gì bạn đã thấy trước đây. Việc sử dụng máy học được đào tạo dựa trên dữ liệu trong quá khứ để dự đoán tương lai đang đưa ra giả định rằng tương lai sẽ hoạt động giống như quá khứ. Điều đó thường không xảy ra.

Vấn đề sai lệch lấy mẫu

Một trường hợp đặc biệt nguy hiểm và phổ biến của dữ liệu không mang tính đại diện là *sai lệch lấy mẫu*. Sai lệch lấy mẫu xảy ra khi quá trình thu thập dữ liệu của bạn tương tác với những gì bạn đang cố gắng dự đoán, dẫn đến các phép đo sai lệch. Một ví dụ lịch sử nổi tiếng xảy ra trong cuộc bầu cử tổng thống Mỹ năm 1948. Vào đêm bầu cử, tờ Chicago Tribune đã in dòng tiêu đề “DEWEY ĐÁNH BẠI TRUMAN.” Sáng hôm sau, Truman nổi lên là người chiến thắng. Biên tập viên của tờ Tribune đã tin tưởng vào kết quả của một cuộc khảo sát qua điện thoại - nhưng những người sử dụng điện thoại vào năm 1948 không phải là một mẫu đại diện ngẫu nhiên của cộng đồng bỏ phiếu. Họ có nhiều khả năng giàu có hơn, bảo thủ hơn và bỏ phiếu cho Dewey, ứng cử viên Đảng Cộng hòa. Ngày nay, mọi cuộc khảo sát qua điện thoại đều tính đến xu hướng lấy mẫu. Điều đó không có nghĩa là xu hướng lấy mẫu đã là quá khứ trong cuộc thăm dò chính trị - còn lâu mới xảy ra. Nhưng không giống như năm 1948, những người thăm dò ý kiến ​​đã nhận thức được điều đó và thực hiện các bước để khắc phục nó.

![](../images/ch06/dewey_truman.015b2e12.jpg)

[Figure 6.1](#figure-6-1): “DEWEY DEFEATS TRUMAN”: a famous example of sampling bias

### Hiểu dữ liệu của bạn

Việc coi tập dữ liệu như một hộp đen là một thực hành tồi. Trước khi bắt đầu đào tạo mô hình, bạn nên khám phá và trực quan hóa dữ liệu của mình để hiểu rõ hơn về yếu tố giúp dữ liệu có tính dự đoán — điều này sẽ cung cấp thông tin cho kỹ thuật tính năng — và sàng lọc các vấn đề tiềm ẩn:

* Nếu dữ liệu của bạn bao gồm hình ảnh hoặc văn bản ngôn ngữ tự nhiên, hãy xem một số
mẫu (và nhãn của chúng) một cách trực tiếp.
* Nếu dữ liệu của bạn chứa các đặc điểm số thì bạn nên vẽ sơ đồ
biểu đồ các giá trị đặc trưng để cảm nhận về phạm vi giá trị được lấy
và tần số có giá trị khác nhau.
* Nếu dữ liệu của bạn bao gồm thông tin vị trí, hãy vẽ nó trên bản đồ. Làm bất kỳ
mô hình rõ ràng xuất hiện?
* Có phải một số mẫu bị thiếu giá trị cho một số tính năng không? Nếu vậy, bạn sẽ cần
để giải quyết vấn đề này khi bạn chuẩn bị dữ liệu (chúng tôi đề cập đến cách thực hiện việc này
ở phần tiếp theo).
* Nếu nhiệm vụ của bạn là vấn đề phân loại, hãy in số lượng phiên bản
của từng lớp trong dữ liệu của bạn. Các lớp có được đại diện gần như bằng nhau không?
Nếu không, bạn sẽ cần phải tính đến sự mất cân bằng này.
* Kiểm tra *rò rỉ mục tiêu* — sự hiện diện của các tính năng trong dữ liệu của bạn
cung cấp thông tin về các mục tiêu có thể không có sẵn trong
sản xuất. Nếu bạn đang đào tạo một mô hình về hồ sơ y tế để dự đoán
liệu ai đó sẽ được điều trị ung thư trong tương lai hay không và hồ sơ
bao gồm tính năng “Người này đã được chẩn đoán mắc bệnh ung thư”, sau đó
mục tiêu của bạn đang bị rò rỉ một cách giả tạo vào dữ liệu của bạn.
Hãy luôn tự hỏi bản thân, liệu mọi tính năng trong dữ liệu của bạn có phải là thứ gì đó không?
có sẵn ở dạng tương tự trong sản xuất?

### Lựa chọn thước đo thành công

Để kiểm soát một cái gì đó, bạn cần có khả năng quan sát nó. Để đạt được thành công trong một dự án, trước tiên bạn phải xác định ý nghĩa của thành công là gì. Sự chính xác? Độ chính xác và thu hồi? Tỷ lệ giữ chân khách hàng? Thước đo thành công của bạn sẽ hướng dẫn tất cả các lựa chọn kỹ thuật mà bạn sẽ thực hiện trong suốt dự án. Nó phải phù hợp trực tiếp với các mục tiêu cấp cao hơn của bạn, chẳng hạn như sự thành công trong kinh doanh của khách hàng.

Đối với các vấn đề phân loại cân bằng, trong đó mọi lớp đều có khả năng xảy ra như nhau, độ chính xác và *diện tích dưới đường cong* (AUC) của *đặc tính vận hành máy thu* (ROC) là các số liệu phổ biến. Đối với các vấn đề mất cân bằng lớp, vấn đề xếp hạng hoặc phân loại nhiều nhãn, bạn có thể sử dụng độ chính xác và thu hồi hoặc số liệu đếm dương tính giả, dương tính thực, âm tính giả và âm tính thực. Và không có gì lạ khi phải xác định số liệu tùy chỉnh của riêng bạn để đo lường thành công. Để hiểu được sự đa dạng của các chỉ số thành công trong học máy và mối liên hệ của chúng với các lĩnh vực vấn đề khác nhau, bạn nên duyệt qua các cuộc thi khoa học dữ liệu trên Kaggle (<https://kaggle.com>); nó cho thấy một loạt các vấn đề và số liệu đánh giá.

## Phát triển một mô hình

Khi bạn biết cách đo lường sự tiến bộ của mình, bạn có thể bắt đầu phát triển mô hình. Hầu hết các hướng dẫn và dự án nghiên cứu đều cho rằng đây là bước duy nhất - bỏ qua việc xác định vấn đề và thu thập dữ liệu, được cho là đã được thực hiện và bỏ qua việc triển khai và bảo trì mô hình, được cho là do người khác xử lý. Trên thực tế, việc phát triển mô hình chỉ là *một bước* trong quy trình học máy và nếu bạn hỏi tôi thì đó không phải là bước khó nhất. Những điều khó nhất trong học máy là sắp xếp các vấn đề và thu thập, chú thích và làm sạch dữ liệu. Vì vậy hãy vui lên, những gì xảy ra tiếp theo sẽ dễ dàng so sánh được!

### Chuẩn bị dữ liệu

Như bạn đã biết trước đây, các mô hình học sâu thường không sử dụng dữ liệu thô. Quá trình tiền xử lý dữ liệu nhằm mục đích làm cho dữ liệu thô có sẵn trở nên dễ sử dụng hơn đối với mạng lưới thần kinh. Điều này bao gồm vector hóa, chuẩn hóa hoặc xử lý các giá trị bị thiếu. Nhiều kỹ thuật tiền xử lý có tính chất cụ thể theo miền (ví dụ: cụ thể đối với dữ liệu văn bản hoặc dữ liệu hình ảnh); chúng ta sẽ đề cập đến những điều đó trong các chương sau khi chúng ta gặp chúng trong các ví dụ thực tế. Hiện tại, chúng ta sẽ xem xét những kiến ​​thức cơ bản chung cho tất cả các miền dữ liệu.

#### Vector hóa

Tất cả các đầu vào và mục tiêu trong mạng thần kinh thường phải là các tensor của dữ liệu dấu phẩy động (hoặc, trong các trường hợp cụ thể, tensor của số nguyên hoặc chuỗi). Bất kỳ dữ liệu nào bạn cần xử lý — âm thanh, hình ảnh, văn bản — trước tiên bạn phải chuyển sang tensor, một bước được gọi là *vector hóa dữ liệu*. Ví dụ: trong hai ví dụ phân loại văn bản trước ở chương 4, chúng tôi bắt đầu từ văn bản được biểu thị dưới dạng danh sách các số nguyên (viết tắt của chuỗi các từ) và chúng tôi đã sử dụng mã hóa đa điểm để biến chúng thành một tenxơ của dữ liệu `float32`. Trong các ví dụ về phân loại chữ số và dự đoán giá nhà, dữ liệu đã ở dạng véc tơ nên bạn có thể bỏ qua bước này.

#### Chuẩn hóa giá trị

Trong ví dụ về phân loại chữ số MNIST ở chương 2, bạn đã bắt đầu từ dữ liệu hình ảnh được mã hóa dưới dạng số nguyên trong phạm vi 0–255, mã hóa các giá trị thang độ xám. Trước khi đưa dữ liệu này vào mạng của mình, bạn phải chuyển dữ liệu đó thành `float32` và chia cho 255 để có được các giá trị dấu phẩy động trong phạm vi 0–1. Tương tự, khi dự đoán giá nhà, bạn bắt đầu từ các tính năng có phạm vi khác nhau - một số tính năng có giá trị dấu phẩy động nhỏ, một số tính năng khác có giá trị số nguyên khá lớn. Trước khi đưa dữ liệu này vào mạng của mình, bạn phải chuẩn hóa từng tính năng một cách độc lập để nó có độ lệch chuẩn là 1 và giá trị trung bình là 0.

Nói chung, sẽ không an toàn khi đưa vào dữ liệu mạng thần kinh có giá trị tương đối lớn (ví dụ: số nguyên có nhiều chữ số, lớn hơn nhiều so với giá trị ban đầu được lấy theo trọng số của mạng) hoặc dữ liệu không đồng nhất (ví dụ: dữ liệu trong đó một tính năng nằm trong phạm vi 0–1 và một tính năng khác nằm trong phạm vi 100–200). Làm như vậy có thể kích hoạt các cập nhật độ dốc lớn khiến mạng không thể hội tụ. Để giúp mạng của bạn tìm hiểu dễ dàng hơn, dữ liệu của bạn phải có các đặc điểm sau:

* *Lấy các giá trị nhỏ*  — Thông thường, hầu hết các giá trị phải nằm trong phạm vi 0–1.
* *Đồng nhất* — Nghĩa là, tất cả các đặc điểm phải nhận giá trị trong khoảng
cùng một phạm vi.

Ngoài ra, phương pháp chuẩn hóa chặt chẽ hơn sau đây rất phổ biến và có thể hữu ích, mặc dù không phải lúc nào cũng cần thiết (ví dụ: bạn đã không thực hiện điều này trong ví dụ về phân loại chữ số):

* Chuẩn hóa từng tính năng một cách độc lập để có giá trị trung bình bằng 0.
* Chuẩn hóa từng tính năng một cách độc lập để có độ lệch chuẩn là 1.

Điều này rất dễ thực hiện với mảng NumPy:

```python
# Assuming x is a 2D data matrix of shape (samples, features)
x -= x.mean(axis=0)
x /= x.std(axis=0)
```

#### Xử lý các giá trị bị thiếu

Đôi khi bạn có thể thiếu các giá trị trong dữ liệu của mình. Ví dụ, trong ví dụ về giá nhà, đặc điểm thứ hai là tuổi trung bình của những ngôi nhà trong quận. Điều gì sẽ xảy ra nếu tính năng này không có sẵn cho tất cả các mẫu? Khi đó, bạn sẽ thiếu các giá trị trong dữ liệu huấn luyện hoặc kiểm tra.

Bạn có thể loại bỏ hoàn toàn tính năng này nhưng không nhất thiết phải:

* Nếu đối tượng địa lý là phân loại thì việc tạo một danh mục mới có nghĩa là an toàn
“giá trị bị thiếu.” Mô hình sẽ tự động tìm hiểu ý nghĩa của điều này
quan đến các mục tiêu.
* Nếu đối tượng địa lý là số, hãy tránh nhập giá trị tùy ý như 0
bởi vì nó có thể tạo ra sự gián đoạn trong không gian tiềm ẩn được hình thành bởi
các tính năng của bạn, khiến cho một mô hình được đào tạo về nó gặp khó khăn hơn
khái quát hóa. Thay vào đó, hãy xem xét việc thay thế giá trị còn thiếu bằng
giá trị trung bình hoặc trung bình cho tính năng trong tập dữ liệu. Bạn cũng có thể luyện tập
một mô hình để dự đoán giá trị tính năng dựa trên giá trị của các tính năng khác.

Lưu ý rằng nếu bạn đang mong đợi các tính năng phân loại bị thiếu trong dữ liệu thử nghiệm, nhưng mạng đã được huấn luyện về dữ liệu mà không có bất kỳ giá trị nào bị thiếu, thì mạng sẽ không học cách bỏ qua các giá trị bị thiếu! Trong tình huống này, bạn nên tạo các mẫu đào tạo một cách giả tạo với các mục bị thiếu: sao chép một số mẫu đào tạo nhiều lần và loại bỏ một số tính năng phân loại mà bạn cho rằng có thể bị thiếu trong dữ liệu thử nghiệm.

### Lựa chọn một giao thức đánh giá

Như bạn đã học ở chương trước, mục đích của mô hình là đạt được sự khái quát hóa và mọi quyết định về mô hình hóa mà bạn đưa ra trong suốt quá trình phát triển mô hình sẽ được hướng dẫn bởi *số liệu xác thực* tìm cách đo lường hiệu suất khái quát hóa. Mục tiêu của giao thức xác thực là ước tính chính xác chỉ số thành công mà bạn lựa chọn (chẳng hạn như độ chính xác) sẽ có trên dữ liệu sản xuất thực tế. Độ tin cậy của quá trình đó là rất quan trọng để xây dựng một mô hình hữu ích.

Trong chương 5, chúng ta đã xem xét ba quy trình đánh giá phổ biến:

* *Duy trì bộ xác thực giữ lại*  — Cách thực hiện khi bạn có
nhiều dữ liệu

* *Thực hiện xác thực chéo K-Fold*  — Lựa chọn đúng khi bạn có quá ít
mẫu để xác nhận giữ lại là đáng tin cậy

* *Thực hiện xác thực K-Fold lặp đi lặp lại*  — Để thực hiện mô hình có độ chính xác cao
đánh giá khi có ít dữ liệu

Chỉ cần chọn một trong số này. Trong hầu hết các trường hợp, cái đầu tiên sẽ hoạt động đủ tốt. Như bạn đã học trước đây, hãy luôn chú ý đến *tính đại diện* của (các) bộ xác thực và cẩn thận để không có các mẫu dư thừa giữa tập huấn luyện và (các) bộ xác thực của bạn.

### Đánh bại đường cơ sở

Khi bạn bắt đầu làm việc trên chính mô hình đó, mục tiêu ban đầu của bạn là đạt được *sức mạnh thống kê*, như bạn đã thấy trong chương 5 — tức là phát triển một mô hình nhỏ có khả năng vượt qua đường cơ sở đơn giản.

Ở giai đoạn này, đây là ba điều quan trọng nhất bạn nên tập trung vào:

* *Kỹ thuật tính năng* — Lọc ra các tính năng không có thông tin (lựa chọn tính năng)
và sử dụng kiến ​​thức của bạn về vấn đề này để phát triển các tính năng mới
điều đó có thể sẽ hữu ích.
* *Chọn kiến ​​trúc ưu tiên chính xác* — Loại kiến ​​trúc mô hình nào
bạn sẽ sử dụng chứ? Một mạng được kết nối dày đặc, một ConvNet, một mạng thần kinh tái phát
mạng, một máy biến áp? Học sâu có phải là một cách tiếp cận tốt cho nhiệm vụ,
hoặc bạn nên sử dụng cái gì khác?
* *Chọn cấu hình đào tạo đủ tốt* — Nên sử dụng chức năng mất gì
bạn sử dụng? Kích thước lô và tốc độ học tập là bao nhiêu?

Chọn hàm mất phù hợp

Thường không thể tối ưu hóa trực tiếp số liệu đo lường mức độ thành công của một vấn đề. Đôi khi không có cách nào dễ dàng để biến một số liệu thành một hàm mất mát; Xét cho cùng, các hàm mất mát chỉ cần có thể tính toán được với một lượng nhỏ dữ liệu (lý tưởng nhất là hàm mất mát phải có thể tính toán được chỉ với một điểm dữ liệu) và phải có tính khả vi (nếu không, bạn không thể sử dụng lan truyền ngược để huấn luyện mạng của mình). Ví dụ: không thể tối ưu hóa trực tiếp chỉ số phân loại ROC AUC được sử dụng rộng rãi. Do đó, trong các nhiệm vụ phân loại, việc tối ưu hóa chỉ số proxy của ROC AUC, chẳng hạn như entropy chéo, là điều thường thấy. Nói chung, bạn có thể hy vọng rằng entropy chéo càng thấp thì ROC AUC sẽ càng cao.

Bảng 6.1 có thể giúp bạn chọn kích hoạt lớp cuối cùng, hàm mất và số liệu cho một số loại vấn đề phổ biến.

| Nhiệm vụ | Kích hoạt lớp cuối cùng | Hàm mất | Số liệu | | --- | --- | --- | --- | | Phân loại nhị phân | Sigmoid | Entropy chéo nhị phân | Độ chính xác nhị phân, ROC AUC | | Phân loại nhiều lớp, một nhãn | Softmax | Crossentropy phân loại | Độ chính xác về mặt phân loại, độ chính xác về mặt phân loại top-k, ROC AUC | | Phân loại đa lớp, nhiều nhãn | Sigmoid | Entropy chéo nhị phân | Độ chính xác nhị phân, ROC AUC | | Hồi quy | Không có | Sai số bình phương trung bình | Có nghĩa là sai số tuyệt đối |

[Bảng 6.1](#table-6-1): Mất mát nào, kích hoạt lớp cuối cùng và số liệu nào sẽ sử dụng cho các nhiệm vụ khác nhau

Đối với hầu hết các vấn đề, bạn có thể bắt đầu từ các mẫu hiện có. Bạn không phải là người đầu tiên cố gắng xây dựng trình phát hiện thư rác, công cụ đề xuất âm nhạc hoặc trình phân loại hình ảnh. Đảm bảo nghiên cứu tình trạng kỹ thuật trước đây để xác định các kỹ thuật kỹ thuật tính năng và kiến ​​trúc mô hình có nhiều khả năng thực hiện tốt nhiệm vụ của bạn nhất.

Lưu ý rằng không phải lúc nào cũng có thể đạt được sức mạnh thống kê. Nếu bạn không thể đạt được đường cơ sở đơn giản sau khi thử nhiều kiến ​​trúc hợp lý, thì có thể câu trả lời cho câu hỏi bạn đang hỏi không có trong dữ liệu đầu vào. Hãy nhớ rằng bạn đang đưa ra hai giả thuyết:

* Bạn đưa ra giả thuyết rằng kết quả đầu ra của bạn có thể được dự đoán dựa trên đầu vào của bạn.
* Bạn đưa ra giả thuyết rằng dữ liệu có sẵn có đủ thông tin để tìm hiểu
mối quan hệ giữa đầu vào và đầu ra.

Rất có thể những giả thuyết này là sai, trong trường hợp đó bạn phải quay lại bảng vẽ.

### Mở rộng quy mô: phát triển một mô hình phù hợp hơn

Khi bạn đã có được một mô hình có sức mạnh thống kê, câu hỏi đặt ra là liệu mô hình của bạn có đủ mạnh không? Liệu nó có đủ các lớp và tham số để mô hình hóa chính xác vấn đề hiện tại không? Ví dụ: mô hình hồi quy logistic có sức mạnh thống kê trên MNIST nhưng sẽ không đủ để giải quyết tốt vấn đề. Hãy nhớ rằng sự căng thẳng chung trong học máy là giữa tối ưu hóa và khái quát hóa; mô hình lý tưởng là mô hình nằm ngay ranh giới giữa trang bị thiếu và thừa, giữa thiếu công suất và thừa công suất. Để tìm ra ranh giới này nằm ở đâu, trước tiên bạn phải vượt qua nó.

Để biết bạn cần một mô hình lớn đến mức nào, bạn phải phát triển một mô hình phù hợp. Điều này khá dễ dàng, như bạn đã học ở chương 5:

* Thêm lớp.
* Làm cho các lớp lớn hơn.
* Đào tạo cho nhiều kỷ nguyên hơn.

Luôn theo dõi tổn thất đào tạo và mất xác thực, cũng như các giá trị đào tạo và xác thực cho bất kỳ số liệu nào bạn quan tâm. Khi bạn thấy hiệu suất của mô hình trên dữ liệu xác thực bắt đầu giảm, bạn đã đạt được tình trạng trang bị quá mức.

### Thường xuyên hóa và điều chỉnh mô hình của bạn

Khi bạn đã đạt được sức mạnh thống kê và có thể tập luyện quá sức, bạn biết mình đang đi đúng hướng. Tại thời điểm này, mục tiêu của bạn trở thành tối đa hóa hiệu suất khái quát hóa.

Giai đoạn này sẽ mất nhiều thời gian nhất: bạn sẽ liên tục sửa đổi mô hình của mình, huấn luyện nó, đánh giá dữ liệu xác thực của bạn (không phải dữ liệu thử nghiệm tại thời điểm này), sửa đổi lại và lặp lại cho đến khi mô hình tốt nhất có thể. Dưới đây là một số điều bạn nên thử:

* Hãy thử các kiến ​​trúc khác nhau; thêm hoặc loại bỏ các lớp.
* Thêm tình trạng bỏ học.
* Nếu mô hình của bạn nhỏ, hãy thêm chính quy hóa L1 hoặc L2.
* Hãy thử các siêu tham số khác nhau (chẳng hạn như số lượng đơn vị trên mỗi lớp hoặc
tốc độ học tập của trình tối ưu hóa) để tìm cấu hình tối ưu.
* Tùy chọn, lặp lại việc quản lý dữ liệu hoặc kỹ thuật tính năng:
thu thập và chú thích nhiều dữ liệu hơn, phát triển các tính năng tốt hơn,
hoặc loại bỏ các tính năng dường như không mang lại nhiều thông tin.

Có thể tự động hóa phần lớn công việc này bằng cách sử dụng *phần mềm điều chỉnh siêu tham số tự động*, chẳng hạn như KerasTuner. Chúng tôi sẽ đề cập đến vấn đề này trong chương 18.

Hãy lưu ý những điều sau: mỗi khi bạn sử dụng phản hồi từ quy trình xác thực để điều chỉnh mô hình của mình, bạn sẽ rò rỉ thông tin về quy trình xác thực vào mô hình. Chỉ lặp đi lặp lại một vài lần, điều này chẳng có hại gì; tuy nhiên, được thực hiện một cách có hệ thống qua nhiều lần lặp lại, cuối cùng nó sẽ khiến mô hình của bạn quá phù hợp với quy trình xác thực (mặc dù không có mô hình nào được đào tạo trực tiếp về bất kỳ dữ liệu xác thực nào). Điều này làm cho quá trình đánh giá kém tin cậy hơn.

Sau khi đã phát triển cấu hình mô hình thỏa đáng, bạn có thể huấn luyện mô hình sản xuất cuối cùng của mình trên tất cả dữ liệu có sẵn (huấn luyện và xác thực) và đánh giá nó lần cuối trên bộ thử nghiệm. Nếu hiệu suất trên tập kiểm tra kém hơn đáng kể so với hiệu suất đo được trên dữ liệu xác thực, điều này có thể có nghĩa là quy trình xác thực của bạn rốt cuộc không đáng tin cậy hoặc bạn đã bắt đầu trang bị quá mức cho dữ liệu xác thực trong khi điều chỉnh các tham số của mô hình. Trong trường hợp này, bạn có thể muốn chuyển sang một giao thức đánh giá đáng tin cậy hơn (chẳng hạn như xác thực K-Fold lặp lại).

## Triển khai mô hình của bạn

Sau khi mô hình của bạn hoàn thành thành công bước đánh giá cuối cùng trên tập thử nghiệm, mô hình đã sẵn sàng để triển khai và bắt đầu hoạt động hiệu quả.

### Giải thích công việc của bạn cho các bên liên quan và đặt kỳ vọng

Thành công và sự tin tưởng của khách hàng là việc luôn đáp ứng hoặc vượt quá sự mong đợi của mọi người; hệ thống thực tế bạn cung cấp chỉ là một nửa của bức tranh đó. Nửa còn lại là đặt ra những kỳ vọng phù hợp trước khi ra mắt.

Kỳ vọng của những người không chuyên đối với hệ thống AI thường không thực tế. Ví dụ, họ có thể mong đợi rằng hệ thống “hiểu” nhiệm vụ của nó và có khả năng thực hiện ý thức chung giống con người trong bối cảnh nhiệm vụ. Để giải quyết vấn đề này, bạn nên xem xét hiển thị một số ví dụ về *chế độ lỗi* trong mô hình của mình (ví dụ: hiển thị các mẫu được phân loại không chính xác trông như thế nào, đặc biệt là những mẫu mà việc phân loại sai có vẻ đáng ngạc nhiên).

Họ cũng có thể mong đợi hiệu suất ở cấp độ con người, đặc biệt là đối với các quy trình trước đây do con người xử lý. Hầu hết các mô hình học máy, vì chúng được đào tạo (không hoàn hảo) để gần đúng với các nhãn do con người tạo ra, nên gần như không đạt được điều đó. Bạn nên truyền đạt rõ ràng những kỳ vọng về hiệu suất của mô hình. Tránh sử dụng các tuyên bố trừu tượng như “Mô hình có độ chính xác 98%” (mà hầu hết mọi người đều làm tròn trong đầu lên tới 100%) và thích nói chuyện, chẳng hạn như về tỷ lệ âm tính giả và tỷ lệ dương tính giả. Bạn có thể nói, "Với những cài đặt này, mô hình phát hiện gian lận sẽ có tỷ lệ âm tính giả là 5% và tỷ lệ dương tính giả là 2,5%. Mỗi ngày, trung bình 200 giao dịch hợp lệ sẽ bị gắn cờ là gian lận và được gửi đi xem xét thủ công, đồng thời trung bình sẽ có 14 giao dịch gian lận bị bỏ sót. Trung bình có 266 giao dịch gian lận sẽ bị phát hiện chính xác." Liên hệ rõ ràng các số liệu hiệu suất của mô hình với mục tiêu kinh doanh.

Bạn cũng nên đảm bảo thảo luận với các bên liên quan về việc lựa chọn các tham số khởi chạy chính - ví dụ: ngưỡng xác suất mà giao dịch sẽ được gắn cờ (các ngưỡng khác nhau sẽ tạo ra tỷ lệ âm tính giả và dương tính giả khác nhau). Những quyết định như vậy liên quan đến sự đánh đổi mà chỉ có thể được xử lý khi có sự hiểu biết sâu sắc về bối cảnh kinh doanh.

### Vận chuyển một mô hình suy luận

Một dự án machine learning không kết thúc khi bạn có một sổ ghi chép Colab có thể lưu mô hình đã được đào tạo. Bạn hiếm khi đưa vào sản xuất chính xác đối tượng mô hình Python mà bạn đã thao tác trong quá trình đào tạo.

Trước tiên, bạn có thể muốn xuất mô hình của mình sang một thứ khác ngoài Python:

* Môi trường sản xuất của bạn có thể hoàn toàn không hỗ trợ Python - ví dụ:
nếu đó là ứng dụng dành cho thiết bị di động hoặc hệ thống nhúng.
* Nếu phần còn lại của ứng dụng không có bằng Python (có thể bằng JavaScript,
C++, v.v.), việc sử dụng Python để phục vụ một mô hình có thể gây ra
chi phí đáng kể.

Thứ hai, vì mô hình sản xuất của bạn sẽ chỉ được sử dụng để đưa ra các dự đoán (giai đoạn được gọi là *suy luận*), chứ không phải để đào tạo nên bạn có chỗ để thực hiện nhiều tối ưu hóa khác nhau nhằm giúp mô hình nhanh hơn và giảm mức chiếm dụng bộ nhớ của mô hình.

Hãy xem nhanh các tùy chọn triển khai mô hình khác nhau mà bạn có sẵn.

#### Triển khai mô hình dưới dạng API REST

Có lẽ cách dễ nhất để biến mô hình thành sản phẩm là cung cấp mô hình đó trực tuyến thông qua API REST. Có một số thư viện hiện có để thực hiện điều này. Keras hỗ trợ hai trong số những phương pháp phổ biến nhất hiện nay — *TensorFlow Serve* và *ONNX* (viết tắt của Open Neural Network Exchange). Cả hai thư viện đều hoạt động bằng cách nâng tất cả trọng số mô hình và biểu đồ tính toán bên ngoài chương trình Python, do đó bạn có thể phân phát nó từ một số môi trường khác nhau (ví dụ: máy chủ C++). Nếu điều này nghe có vẻ giống với cơ chế biên dịch được thảo luận ở chương 3 thì bạn đã đúng. TensorFlow Serve về cơ bản là một thư viện để phục vụ các biểu đồ tính toán `tf.function` với một tập hợp trọng số đã lưu cụ thể.

Keras cho phép truy cập vào cả TensorFlow Serve và ONNX thông qua phương thức `export()` dễ sử dụng có sẵn trên tất cả các mô hình Keras. Đây là đoạn mã cho thấy cách hoạt động của tính năng này đối với Dịch vụ TensorFlow:

```python
# Exports the model as a TensorFlow SavedModel artifact
model.export("path/to/location", format="tf_saved_model")

# Loads the artifact in a different process, environment, or
# programming language
reloaded_artifact = tf.saved_model.load("path/to/location")
predictions = reloaded_artifact.serve(input_data)
```

Một luồng tương tự tồn tại cho ONNX:

```python
model.export("path/to/location", format="onnx")

ort_session = onnxruntime.InferenceSession("path/to/location")
predictions = ort_session.run(None, input_data)
```

Bạn nên sử dụng thiết lập triển khai này khi

* Ứng dụng sử dụng dự đoán của mô hình sẽ có độ tin cậy
truy cập internet (rõ ràng). Ví dụ,
nếu ứng dụng của bạn là ứng dụng dành cho thiết bị di động, cung cấp dự đoán từ API từ xa
có nghĩa là ứng dụng sẽ không thể sử dụng được ở chế độ trên máy bay hoặc trong
môi trường kết nối thấp.
* Ứng dụng không có yêu cầu nghiêm ngặt về độ trễ: yêu cầu,
suy luận và trả lời chuyến đi khứ hồi thường sẽ mất khoảng 500 mili giây.
* Dữ liệu đầu vào được gửi để suy luận không có độ nhạy cao: dữ liệu sẽ cần
sẵn có trên máy chủ ở dạng đã được giải mã, vì nó cần phải được
được mô hình nhìn thấy (nhưng lưu ý rằng bạn nên sử dụng mã hóa SSL cho
yêu cầu và câu trả lời HTTP).

Ví dụ: dự án công cụ tìm kiếm hình ảnh, hệ thống đề xuất âm nhạc, dự án phát hiện gian lận thẻ tín dụng và dự án hình ảnh vệ tinh đều phù hợp để phân phối thông qua API REST.

Một câu hỏi quan trọng khi triển khai mô hình dưới dạng API REST là liệu bạn muốn tự lưu trữ mã hay muốn sử dụng dịch vụ đám mây của bên thứ ba được quản lý hoàn toàn. Ví dụ: Cloud AI Platform, một sản phẩm của Google, cho phép bạn chỉ cần tải mô hình TensorFlow của mình lên Google Cloud Storage (GCS) và cung cấp cho bạn điểm cuối API để truy vấn mô hình đó. Nó xử lý nhiều chi tiết thực tế như dự đoán theo đợt, cân bằng tải và chia tỷ lệ.

#### Triển khai mô hình trên thiết bị

Đôi khi, bạn có thể cần mô hình của mình hoạt động trên cùng một thiết bị chạy ứng dụng sử dụng nó — có thể là điện thoại thông minh, CPU ARM nhúng trên robot hoặc bộ vi điều khiển trên một thiết bị nhỏ. Ví dụ: có lẽ bạn đã từng thấy một chiếc máy ảnh có khả năng tự động phát hiện người và khuôn mặt trong những cảnh bạn chỉ vào: đó có thể là một mô hình deep learning nhỏ chạy trực tiếp trên máy ảnh.

Bạn nên sử dụng thiết lập này khi

* Mô hình của bạn có những hạn chế nghiêm ngặt về độ trễ hoặc cần chạy trong một
môi trường kết nối thấp. Nếu bạn đang xây dựng một trò chơi nhập vai
ứng dụng thực tế tăng cường, việc truy vấn máy chủ từ xa không khả thi
lựa chọn.
* Mô hình của bạn có thể được làm đủ nhỏ để có thể chạy dưới
hạn chế về bộ nhớ và năng lượng của thiết bị mục tiêu.
* Đạt được độ chính xác cao nhất có thể không phải là nhiệm vụ quan trọng đối với nhiệm vụ của bạn:
luôn có sự cân bằng giữa hiệu quả thời gian chạy và độ chính xác, vì vậy
những hạn chế về bộ nhớ và năng lượng thường yêu cầu bạn gửi một mô hình không phù hợp
khá tốt như mô hình tốt nhất bạn có thể chạy trên GPU lớn.
* Dữ liệu đầu vào rất nhạy cảm và do đó không thể giải mã được
trên một máy chủ từ xa.

Ví dụ: mô hình phát hiện thư rác của chúng tôi sẽ cần chạy trên điện thoại thông minh của người dùng cuối như một phần của ứng dụng trò chuyện, vì tin nhắn được mã hóa hai đầu và do đó mô hình được lưu trữ từ xa hoàn toàn không thể đọc được. Tương tự như vậy, mô hình phát hiện cookie xấu có các hạn chế nghiêm ngặt về độ trễ và sẽ cần phải chạy tại nhà máy. Rất may, trong trường hợp này, chúng tôi không có bất kỳ hạn chế nào về nguồn điện hoặc không gian, vì vậy chúng tôi thực sự có thể chạy mô hình trên GPU.

Để triển khai mô hình Keras trên điện thoại thông minh hoặc thiết bị nhúng, bạn có thể sử dụng lại phương thức `export()` để tạo bản lưu TensorFlow hoặc ONNX cho mô hình của mình, bao gồm cả biểu đồ tính toán. TensorFlow Lite (<https://www.tensorflow.org/lite>) là một khung để suy luận deep learning hiệu quả trên thiết bị chạy trên điện thoại thông minh Android và iOS, cũng như CPU ​​ARM, Raspberry Pi hoặc một số bộ vi điều khiển nhất định. Nó sử dụng định dạng mô hình lưu TensorFlow tương tự như TensorFlow Serve. Thời gian chạy ONNX cũng có thể chạy trên thiết bị di động.

#### Triển khai mô hình trong trình duyệt

Học sâu thường được sử dụng trong các ứng dụng JavaScript dựa trên trình duyệt hoặc trên máy tính để bàn. Mặc dù thông thường có thể yêu cầu ứng dụng truy vấn một mô hình từ xa thông qua API REST, nhưng có thể có những lợi thế chính khi để mô hình chạy trực tiếp trong trình duyệt, trên máy tính của người dùng (sử dụng tài nguyên GPU nếu có).

Sử dụng thiết lập này khi

* Bạn muốn giảm tải điện toán cho người dùng cuối, điều này có thể
giảm đáng kể chi phí máy chủ.
* Dữ liệu đầu vào cần được lưu giữ trên máy tính hoặc điện thoại của người dùng cuối.
Ví dụ: trong dự án phát hiện thư rác của chúng tôi, phiên bản web và
phiên bản dành cho máy tính để bàn của ứng dụng trò chuyện (được triển khai dưới dạng ứng dụng đa nền tảng
được viết bằng JavaScript) nên sử dụng mô hình chạy cục bộ.
* Ứng dụng của bạn có những hạn chế nghiêm ngặt về độ trễ: trong khi một mô hình chạy trên
máy tính xách tay hoặc điện thoại thông minh của người dùng cuối có thể sẽ chậm hơn so với máy chạy trên
GPU lớn trên máy chủ của riêng bạn, bạn không có thêm 100 mili giây mạng
chuyến đi khứ hồi.
* Bạn cần ứng dụng của mình tiếp tục hoạt động mà không cần kết nối, sau khi mô hình đã kết nối
đã được tải xuống và lưu trữ.

Tất nhiên, bạn chỉ nên chọn tùy chọn này nếu kiểu máy của bạn đủ nhỏ để nó không ngốn CPU, GPU hoặc RAM của máy tính xách tay hoặc điện thoại thông minh của người dùng. Ngoài ra, vì toàn bộ mô hình sẽ được tải xuống thiết bị của người dùng nên bạn nên đảm bảo rằng không có nội dung nào về mô hình cần được giữ bí mật. Hãy lưu ý đến thực tế rằng, với một mô hình deep learning đã được huấn luyện, thường có thể khôi phục một số thông tin về dữ liệu huấn luyện: tốt hơn hết là bạn không nên công khai mô hình đã huấn luyện của mình nếu nó được huấn luyện trên dữ liệu nhạy cảm.

Để triển khai một mô hình bằng JavaScript, hệ sinh thái TensorFlow bao gồm TensorFlow.js (<https://www.tensorflow.org/js>) và ONNX hỗ trợ thời gian chạy JavaScript gốc. TensorFlow.js thậm chí còn triển khai gần như toàn bộ API Keras (ban đầu nó được phát triển dưới tên hoạt động là WebKeras) cũng như nhiều API TensorFlow cấp thấp hơn. Bạn có thể dễ dàng nhập mô hình Keras đã lưu vào TensorFlow.js để truy vấn mô hình đó như một phần của ứng dụng JavaScript dựa trên trình duyệt hoặc ứng dụng Electron trên máy tính để bàn của bạn.

#### Tối ưu hóa mô hình suy luận

Tối ưu hóa mô hình của bạn để suy luận đặc biệt quan trọng khi triển khai trong môi trường có những hạn chế nghiêm ngặt về nguồn điện và bộ nhớ khả dụng (điện thoại thông minh và thiết bị nhúng) hoặc cho các ứng dụng có yêu cầu về độ trễ thấp. Bạn phải luôn tìm cách tối ưu hóa mô hình của mình trước khi nhập mô hình vào TensorFlow.js hoặc xuất mô hình sang TensorFlow Lite.

Có hai kỹ thuật tối ưu hóa phổ biến mà bạn có thể áp dụng:

* *Giảm trọng lượng* — Không phải mọi hệ số trong tensor trọng số đều đóng góp
tương đương với các dự đoán. Có thể giảm đáng kể
số lượng tham số trong các lớp của mô hình của bạn bằng cách chỉ giữ
những điều quan trọng nhất.
Điều này làm giảm bộ nhớ và dung lượng điện toán của mô hình của bạn
với chi phí nhỏ về số liệu hiệu suất. Bằng cách điều chỉnh mức độ cắt tỉa của bạn
muốn đăng ký, bạn có thể kiểm soát sự cân bằng giữa kích thước và độ chính xác.
* *Lượng tử hóa trọng lượng* — Các mô hình học sâu được đào tạo với độ chính xác đơn
trọng số dấu phẩy động (`float32`).
Tuy nhiên, có thể *lượng tử hóa* trọng số thành số nguyên có dấu 8 bit
(`int8`) để chỉ suy luận
mô hình nhỏ hơn bốn lần nhưng vẫn gần đạt độ chính xác
của mô hình ban đầu. Các mô hình Keras đi kèm với API `quantize()` tích hợp
có thể giúp được việc này. Đơn giản chỉ cần gọi `model.quantize("int8")` để nén
mỗi trọng số trong mô hình của bạn thành một byte đơn.

### Giám sát mô hình của bạn trong tự nhiên

Bạn đã xuất một mô hình suy luận, bạn đã tích hợp nó vào ứng dụng của mình và bạn đã thực hiện chạy thử trên dữ liệu sản xuất — mô hình hoạt động chính xác như bạn mong đợi. Bạn đã viết các bài kiểm tra đơn vị cũng như mã ghi nhật ký và theo dõi trạng thái - hoàn hảo. Bây giờ là lúc nhấn nút lớn màu đỏ và triển khai vào sản xuất.

Thậm chí đây không phải là kết thúc. Sau khi triển khai một mô hình, bạn cần tiếp tục theo dõi hành vi của nó, hiệu suất của nó trên dữ liệu mới, sự tương tác của nó với phần còn lại của ứng dụng và tác động cuối cùng của nó đối với các số liệu kinh doanh:

* Mức độ tương tác của người dùng trên đài phát thanh trực tuyến của bạn tăng hay giảm sau khi triển khai phiên bản mới
hệ thống gợi ý âm nhạc? Tỷ lệ nhấp vào quảng cáo trung bình đã tăng lên sau khi chuyển sang
mô hình dự đoán tỷ lệ nhấp mới? Hãy cân nhắc sử dụng *thử nghiệm A/B ngẫu nhiên*
để tách biệt tác động của chính mô hình khỏi những thay đổi khác: a
tập hợp con của các trường hợp sẽ đi qua mô hình mới, trong khi tập hợp con điều khiển khác
nên tuân theo quy trình cũ. Một khi đã có đủ nhiều trường hợp
được xử lý, sự khác biệt trong kết quả
giữa hai điều này có thể là do mô hình.
* Nếu có thể, hãy thực hiện kiểm tra thủ công thường xuyên các dự đoán của mô hình
về dữ liệu sản xuất. Nói chung có thể tái sử dụng cùng một cơ sở hạ tầng
đối với chú thích dữ liệu: gửi một phần dữ liệu sản xuất tới
chú thích thủ công và so sánh dự đoán của mô hình với chú thích mới.
Ví dụ: bạn chắc chắn nên làm điều này cho công cụ tìm kiếm hình ảnh
và hệ thống gắn cờ cookie xấu.
* Khi không thể kiểm tra thủ công, hãy xem xét các cách đánh giá thay thế
chẳng hạn như khảo sát người dùng (ví dụ: trong trường hợp thư rác và xúc phạm
hệ thống gắn cờ nội dung).

### Duy trì mô hình của bạn

Cuối cùng, không có mô hình nào tồn tại mãi mãi. Bạn đã tìm hiểu về *sự trôi dạt khái niệm*: theo thời gian, các đặc điểm của dữ liệu sản xuất của bạn sẽ thay đổi, làm giảm dần hiệu suất và mức độ phù hợp của mô hình của bạn. Tuổi thọ của hệ thống giới thiệu âm nhạc của bạn sẽ được tính bằng tuần. Đối với hệ thống phát hiện gian lận thẻ tín dụng, sẽ là ngày; một vài năm trong trường hợp tốt nhất cho công cụ tìm kiếm hình ảnh.

Ngay sau khi mô hình của bạn được ra mắt, bạn nên sẵn sàng đào tạo thế hệ tiếp theo sẽ thay thế nó:

* Chú ý đến những thay đổi trong dữ liệu sản xuất. Là tính năng mới
trở nên có sẵn? Bạn nên mở rộng hoặc chỉnh sửa bộ nhãn?
* Tiếp tục thu thập và chú thích dữ liệu, đồng thời giữ
cải thiện quy trình chú thích của bạn theo thời gian. Đặc biệt, bạn nên
đặc biệt chú ý đến việc thu thập các mẫu có vẻ khó khăn
để phân loại cho mô hình hiện tại của bạn - những mẫu như vậy có nhiều khả năng nhất
để giúp cải thiện hiệu suất.

Điều này kết thúc quy trình làm việc phổ biến của học máy - có rất nhiều điều cần ghi nhớ. Cần có thời gian và kinh nghiệm để trở thành một chuyên gia, nhưng đừng lo lắng, bạn đã khôn ngoan hơn rất nhiều so với vài chương trước. Bây giờ bạn đã quen với bức tranh tổng thể - toàn bộ những gì mà các dự án học máy đòi hỏi. Mặc dù hầu hết cuốn sách này sẽ tập trung vào phần phát triển mô hình nhưng giờ đây bạn đã biết rằng đó chỉ là một phần trong toàn bộ quy trình làm việc. Hãy luôn ghi nhớ bức tranh lớn!

## Bản tóm tắt

* Khi bạn thực hiện một dự án machine learning mới, trước tiên, hãy xác định vấn đề trước mắt:
+ Hiểu bối cảnh rộng hơn của những gì bạn sắp làm —
mục tiêu cuối cùng là gì và những hạn chế là gì?
+ Thu thập và chú thích một tập dữ liệu; đảm bảo bạn hiểu dữ liệu của mình
chiều sâu.
+ Chọn cách bạn sẽ đo lường sự thành công trong vấn đề của mình. Bạn sẽ đo những số liệu nào
theo dõi dữ liệu xác nhận của bạn?
* Khi bạn hiểu được vấn đề và có tập dữ liệu phù hợp, hãy phát triển một mô hình:
+ Chuẩn bị dữ liệu.
+ Chọn giao thức đánh giá của bạn. Xác thực chờ đợi? Xác thực K-Fold?
Bạn nên sử dụng phần dữ liệu nào để xác thực?
+ Đạt được sức mạnh thống kê: đánh bại một đường cơ sở đơn giản.
+ Mở rộng quy mô: phát triển một mô hình có thể phù hợp quá mức.
+ Thường xuyên hóa mô hình của bạn và điều chỉnh các siêu tham số của nó, dựa trên hiệu suất trên
dữ liệu xác nhận. Rất nhiều nghiên cứu về học máy có xu hướng chỉ tập trung
ở bước này — nhưng hãy ghi nhớ bức tranh tổng thể.
* Khi mô hình của bạn đã sẵn sàng và mang lại hiệu suất tốt trên dữ liệu thử nghiệm,
đã đến lúc triển khai:
+ Đầu tiên, hãy đảm bảo đặt ra những kỳ vọng phù hợp với các bên liên quan.
+ Tối ưu hóa mô hình cuối cùng để suy luận và đưa mô hình đi triển khai
môi trường được lựa chọn - máy chủ web, thiết bị di động, trình duyệt, thiết bị nhúng, v.v.
+ Theo dõi hiệu suất của mô hình trong quá trình sản xuất và tiếp tục thu thập dữ liệu
để bạn có thể phát triển thế hệ tiếp theo của mô hình.

#### **Slide**

<div class="pdf-container" style="margin-bottom: 20px; width: 100%; height: 85vh;">
  <iframe src="TaiLieu/slideDL/Chapter06.pdf#view=FitH" width="100%" height="100%" style="border: none;"></iframe>
</div>


#### ** 🎥 Video **

<div class="video-mobile-wrapper">
<iframe src="TaiLieu/Video/Chapter_06/index.html" width="100%" height="600px" style="border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" allowfullscreen></iframe>
</div>

#### ** 📝 Bài tập Trắc nghiệm **

<iframe src="quizzes/Chapter06/index.html" style="width: 100%; min-height: 700px; border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></iframe>

<!-- tabs:end -->
