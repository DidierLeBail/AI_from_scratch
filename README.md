# diffusion models
The goal here is to analyze the generalization capabilities of a diffusion model generating images, without conditioning on text.

## framework used
To do this, let us first define the distribution learned by a model as the distribution according to which it generates samples.
This distribution depends on:
- the training set
- the training method
- the model trained

The training method is typically stochastic gradient descent (SGD), and the model trained is typically a convolutional neural network.
Here, the model will return a score vector function, that will be used to generate images by running backward Langevin dynamics.
This strategy is described e.g. in the lectures given by Professor Giulio Biroli and available [here](https://youtu.be/9b26lbsIt8c?si=JZf8J1N385J2lMfv).

## what we know from these lectures
We know that the learned distribution, if the model trained is expressive enough and SGD is run for a number of epochs of the order of the size of the training set, will match the training distribution.
In this case, the model can only generate samples included in its training set.

What we are interested in, is the training regime in which the model produces images of high quality, but yet not included in its training set.
In this regime, what is the learned distribution?

In practice, we want the model to learn a given distribution, that we call here the target distribution.
The typical strategy is to build the training set as samples drawn from this distribution.
Thus, it assumes that it is possible to learn a distribution from a finite number of examples.

In cases where it is possible to cover the full support of the target distribution, this strategy will be successful, on condition that the training set is large enough.
Here, a first mystery appears:
while a simple theoretical model predicts that the training set should be exponential in the dimension ($n=e^{\alpha d}$, $\alpha\sim1$) ; it is observed in practice that this generalizing regime is reached for a much smaller training set.
Here, we propose to take into account the discrete nature of the data, to show that in practice, the supports of the target distributions usually considered can be covered with a number of examples only polynomial in the dimension.

We also show a case where an exponential number of examples is actually needed, and in this case a training set of size exponential in the dimension is needed as expected (cf [experiment 1](#experiment-1-some-distributions-require-a-training-set-exponential-in-the-dimension)).

Even when neglecting the discrete nature of data, in some cases, the support cannot be covered fully, because it is of infinite volume.
Consider for example, the uniform distribution on the set of all polygons (with a number of vertices less than some upper bound).
Will a diffusion model (endowed with a typical architecture) be able to learn this distribution from a finite number of examples?
The answer seems to be no (cf [experiment 2](#experiment-2-some-simple-distributions-cannot-be-learned-by-typical-models)).

From this observation, two remarks can be made:

### remark 1
The learned distribution actually depends on the training set, and choosing the training set as drawn from the target distribution is not always the best choice (cf [experiment 3](#experiment-3-sampling-from-the-target-distribution-does-not-yield-an-optimal-training-set)).
Then, let us introduce the following concept:

We say that a model can learn a given distribution, if there is a finite training set, a training duration and a number of parameters, such that training a model of this size on this training set by SGD for that duration, will result in the model having learned the distribution.

### remark 2
From the same training set, a very simple model is able to generate polygons, so this distribution is learnable, conditioned to a good choice of model architecture (cf [experiment 4](#experiment-4-how-to-learn-all-polygons-from-backward-langevin-dynamics)).

Then, we can ask the following question:
Is there a model, that is able to learn any[^1] target distribution?
If yes, what does such a model look like?

PASS

## experiments
Now, let us describe each numerical experiment we have both imagined and performed.
For each experiment, we precise the model architecture and the training data.

## experiment 1: some distributions require a training set exponential in the dimension
In this experiment, we will consider two different target distributions:
1. one with a support polynomial in the dimension (polynomial support)
1. one with a support exponential in the dimension (exponential support)

The associated training sets are built by sampling from these distributions.
The model used is expressive enough so that it has the capacity to memorize the whole training set.

Then, we will determine the size of the training set, above which we observe the generalization regime at intermediate training time.

PASS

From the results obtained, the following conjecture can be made:

**The reason why learning e.g. the distribution of all faces is possible with a smaller training set than expected, may be simply because the number of all possible faces is polynomial in the dimension.**

Here are some simple arguments to support this conjecture:

PASS

### experiment 2: some simple distributions cannot be learned by typical models
Here we use the training set of polygons.
The model taken is large enough to memorize the training set, and the latter is sampled from the target distribution.

PASS

The conjecture accompanying this experiment is the following:

**To be learnable with the usual strategy and the usual generic image-processing models, the support of distribution must be both of finite volume and locally convex.**

The hypothesis of local convexity could be checked by considering e.g. a uniform distribution on a fractal set.

### experiment 3: sampling from the target distribution does not yield an optimal training set
Here, the target distribution still describes the set of all polygons and the same model is used.
Then, we want to find the training set that yields the best generalization capabilities of the model.

We use a genetic algorithm to explore the space of training sets, and then we estimate the minimum size required to achieve a successful generalization.

Humans do not learn from a finite number of examples.
Actually, it is easy to show that learning a full distribution from a finite number of samples is an ill-posed problem, since we do not give to the model how it should generalize beyond its these examples.
Instead, a teacher human describes a full distribution to a student human, without using any samples from it.
They transmit a symbolic description, of finite length and yet contains all information about the target distribution.
In contrast, a diffusion model is able to learn a distribution, only if the information contained in its training set matches the entropy of the target distribution.
Thus, using samples is a very inefficient way of describing a distribution.

Instead, we propose to use the training set as an encoding, a compressed description of the full target distribution.
The associated research questions are:
- what languages can we use to describe distributions?
- is there a language in which any computable distribution can be described with a sentence of finite length?
- what is the language implicitly used by humans? (note that the ARC-AGI benchmark would be useful to answer this)

### experiment 4: how to learn all polygons from backward Langevin dynamics
Here, we use the same training set, but a different model, crafted specifically to the distribution of all polygons.
Then, we show that the typical training strategy will result in this model to successfully generalize.

The takeaway of this experiment:

**Some models learn some distribution more easily. How to build a model that is able to learn every computable distribution using backard Langevin dynamics?**

## training data

### dataset 1
The training data we take is the set of black and white matrices $\sqrt{d}\times\sqrt{d}$, with a single white dot.
This training set in particular, in the sense that its maximal size is equal to $d$, i.e. $n=d$.

## model architectures

### model 1

[^1]: Of course, some restrictions should be considered, like the target distribution being computable.
