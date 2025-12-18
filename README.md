# AI_from_scratch
The goal is to implement from scratch various AI models and methods, to show off our understanding of these.

## diffusion models
The goal here is to analyze the generalization capabilities of a diffusion model.
More precisely, given a fixed model architecture, we want to determine what is the set learned by the model given a training set.
At a more fundamental level, we question the possibility to learn a set from a finite number of examples.
In particular, we question whether it is possible for a model to learn the set of all polygons.

If the training time is less than $\mathcal{O}(n)$, where $n$ is the size of the training set, then the diffusion model is expected to ``generalize'', meaning that the model has learned a set that strictly includes its training set.
However, the set learned has still not been characterized.
Here, we propose a simple numerical experiment, that would help to answer this question in a simple case.

At fixed training data, the only ingredients that can account for the learned set are the training method (SGD + particular cost function) and the kind of model trained.
Both these can introduce inductive biases (also called regularization in the literature) that will result in a specific kind of generalization.

Because this generalization is completely different from the way humans generalize, the set learned does not reveal ``hidden structure'' in the training data.
Instead, it reveals the inductive biases of a particular model trained with a particular method.

To show this, we will (1) train two different models using the same method, but these models having a different architecture (for example transformer vs convolutional network).
Then, we will look at the sets learned by each model.
The training data we take is the set of black and white matrices $\sqrt{d}\times\sqrt{d}$, with a single white dot.
This training set in particular, in the sense that its maximal size is equal to $d$, i.e. $n=d$.
This regime is far from the regime $n=e^{\alpha d}$, where a perfect model is supposed to exhibit generalization at some point in the backward Langevin dynamics.
However, in practice people acknowledge that generalization is possible even for $n\ll e^{\alpha d}$.

(2) we will ask the question: given a model (and a training method), what should be the training data that allow this model to learn a particular set?
If this set is finite, it is enough to take it as the training set and push the learning long enough to reach the memorization regime.
But if this set is infinite, such as the set of all polygons or the set of matrices with $k$ white separated dots ($k$ being variable), how should we proceed?
