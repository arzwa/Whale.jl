
# Tutorial

This tutorial aims to make users familiar with the key components of the
Whale library. The material on this page should enable a user familiar with
Bayesian statistics or optimization to make use of the package for conducting
inference for models of gene family evolution from gene trees.

!!! warning
    This tutorial will not show how to get the input data (CCDs) for running
    analyses with Whale. Please consider the [Introduction](@ref) page.

## 0. Introduction

Whale is a library for conducting **statistical inference for phylogenetic
birth-death process models of gene family evolution** and **statistical gene
tree -- species tree reconciliation** using the **amalgamated likelihood
approximation** (Szöllősi et al. 2014) to the marginal sequence likelihood.

Let us unpack the above. Consider a sequence alignment $y$, a known species
tree $S$, a model of sequence evolution with parameters $\phi$ and a model of
gene family evolution with parameters $\theta$. The likelihood of a sequence
alignment $y$, marginalizing over gene tree topologies, is given by:

$$p(y|\phi,\theta,S) = \sum_{G \in \mathcal{G}_y}{\int}p(y|G, t, \phi)
  p(G, t|S, \theta)dt$$

Where we sum over the set $\mathcal{G}_y$ of all possible gene trees on the
alignment $y$ and integrate over all possible node ages in the tree $t$.
Clearly, this represents a very challenging inference problem.

When we are mainly interested in inference under the model of gene family
evolution (e.g. inference of reconciled trees, orthology relationships,
duplication and loss rates, etc.), we can adopt the ALE approximation to this
marginal likelihood, which takes the following form

$$p(y|\theta,S) = \sum_{G \in \mathcal{G}_y} p(y|G) p(G|S,\theta) \approx
\sum_{G \in \mathcal{G}_y} \xi_y(G) p(G|S,\theta) := q(y|\theta,S)$$

(note that $p(y|G) = \int_\phi p(y,\phi|G)$)). Here $\xi_y(G)$ serves as an
approximation to the sequence likelihood, and is assumed to be proportional
to $p(y|G)$.

In the ALE approach (for *amalgamated likelihood estimation*), we obtain a
$\xi_y(G)$ by constructing a *conditional clade distribution* (CCD) from a
sample of the posterior distribution of gene trees under the model of
sequence evolution, and using the probability mass function associated with
this CCD. That is $\xi_y(G) \propto p(y|G)p(G)$ under the model of sequence
evolution alone, which is $\propto p(y|G)$ when a uniform prior is used for
$G$. The reason we restrict ourselves to a CCD as approximating family is
that this allows for efficient computation of the potentially huge sum
$\sum_{G \in \mathcal{G}_y} \xi_y(G) p(G|S,\theta)$, as was shown by Szöllősi
et al.

With an approximation to $p(y|\theta,S)$ in hand, we can conduct inference
for the parameters of the model of gene family evolution $\theta$ using
either Bayesian inference or maximum likelihood estimation (MLE).
The Whale library implements inferential tools for **phylogenetic birth-death
process (BDP) models** of gene family evolution, potentially with **ancient
whole-genome duplication (WGDs)**.
Parameter inferences under the models which include ancient WGDs can be used
to assess WGD hypotheses based on gene trees.
With a little more work, we can also estimate (or rather sample) gene tree
topologies and their reconciliations under the model of gene family evolution
and the ALE approximation.
This then allows us to conduct probabilistic orthology assignments and
ancestral gene family reconstructions.
Viewed in this regard, the ALE and Whale approach provide a methodology to
conduct model-based gene tree -- species tree reconciliation and tree-based
orthology inference, while taking into account the uncertainty in the gene
tree topology.

## 1. Loading data and the `WhaleModel`

Before moving on to actual inference, we illustrate the key components of the
library. We start by loading the data and constructing a simple model object.

The minimally required packages to do something useful are the following:

````@example tutorial
using Whale, NewickTree
````

We will also load `Plots.jl` to have some graphics.

````@example tutorial
using Plots
default(legend=false, grid=false, framestyle=:box, size=(500,300))
````

We will use the land plant example data set (which is available in the github
repository).
We shall need a species tree

````@example tutorial
datadir = joinpath(@__DIR__, "../data/landplant")
tree = readnw(readline(joinpath(datadir, "speciestree.nw")))
````

... and a model object

````@example tutorial
rates = ConstantDLWGD(λ=0.1, μ=0.2, η=1/1.5)
model = WhaleModel(rates, tree, 0.01)
````

`rates` represents the model parameters: here we assume DLWGD (duplication,
loss and WGD) model with constant duplication and loss rates across the
species tree.
We assume a duplication rate $\lambda = 0.1$ and loss rate $\mu = 0.2$, with
the number of genes at the root of a gene family distributed according to a
Geometric distribution with mean $1.5$ (i.e. $\eta = 1/1.5$).
We combine the rates which parameterize the phylogenetic BDP model with the
species tree in a `WhaleModel` object (`model`). The last argument to
`WhaleModel` (here 0.01) determines the discretization of the species tree.
This entails that we assume there to be at most one *represented* gene
duplication event in each time slice of length $0.01$. This parameter should
be adjusted relative to the branch lengths of the species tree.

Note that when the model object is printed to the screen, we get quite some
information. In particular, under `Model structure` we get a CSV formatted
table with header `node_id, wgd_id, distance, Δt, n, subtree`, here `n` shows the
number of slices for the relevant branch of the species tree, i.e. the
maximum number of represented gene duplication events along that branch.
It is best to check whether these are large enough (at least 10 or so, except
for the root).

Now we can load the gene tree distributions (CCDs)

````@example tutorial
data = read_ale(joinpath(datadir, "100fams"), model)
````

Note that to load the CCDs (gene tree distributions), we need a `WhaleModel`
object!

## 2. The loglikelihood

We can now, for instance, compute the log-likelihood for the first family

````@example tutorial
ℓ = logpdf(model, data[1])
````

Or, assuming iid data, for the full data

````@example tutorial
ℓ = logpdf(model, data)
````

One can easily reparameterize the model using the following syntax:

````@example tutorial
model = model((λ=0.2,μ=0.1))
````

So that, for instance, we can easily graph a likelihood surface in the
following way

````@example tutorial
plot(0:0.01:2, λ->logpdf(model((λ=λ, μ=λ)), data[1]), ylabel="\$\\ell\$", xlabel="\$\\lambda\$")
````

(where we assumed $\lambda = \mu$).
Note that the loglikelihood can be differentiated using forward-mode
automatic differentiation (AD) with `ForwardDiff.jl`. The library is
currently not compatible with reverse mode AD or other fancy stuff.

## 3. Backtracking reconciled trees

We can sample a reconciled tree under the phylogenetic BDP using stochastic
backtracking. That is, we sample a *reconciled* gene tree conditional on the
*unreconciled* gene tree topology distribution and the likelihood under the
model of gene family evolution:

````@example tutorial
ℓ = logpdf!(model, data)
G = Whale.backtrack(model, data[1])
````

Note that the `logpdf!` step is crucial: the `data` is modified during the
likelihood computations to enable backtracking.

The reconciled tree consists of nodes which look like this:

````@example tutorial
prewalk(G)[1:10]
````

Here `σ` marks the species tree branch/node to which the gene tree node is
reconciled. If a gene tree node is marked as a `duplication` with `σ=3`, this
means for instance that this node represents a duplication event along the
branch leading to node 3 of the species tree. `t` marks the time point along
the branch of the species tree, going from present to past, where the gene
tree node is reconciled to. Speciation events have `t=0` because they are
mapped to the *nodes* in the species tree, instead of the branches (as
duplication events are).

There are some plotting functions available, some better than others. The
following plots the reconciled tree inside the species tree (duplication
nodes in red, speciation nodes in blue)

````@example tutorial
plot(model, G, sscale=50.)
````

toy around with `sscale` if the plot doesn't look good (or implement a new
function and do a pull request!). Simply plotting the reconciled tree is also
possible

````@example tutorial
plot(G, right_margin=30Plots.mm, size=(500,500))
````

## 4. Inference

With likelihoods available, one has many possibilities for conducting
inference. I suggest using the `Turing.jl` library for probabilistic
programming, potentially together with `Optim.jl` for maximum likelihood
or maximum *a posteriori* estimation.
You might want to get familiar with the basic syntax for specifying
probabilistic models using Turing, please consult the relevant docs and
tutorials at [https://turing.ml](https://turing.ml/dev/).

````@example tutorial
using Turing, Optim
````

We can for instance consider the $\lambda = \mu$ model for the first family,
i.e. the problem for which we computed the likelihood curve above.

````@example tutorial
@model simplemodel(M, y, ::Type{T}=Float64) where T = begin
    λ ~ Exponential(0.2)
    y ~ M((λ=λ, μ=λ, q=T[]))
end
````

Here we specify a probabilistic model with an Exponential(1) prior for the
duplication (=loss) rate using `Turing.jl` syntax. Some minor annoyances with
type stability lead to the `[...] ::Type{T}=Float64) where T` part and the
explicit passing of `q=T[]` in the model object, sorry.

We can obtain a sample from the posterior using the NUTS algorithm

````@example tutorial
chain = sample(simplemodel(model, data[1]), NUTS(), 200)
````

That makes sense. Alternatively, we can conduct MLE using `Optim` and
`Turing`:

````@example tutorial
optimize(simplemodel(model, data[1]), MLE())
````

Using probabilistic programs we can construct complicated hierarchical
models of gene family evolution. Here's a very slight elaboration of the
previous model, where we now assign a hyperprior to th number of lineages at
the root:

````@example tutorial
@model secondmodel(M, y, ::Type{T}=Float64) where T = begin
    λ ~ Exponential(0.2)
    η ~ Beta(4,2)
    y ~ M((λ=λ, μ=λ, q=T[], η=η))
end
````

we consider the first 10 families as data this time

````@example tutorial
chain = sample(secondmodel(model, data[1:10]), NUTS(), 200)
````

Let's compare the $\eta$ prior and posterior:

````@example tutorial
histogram(chain[:η], normalize=true, color=:white, xlabel="\$\\eta\$", ylabel="probability density")
plot!(0:0.01:1, x->pdf(Beta(4,2), x))
````

## 5. Sampling reconciled trees from the posterior

The interface for sampling reconciled trees from the posterior by stochastic
backtracking is not so user-friendly yet and is due for some updates, but it
is not difficult either. We first cast the chain as a data frame:

````@example tutorial
using DataFrames
df = DataFrame(chain);
nothing #hide
````

We have to define a function which takes a model and a row from the dataframe
(i.e. a sample from the posterior) and returns a model parameterized by that
sample.

````@example tutorial
modelfun(M, x) = M((λ=x[:λ], μ=x[:λ], η=x[:η]))
````

Next we define the tracker

````@example tutorial
tracker = TreeTracker(model, data[1:10], df, modelfun)
````

and we sample for each family 100 reconciled trees from the posterior, using
the posterior sample in `df`.

````@example tutorial
out = track(tracker, 100)
````

Each family has a `RecSummary` object, which among other things stores the
sampled reconciled trees

````@example tutorial
out[1].trees
````

This is the MAP tree of the first family:

````@example tutorial
out[1].trees[1]
````

We plot the MAP tree for the first family

````@example tutorial
plot(out[1].trees[1].tree, cred=true, size=(500,500), right_margin=30Plots.mm)
````

The numbers associated with each node indicate the estimated posterior
probability that the reconciled tree contains this *split* reconciled as a
duplication/speciation on that particular branch of the species tree.

## 6. Events and orthology

We can summarize the number of events on each branch of the species tree
using the following function:

````@example tutorial
smry = Whale.summarize(out)
smry.sum
````

Here we `summary[1,"duplication_mean"]` will for instance record the
posterior expected number of duplications per family along the branch leading
to node 1.

Another useful output are the posterior probabilities for pairs of genes to
be derived from certain events in the species tree

````@example tutorial
pairs = Whale.getpairs(out, model);
nothing #hide
````

For instance, we can take a look at the first 10 gene pairs in this data
frame. We first filter out all irrelevant columns:

````@example tutorial
idx = map(x->!all(x .== 0), eachcol(pairs[1:10,:]))
subset = pairs[1:10,idx]
````

The estimated posterior probabilty that gene pair

````@example tutorial
subset[1,"pair"]
````

... is derived from a duplication along the branch leading to node 21 is

````@example tutorial
subset[1,"21_duplication"]
````

## 7. Going further

For more details with regard to, for instance, the inference of ancient WGDs,
or fitting more complicated model, consider having a look at the examples.
If there are questions, comments, issues, or anything else, feel free to open
issues on the gitub repository.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

