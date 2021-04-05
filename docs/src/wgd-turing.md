
# Bayesian inference using `Turing.jl`

Note this documentation is generated from a julia script using `Literate.jl`.
You can find the associated script by following the 'edit on github' link
on top.

In this example we will use the probabilistic programming language
implemented in [`Turing.jl`](https://turing.ml/dev/) with Whale to specify
Bayesian hierarchical models for gene tree reconciliation in a flexible way

```@example wgd-turing
using Whale, NewickTree, Distributions, Turing, DataFrames, LinearAlgebra, Random
using Plots, StatsPlots
Turing.turnprogress(false)  # you probably don't want to do this
Random.seed!(7137);
nothing #hide
```

## Using a constant-rates model

First we will do inference for a simple constant-rates model (i.e. assuming a
single duplication and loss rate for the entire species tree). First we load
the species tree (using the example tree available in the WHale library)

```@example wgd-turing
t = deepcopy(Whale.extree)
n = length(postwalk(t))  # number of internal nodes
```

Now we add two WGD nodes to the tree. We do this by specifying the last
common ancestor node for the lineages that share the hypothetical WGD. By
default, the added node is halfway between the specified node and its parent.

```@example wgd-turing
insertnode!(getlca(t, "PPAT", "PPAT"), name="wgd_1")
insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_2")
```

and we obtain a reference model object, using the constant-rates model with
two WGDs

```@example wgd-turing
θ = ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2, 0.1], η=0.9)
r = Whale.RatesModel(θ, fixed=(:p,))
w = WhaleModel(r, t, .1)
```

Note the last argument to `WhaleModel`, this is the slice length `Δt`, here
set to `0.1`. This determines the discretization of the branches of the
species tree, and may affect the accuracy of the ALE likelihood. While the
transition probabilities over the slices are exact, the number of slices on
a branch constrains the maximum number of duplications possible along the
branch. By default, Whale will ensure 5 slices on each branch if your `Δt`
would be chosen too coarse. Note that the information printed to the `stdout`
for the `WhaleModel` struct gives you this information, e.g. the line

```
10,0,0.292,0.0584,5,"(MPOL,(PPAT));"
```

indicates that the branch leading to node 10 (which is not a WGD), has length
`0.292` and is sliced in 5 slices of length `0.0584`.

next we get the data (we need a model object for that)

```@example wgd-turing
ccd = read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), w)
```

Now we define the Turing model

```@example wgd-turing
@model constantrates(model, ccd) = begin
    r  ~ MvLogNormal(ones(2))
    η  ~ Beta(3,1)
    q1 ~ Beta()
    q2 ~ Beta()
    ccd ~ model((λ=r[1], μ=r[2], η=η, q=[q1, q2]))
end
```

In this model we have line by line:
- `r`: the duplication and loss rate, for which we assume a bivariate
  lognormal prior with mean `log(1)=0`
- `η`: the parameter of the geometric prior distribution on the number
  of genes at the root (i.e. the Whale likelihood is integrated over
  a geometric prior for the number of ancestral genes)
- `q1`: the retention rate for the first WGD (with a uniform prior on [0,1],
  i.e. Beta(1,1))
- `q2`: as `q1`
- `ccd ~ model(...)`: here we specify the likelihood, we assume the data
  (`ccd`) is iid from the duplication+loss+WGD model with the relevant
  parameters.

```@example wgd-turing
model = constantrates(w, ccd)
chain = sample(model, NUTS(0.65), 100)
```

Making some trace plots is straightforward using tools from the Turing
probabilistic programming ecosystem

```@example wgd-turing
aesthetics = (grid=false, size=(500,800), titlefontsize=9,
              title_loc=:left, guidefont=8, color=:black)
plot(chain; aesthetics...)
```

!!! warning
    Of course such a chain should be run much longer than in this example!
    Here a very short chain is presented to ensure reasonable build times for
    this documentation. Generally, one should at least strive for ESS values
    exceeding 100 although short chains may be good for exploring and testing
    different models.

Now let's obtain reconciled trees

```@example wgd-turing
pdf = DataFrame(chain)[13:end]
fun = (m, x)-> Array(x) |> x->m((λ=x[3], μ=x[4], η=x[end], q=x[1:2]))
tt = TreeTracker(w, ccd, pdf, fun)
trees = track(tt)
```

Let's have a look at the first family

```@example wgd-turing
trees[1].trees
```

The maximum a posterior tree for this family is

```@example wgd-turing
map1 = trees[1].trees[1]
nwstr(map1.tree)
```

with posterior probability

```@example wgd-turing
map1.freq
```

Or maybe all the gene pairs

```@example wgd-turing
ps = Whale.getpairs(trees, w);
nothing #hide
```

Now let's look at the gene pairs which have a non-zero posterior probability
of being derived from WGD node 18 (the *P. patens* WGD, execute `@show w` to
check the model structure)

```@example wgd-turing
p = filter(x->x[Symbol("18_wgd")] > 0.0, ps)[!,:pair]
```

Now we can look at the complete posterior reconciliation distribution for these
gene pairs

```@example wgd-turing
df18 = filter(x->x[:pair] ∈ p, ps)
df18[[!(all(col .== 0)) for col in eachcol(df18)]]  # filter out all zero columns...
```

The column names are `<branch id>_<event>`, and the entries are the posterior
probability that the gene pair is reconciled to the relevant branch + event
combination.

The following can also be helpful

```@example wgd-turing
tables = Whale.getwgdtables(trees, ccd, w)
tables
```

Here we have for each WGD event in the tree all gene pairs that have non-zero
posterior probability (as measured by the frequency in the posterior sample)
to be reconciled to the relevant WGD event.

## Using a branch-specific rates model

Now we will consider a model with branch-specific duplication and loss rates,
using a more complicated hierarchical model with an bivariate uncorrelated
relaxed clock prior.  We'll use the same tree as above. The relevant model
now is the DLWGD model:

```@example wgd-turing
params = DLWGD(λ=randn(n), μ=randn(n), q=rand(2), η=rand())
r = Whale.RatesModel(params, fixed=(:p,))
w = WhaleModel(r, t, 0.1)
ccd = read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), w)
```

Note that the duplication and loss rates should here be specified on a
log-scale for the DLWGD model. We use an LKJ prior for the covariance matrix,
specifying a prior for the correlation of duplication and loss rates (`ρ`)
and a prior for the scale parameter `τ` (see e.g. the [stan
docs](https://mc-stan.org/docs/2_23/stan-users-guide/multivariate-hierarchical-priors-section.html)):

```@example wgd-turing
@model branchrates(model, ccd, ::Type{T}=Float64) where T = begin
    η ~ Beta(3,1)
    ρ ~ Uniform(-1, 1.)
    τ ~ Exponential()
    S = [τ 0. ; 0. τ]
    R = [1. ρ ; ρ 1.]
    Σ = S*R*S
    !isposdef(Σ) && return -Inf
    r = Matrix{T}(undef, 2, n)
    o = id(getroot(model))
    r[:,o] ~ MvNormal(zeros(2), ones(2))
    for i=1:n
        i == o && continue
        r[:,i] ~ MvNormal(r[:,o], Σ)
    end
    q1 ~ Beta()
    q2 ~ Beta()
    ccd ~ model((λ=r[1,:], μ=r[2,:], η=η, q=[q1, q2]))
end
```

In this model we store the mean duplication and loss rate across branches at
the root index (or in other words, we interpret the rates at the root node as
the expected rates for the branches in the tree).

```@example wgd-turing
chain = sample(branchrates(w, ccd), NUTS(0.65), 100)
```

Of course, again bear in mind that in real applications you will want to take
larger samples, e.g. 1000 instead of 100.

Now let's obtain reconciled trees. Note that the function `fun` below is
a function that takes a row from the posterior data frame and returns a
parameterized Whale model.

```@example wgd-turing
pdf = DataFrame(chain)[13:end]
fun = (m, x)-> Array(x) |> x->m((λ=x[3:2:36], μ=x[4:2:36], η=x[end-2], q=x[1:2]))
tt = TreeTracker(w, ccd[end-1:end], pdf, fun)
trees = track(tt)
```

The rest is the same as above.

## A critical branch-specific rates model

It may also be of interest to specify a similar model with a single
'turnover' rate for each branch, i.e.  enforcing `λ = μ` for each branch, but
allowing this rate to vary across branches. A birth-death process with this
property is said to be a *critical* birth-death process. It is
straightforward to specify a hierarchical model for this:

```@example wgd-turing
@model critical(model, ccd, ::Type{T}=Float64) where {T} = begin
    η ~ Beta(3,1)
    σ ~ Exponential()
    r = Vector{T}(undef, n)
    o = id(getroot(model))
    r[o] ~ Turing.Flat()
    for i=1:n
        i == o && continue
        r[i] ~ Normal(r[o], σ)
    end
    q1 ~ Beta()
    q2 ~ Beta()
    ccd ~ model((λ=r, μ=r, η=η, q=[q1, q2]))
end

Random.seed!(54)
chain = sample(critical(w, ccd), NUTS(0.65), 100)
```

Note that this model seems to be somewhat easier to sample from, as can be judged
by the ESS values. The results, although based on a small data set and a very
short chain, already seem to suggest that the gene trees contain some evidence
for a *P. patens* WGD (as `q1` seems to be decidedly non-zero).

!!! note
    Often it can be beneficial to set the hyperparameter `η` to a fixed value
    based on the data. `η` is the parameter for the shifted geometric prior
    on the number of genes at the root of the gene tree across families, so
    that the expected number of genes at the root is `1/η`. Under the
    assumption that the evolutonary process is more or less stationary (i.e.
    there is no systematic growth or contraction of families across the
    genome), we may wish to set `η` to the average number of genes in a
    family in a genome observed in the data.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

