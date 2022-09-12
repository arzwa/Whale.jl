# # Bayesian inference for the DLWGD model

# !!! note
#     Here it is assumed the reader is already familiar with the material
#     outlined in the [Tutorial](@ref).

using Whale, NewickTree, Distributions, Turing, DataFrames, LinearAlgebra
using Plots, StatsPlots
using Random; Random.seed!(7137);

# plotting defaults
default(grid=false, size=(500,800), titlefontsize=9, title_loc=:left, guidefont=8)

# ## Using a constant-rates model

# First we will do inference for a simple constant-rates model (i.e. assuming a
# single duplication and loss rate for the entire species tree). First we load
# the species tree (using the example tree available in the Whale library)
t = deepcopy(Whale.extree)
n = length(postwalk(t))  # number of internal nodes

# Now we add two WGD nodes to the tree. We do this by specifying the last
# common ancestor node for the lineages that share the hypothetical WGD. By
# default, the added node is halfway between the specified node and its parent.
insertnode!(getlca(t, "PPAT", "PPAT"), name="wgd_1")
insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_2")

# and we obtain a reference model object, using the constant-rates model with
# two WGDs
θ = ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2, 0.1], η=0.9)
w = WhaleModel(θ, t, .1, minn=10, maxn=20)

# next we get the data (we need a model object for that)
data = joinpath(@__DIR__, "../../example/example-1/ale")
ccd = read_ale(data, w)

# Now we define the Turing model
@model constantrates(model, ccd) = begin
    λ  ~ Exponential()
    μ  ~ Exponential()
    η  ~ Beta(3,1) 
    q1 ~ Beta() 
    q2 ~ Beta()
    ccd ~ model((λ=λ, μ=μ, η=η, q=[q1, q2]))
end

# In this model we have line by line:
# - `λ` and `μ`: the duplication and loss rate, for which we assume Exponential
#   priors.
# - `η`: the parameter of the geometric prior distribution on the number
#   of genes at the root (i.e. the Whale likelihood is integrated over
#   a geometric prior for the number of ancestral genes)
# - `q1`: the retention rate for the first WGD (with a uniform prior on [0,1], 
#   i.e. Beta(1,1))
# - `q2`: as `q1`
# - `ccd ~ model(...)`: here we specify the likelihood, we assume the data 
#   (`ccd`) is iid from the duplication+loss+WGD model with the relevant 
#   parameters.
chain0 = sample(constantrates(w, ccd), NUTS(), 200)

# Making some trace plots is straightforward using tools from the Turing
# probabilistic programming ecosystem
plot(chain0, size=(700,900))

# We can compute Bayes factors for the WGD hypotheses
summarize(chain0[[:q1,:q2]], Whale.bayesfactor)

# This is the log10 Bayes factor in favor of the $q = 0$ model. A Bayes factor
# smaller than -2 could be considered as evidence in favor of the $q \ne 0$
# model *compared to the $q=0$ model*. This in itself need not say much, as it
# says nothing about how well the model actually fits the data.


# !!! warning
#     Of course the chain should be run longer than in this example!
#     Here a short chain is presented to ensure reasonable build times for this
#     documentation. Generally, one should at least strive for ESS values
#     exceeding at least 100, although short chains may be good for exploring
#     and testing different models. 

# Now let's obtain reconciled trees
posterior = DataFrame(chain0)
ffun = (m, x)->m((λ=x[:λ], μ=x[:μ], η=x[:η], q=[x[:q1], x[:q2]])) 
tt = TreeTracker(w, ccd, posterior, ffun)
trees = track(tt, 1000)

# Note that `fun` is a function that takes the model object and a row from the
# posterior data frame, returning a model parameterized by the posterior
# sample in the row `x`.

# Let's have a look at the first family
trees[1].trees

# We can write these to a file using `Whale.writetrees("filename.nw",
# trees[1].trees)` if we would want that.  The maximum a posterior tree for
# this family is
map1 = trees[1].trees[1]
nwstr(map1.tree)

# with posterior probability
map1.freq

# Or maybe all the gene pairs
ps = Whale.getpairs(trees, w);

# Now let's look at the gene pairs which have a non-zero posterior probability
# of being derived from WGD node 18 (the *P. patens* WGD, execute `@show w` to
# check the model structure)
p = filter(x->x[Symbol("18_wgd")] > 0.0, ps)[!,:pair]

# Now we can look at the complete posterior reconciliation distribution for these
# gene pairs
df18 = filter(x->x[:pair] ∈ p, ps)
df18[:,[!(all(col .== 0)) for col in eachcol(df18)]]  # filter out all zero columns...

# The column names are `<branch id>_<event>`, and the entries are the posterior
# probability that the gene pair is reconciled to the relevant branch + event
# combination.

# Here we have for each WGD event in the tree all gene pairs that have non-zero
# posterior probability (as measured by the frequency in the posterior sample) 
# to be reconciled to the relevant WGD event.

# A table summarizing events for each branch can be obtained as well
smry = Whale.summarize(trees)
smry.sum

# here we have the expected number of duplications, losses, etc. per family for
# the different branches in the species tree (the row associated with a node
# corresponds to the branch leading to that node)

# ## Maximum-likelihood estimation

using Optim
result = optimize(constantrates(w, ccd), MLE())

# One could now compute the likelihood using the model without WGD 1 and/or 2
# and compare the likelihoods using a likelihood ratio test as in Rabier et al.
# (2014) to assess whether the data is compatible with the hypothesis $q = 0$
# (which should represent absence of a WGD).

# ## Using a branch-specific rates model

# Now we will consider a model with branch-specific duplication and loss rates,
# using a more complicated hierarchical model with an uncorrelated relaxed
# clock model.  We'll use the same tree as above. The relevant model now is the
# DLWGD model:
θ = DLWGD(λ=zeros(n), μ=zeros(n), q=rand(2), η=rand())
w = WhaleModel(θ, t, 0.1)
ccd = read_ale(data, w)

# Note that the duplication and loss rates should here be specified on a
# log-scale for the DLWGD model. We assume a Normal prior for the mean
# duplication and loss rates, and assume the log-scale branch-specific rates to
# be distributed according to a multivariate normal with diagonal covariance
# matrix $\tau I$. We assume duplication and loss rates to be independent.

@model branchrates(model, n, ccd, ::Type{T}=Float64) where T = begin
    η ~ Beta(3,1)
    λ̄ ~ Normal(log(0.15), 2)
    μ̄ ~ Normal(log(0.15), 2)
    τ ~ Exponential(0.1)
    λ ~ MvNormal(fill(λ̄, n-1), τ)
    μ ~ MvNormal(fill(μ̄, n-1), τ)
    q1 ~ Beta()
    q2 ~ Beta()
    ccd ~ model((λ=λ, μ=μ, η=η, q=[q1, q2]))
end

# ... and sample (this takes a bit longer!)

chain1 = sample(branchrates(w, n, ccd), NUTS(), 200)

# Make a plot for the retention parameters
plot(chain1[[:q1,:q2]], size=(700,300))

# ## Posterior predictive simulations
#
# We can do posterior predictive simulations to assess model fit. There are of
# course many possible posterior predictive observables that we may employ to
# do so. The approach below is but one that is (partially) implemented.
# Here we compare simulated number of events for each branch with reconstructed
# number of events for each branch. That is, for $N$ samples from the
# posterior, we (1) simulate a data set of the size of our empirical data set
# and (2) sample a reconciled tree for each gene family. We then compare, for
# instance, the number of duplications on the branch leading to node $m$ in the
# two simulated sets. If the model fits these should be similar.

# We need a function to get a parameterized model from a chain iterate:
function mfun(M, x)
    q1 = get(x, :q1).q1[1]
    q2 = get(x, :q2).q2[1]
    λ = vec(vcat(get(x, :λ).λ...))
    μ = vec(vcat(get(x, :μ).μ...))
    η = get(x, :η).η[1]
    M((λ=λ, μ=μ, q=[q1,q2], η=η))
end

# The following will then do 100 posterior predictive simulations
pps = Whale.ppsims(chain1, mfun, w, ccd, 100);

# and we make a plot for duplication events
lab = "duplication"
ps = map(w.order) do mnode
    l = "$(id(mnode))_$lab"
    dots = map(x->(x[1][l], x[2][l]), pps)
    xmn, xmx = extrema(vcat(first.(dots), first.(dots)))
    scatter(dots, color=:black, ms=2, alpha=0.5, legend=false, 
            xlabel="\$y\$", ylabel="\$\\tilde{y}\$", 
            title=Whale.cladelabel(mnode),
            xlim=(xmn-0.5,xmx+0.5), xticks=xmn:1:xmx)
    plot!(x->x, color=:lightgray)
end
plot(ps..., size=(700,600))

# one can of course also compare loss events, speciation events, WGD-derived
# duplicates etc.

# The above is somewhat more informative when analyzing more data. Using this
# approach on `chain0` and comparing against the results displayed here, one
# could for instance check whether the constant rates assumption is strongly
# violated or not. If the dots in these plots are not systematically above or
# below the 1-1 line, at least this aspect of the data is explained well by the
# model.
 
# !!! note
#     I know that this, and the sampling methods for reconciled trees, should
#     be implemented in a somewhat more consistent and user-friendly style.
#     It's on the to-do list.
