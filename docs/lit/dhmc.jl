# ## Quick start/showcase

using Whale, DynamicHMC, Random, NewickTree, Distributions
using DynamicHMC.Diagnostics

# Set up the model and the data, here I use a model with constant duplication and loss rates across the species tree. Note that the tree contains two WGD events.
tree  = readnw("((MPOL:4.752,(PPAT:2.752)wgd_1:2.0):0.292,(SMOE:4.457,((((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.0)wgd_2:0.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")
n = length(postwalk(tree))
ntaxa = (n+1)÷2
# rates = RatesModel(
#     ConstantDLWGD(λ=0.1, μ=0.1, q=[0.2, 0.3], η=0.9, p=rand(ntaxa) .* 0.1))
rates = RatesModel(
    DLWGD(λ=randn(n), μ=randn(n), q=[0.2, 0.3], η=0.9, p=rand(ntaxa) .* 0.1))
model = WhaleModel(rates, tree, Δt=0.1)
data  = read_ale(joinpath("example/example-1/ale"), model)
# prior = Whale.CRPriorMissing(πp=[Beta(1,24) for i=1:ntaxa])
prior = Whale.IWIRPriorMissing(πp=[Beta(1,24) for i=1:ntaxa])
problem = WhaleProblem(data, model, prior)

# Run HMC using [`DynamicHMC`](https://github.com/tpapp/DynamicHMC.jl), (of course this is a ridicuously short run)
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 500,
    warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=3))
summarize_tree_statistics(results.tree_statistics)

# Obtain the posterior distribution
posterior = transform.(Ref(problem), results.chain)
df = Whale.unpack(posterior)
first(df, 5)

# Obtain reconciled trees sampled from the posterior
trees = sumtrees(problem, posterior)

# Consider the first gene family
family1 = trees[1].trees

# get the MAP tree as a newick string
nwstr(family1[1].tree)

# The support values are posterior probabilities for the associated reconciled split. Note that the tree does not contain branch lengths.

# The events field for each gene family contains a summary of the expected number of events for each branch
trees[1].events

