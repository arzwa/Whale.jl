
# Bayesian inference using NUTS with `DynamicHMC.jl`

```@example wgd-dhmc
using Whale, DynamicHMC, Random, NewickTree, Distributions, DataFrames
using DynamicHMC.Diagnostics
Random.seed!(562)
```

Set up the model and the data, here I use a model with constant duplication and loss rates across the species tree. Note that the tree contains two WGD events.

```@example wgd-dhmc
tree  = readnw("((MPOL:4.752,(PPAT:2.752)wgd_1:2.0):0.292,(SMOE:4.457,((((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.0)wgd_2:0.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")
n = length(postwalk(tree))
ntaxa = (n+1)÷2
rates = RatesModel(ConstantDLWGD(λ=0.1, μ=0.1, q=[0.2, 0.3], η=0.9))
model = WhaleModel(rates, tree, 0.1)
data  = read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), model, true)
prior = Whale.CRPrior()
problem = WhaleProblem(data, model, prior)
```

Run NUTS using [`DynamicHMC`](https://github.com/tpapp/DynamicHMC.jl), (of course this is a ridicuously short run, and it's better to keep `doubling_stages` >= 3)

```@example wgd-dhmc
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100,
    warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=2))
summarize_tree_statistics(results.tree_statistics)
```

Obtain the posterior distribution

```@example wgd-dhmc
posterior = Whale.transform(problem, results.chain)
df = Whale.unpack(posterior)
describe(df)
```

Obtain reconciled trees sampled from the posterior

```@example wgd-dhmc
trees = track(problem, posterior)
```

Consider the first gene family

```@example wgd-dhmc
family1 = trees[1].trees
```

get the MAP tree as a newick string

```@example wgd-dhmc
nwstr(family1[1].tree)
```

The support values are posterior probabilities for the associated reconciled split. Note that the tree does not contain branch lengths.

The events field for each gene family contains a summary of the expected number of events for each branch

```@example wgd-dhmc
trees[1].events
```

We can get for every gene pair the posterior reconciliation probability

```@example wgd-dhmc
pair_pps = Whale.getpairs(trees, model)
first(pair_pps, 5)
```

which sum to one, as we can verify

```@example wgd-dhmc
map(sum, eachrow(pair_pps[!,1:end-2]))
```

Take for instance the following gene pair (second row)

```@example wgd-dhmc
x = pair_pps[2,:]
for (n, v) in zip(names(x), Array(x))
    (!(typeof(v)<:Number) || v > 0.) && println(n, ": ", v)
end
```

The posterior probability (under the DL model) that this gene pair traces back to the speciation corresponding to node 17 (i.e. the root) is approximately 0.86, whereas the posterior probability that this gene pair traces back to an acnestral duplication event is 0.14.

We can get a WGD-centric view as well. The following retrieves a table for each WGD with all gene tree nodes that have a non-zero psoterior probability of being reconciled to that particular WGD node

```@example wgd-dhmc
tables = Whale.getwgdtables(trees, data, model)
tables[1]
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

