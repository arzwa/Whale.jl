# [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://arzwa.github.io/Whale.jl/dev/index.html)

# # Whale: Bayesian gene tree reconciliation and whole-genome duplication inference by amalgamated likelihood estimation

```

                             .-------------'```'----....,,__                        _,
                            |                               `'`'`'`'-.,.__        .'(
                            |                                             `'--._.'   )
                            |                                                   `'-.<
                            \               .-'`'-.                            -.    `\
                             \               -.o_.     _                     _,-'`\    |
                              ``````''--.._.-=-._    .'  \            _,,--'`      `-._(
                                (^^^^^^^^`___    '-. |    \  __,,..--'                 `
                                 `````````   `'--..___\    |`
                                                       `-.,'
```

# Whale.jl is a julia library implementing joint inference of gene tree topologies and their reconciliations to a species tree using the **amalgamation** method of Szollosi et al. (2014) to compute the marginalize the reconciliation likelihood over a distribution over tree topologies. Whale implements the duplication-loss (DL) model of gene family evolution as well as a duplication-loss and whole-genome duplication (DLWGD) model (Rabier et al. 2014, Zwaenepoel et al. 2019). The latter can be used for the inference of ancient whole-genome duplications (WGDs) from gene trees while taking into account gene tree and reconciliation uncertainty.

# The likelihood routines implemented in Whale support **automatic differentiation** using `ForwardDiff.jl`, allowing for efficient gradient-based Maximum-likelihood estimation and Hamiltonian Monte Carlo (HMC) based Bayesian inference. The library focuses on the Bayesian case, and implements relaxed clock priors to model the evolution of gene duplication and loss rates. Lastly, Whale allows to sample reconciled trees from the posterior distribution or a parameterized DL(+WGD) model using a stochastic backtracking agorithm (as in [ALE](https://github.com/ssolo/ALE)).

# Please have a look at the [docs](https://arzwa.github.io/Whale.jl/dev/index.html) for usage instructions and documentation. You might want to get some minimal familiarity with the Julia REPL and its package manager when using Whale, see [the julia docs](https://docs.julialang.org/en/v1/).

# Note that the scripts in the `scripts` directory might be helpful to prepare data for Whale analyses.

# ## Quick start/showcase

using Whale, DynamicHMC, Random, NewickTree
using DynamicHMC.Diagnostics

# Set up the model and the data
tree  = readnw("((MPOL:4.752,(PPAT:2.752)wgd_1:2.0):0.292,(SMOE:4.457,((((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:1.293):1.0)wgd_2:1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")
rates = Whale.ConstantDLWGD(λ=0.1, μ=0.1, q=[0.2, 0.3], η=0.9)
model = WhaleModel(rates, tree, Δt=0.1)
data  = read_ale(joinpath(@__DIR__, "../example/example-1/ale"), model)
prior = CRPrior()
problem = WhaleProblem(data, model, prior)

# Run HMC using [`DynamicHMC`](https://github.com/tpapp/DynamicHMC.jl)
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100,
    warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=3))

# Obtain the posterior distribution
posterior = transform.(Ref(problem), results.chain)

# Obtain reconciled trees sampled from the posterior
trees = sumtrees(problem, posterior)

# Consider the first gene family
trees[1].trees

# and
trees[1].events

# ## Reference

# If you use Whale, please cite:

# >[Zwaenepoel, A. and Van de Peer, Y., 2019. Inference of Ancient Whole-Genome Duplications and the Evolution of Gene Duplication and Loss Rates. *Molecular biology and evolution*, 36(7), pp.1384-1404.](https://academic.oup.com/mbe/article-abstract/36/7/1384/5475503)