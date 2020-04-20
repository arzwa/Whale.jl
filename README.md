[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://arzwa.github.io/Whale.jl/dev/index.html)

# Whale: Bayesian gene tree reconciliation and whole-genome duplication inference by amalgamated likelihood estimation

```julia
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
```
```
LoadError("string", 3, ErrorException("syntax: invalid operator \"--\""))
```

Whale.jl is a julia library implementing joint inference of gene tree topologies and their reconciliations to a species tree using the **amalgamation** method of Szollosi et al. (2014) to compute the marginalize the reconciliation likelihood over a distribution over tree topologies. Whale implements the duplication-loss (DL) model of gene family evolution as well as a duplication-loss and whole-genome duplication (DLWGD) model (Rabier et al. 2014, Zwaenepoel et al. 2019). The latter can be used for the inference of ancient whole-genome duplications (WGDs) from gene trees while taking into account gene tree and reconciliation uncertainty.

The likelihood routines implemented in Whale support **automatic differentiation** using `ForwardDiff.jl`, allowing for efficient gradient-based Maximum-likelihood estimation and Hamiltonian Monte Carlo (HMC) based Bayesian inference. The library focuses on the Bayesian case, and implements relaxed clock priors to model the evolution of gene duplication and loss rates. Lastly, Whale allows to sample reconciled trees from the posterior distribution or a parameterized DL(+WGD) model using a stochastic backtracking agorithm (as in [ALE](https://github.com/ssolo/ALE)).

Please have a look at the [docs](https://arzwa.github.io/Whale.jl/dev/index.html) for usage instructions and documentation. You might want to get some minimal familiarity with the Julia REPL and its package manager when using Whale, see [the julia docs](https://docs.julialang.org/en/v1/).

Note that the scripts in the `scripts` directory might be helpful to prepare data for Whale analyses.

## Quick start/showcase

```julia
using Whale, DynamicHMC, Random, NewickTree
using DynamicHMC.Diagnostics
```

Set up the model and the data, here I use a model with constant duplication and loss rates across the species tree. Note that the tree contains two WGD events.

```julia
tree  = readnw("((MPOL:4.752,(PPAT:2.752)wgd_1:2.0):0.292,(SMOE:4.457,((((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:1.293):1.0)wgd_2:1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")
rates = Whale.ConstantDLWGD(λ=0.1, μ=0.1, q=[0.2, 0.3], η=0.9)
model = WhaleModel(rates, tree, Δt=0.1)
data  = read_ale(joinpath(@__DIR__, "example/example-1/ale"), model)
prior = CRPrior()
problem = WhaleProblem(data, model, prior)
```
```
WhaleProblem{Float64,Whale.RatesModel{Float64,Whale.ConstantDLGWGD{Float64},TransformVariables.TransformTuple{NamedTuple{(:λ, :μ, :q, :η),Tuple{TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ArrayTransform{TransformVariables.ScaledShiftedLogistic{Float64},1},TransformVariables.ScaledShiftedLogistic{Float64}}}}},CRPrior,UInt16}(CCD{UInt16,Float64}[CCD{UInt16,Float64}(Γ=83, 𝓛=13), CCD{UInt16,Float64}(Γ=55, 𝓛=13), CCD{UInt16,Float64}(Γ=89, 𝓛=13), CCD{UInt16,Float64}(Γ=131, 𝓛=13), CCD{UInt16,Float64}(Γ=107, 𝓛=13), CCD{UInt16,Float64}(Γ=59, 𝓛=13), CCD{UInt16,Float64}(Γ=53, 𝓛=13), CCD{UInt16,Float64}(Γ=83, 𝓛=13), CCD{UInt16,Float64}(Γ=59, 𝓛=13), CCD{UInt16,Float64}(Γ=95, 𝓛=13), CCD{UInt16,Float64}(Γ=67, 𝓛=13), CCD{UInt16,Float64}(Γ=65, 𝓛=13)], WhaleModel(
RatesModel with (:κ,) fixed
Whale.ConstantDLGWGD{Float64}
  λ: Float64 0.1
  μ: Float64 0.1
  q: Array{Float64}((2,)) [0.2, 0.3]
  κ: Float64 0.0
  η: Float64 0.9
), CRPrior
  πr: Distributions.MvLogNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},FillArrays.Zeros{Float64,1,Tuple{Base.OneTo{Int64}}}}
  πq: Distributions.Beta{Float64}
  πη: Distributions.Beta{Float64}
)
```

Run HMC using [`DynamicHMC`](https://github.com/tpapp/DynamicHMC.jl), (of course this is a ridicuously short run)

```julia
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100,
    warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=3))
summarize_tree_statistics(results.tree_statistics)
```
```
Hamiltonian Monte Carlo sample of length 100
  acceptance rate mean: 0.92, 5/25/50/75/95%: 0.77 0.88 0.96 0.99 1.0
  termination: divergence => 0%, max_depth => 0%, turning => 100%
  depth: 0 => 0%, 1 => 2%, 2 => 18%, 3 => 80%
```

Obtain the posterior distribution

```julia
posterior = transform.(Ref(problem), results.chain)
df = Whale.unpack(posterior)
first(df, 5)
```
```
5×5 DataFrames.DataFrame
│ Row │ λ        │ μ        │ q_1      │ q_2       │ η        │
│     │ Float64  │ Float64  │ Float64  │ Float64   │ Float64  │
├─────┼──────────┼──────────┼──────────┼───────────┼──────────┤
│ 1   │ 0.139883 │ 0.114902 │ 0.525921 │ 0.0823829 │ 0.862413 │
│ 2   │ 0.161491 │ 0.1439   │ 0.274991 │ 0.0225729 │ 0.771446 │
│ 3   │ 0.103524 │ 0.172376 │ 0.232388 │ 0.0952496 │ 0.833309 │
│ 4   │ 0.111649 │ 0.161276 │ 0.223542 │ 0.234012  │ 0.720894 │
│ 5   │ 0.151031 │ 0.181961 │ 0.538567 │ 0.0240763 │ 0.885975 │
```

Obtain reconciled trees sampled from the posterior

```julia
trees = sumtrees(problem, posterior)
```
```
12-element DistributedArrays.DArray{Whale.RecSummary,1,Array{Whale.RecSummary,1}}:
 RecSummary(# unique trees = 22)
 RecSummary(# unique trees = 10)
 RecSummary(# unique trees = 28)
 RecSummary(# unique trees = 45)
 RecSummary(# unique trees = 25)
 RecSummary(# unique trees = 18)
 RecSummary(# unique trees = 16)
 RecSummary(# unique trees = 24)
 RecSummary(# unique trees = 7)
 RecSummary(# unique trees = 12)
 RecSummary(# unique trees = 15)
 RecSummary(# unique trees = 30)
```

Consider the first gene family

```julia
family1 = trees[1].trees
```
```
22-element Array{NamedTuple,1}:
 (freq = 0.41, tree = Node(83, 1, ))
 (freq = 0.16, tree = Node(83, 1, ))
 (freq = 0.1, tree = Node(83, 1, ))
 (freq = 0.07, tree = Node(83, 1, ))
 (freq = 0.04, tree = Node(83, 1, ))
 (freq = 0.03, tree = Node(83, 1, ))
 (freq = 0.02, tree = Node(83, 1, ))
 (freq = 0.02, tree = Node(83, 1, ))
 (freq = 0.02, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
 (freq = 0.01, tree = Node(83, 1, ))
```

get the MAP tree as a newick string

```julia
nwstr(family1[1].tree)
```
```
"(((PPAT_Pp3c9_4950V3.1_Pp3c9_4950,loss_8072255387089123919)1.0,MPOL_Mapoly0036s0119.1_Mapoly0036s0119)0.98,(SMOE_SMO111G0185.1_SMO111G0185,(((PABI_PAB00009793.1_PAB00009793,loss_849280707767022230)1.0,loss_13036997451547776001)0.99,((PABI_PAB00012681.1_PAB00012681,GBIL_Gb_13638)0.98,((ATRI_ATR0705G185.1_ATR0705G185,(((CPAP_Cpa.t.sc25.4_Cpa.g.sc25.4,loss_7444163200399977692)0.98,OSAT_LOC_Os07g08050.1_LOC_Os07g08050)0.96,(((ATHA_AT5G48120.1_AT5G48120,CPAP_Cpa.t.sc25.8_Cpa.g.sc25.8)0.51,((CPAP_Cpa.t.sc25.5_Cpa.g.sc25.5,CPAP_Cpa.t.sc25.3_Cpa.g.sc25.3)0.47,loss_5510716006007477900)0.47)0.48,loss_9305511805044521356)0.86)0.93)0.99,loss_10774710300463594448)0.99)0.98)0.99)0.99)0.98;"
```

The support values are posterior probabilities for the associated reconciled split. Note that the tree does not contain branch lengths.

The events field for each gene family contains a summary of the expected number of events for each branch

```julia
trees[1].events
```
```
19×6 DataFrames.DataFrame
│ Row │ duplication │ loss    │ speciation │ sploss  │ wgd     │ wgdloss │
│     │ Float64     │ Float64 │ Float64    │ Float64 │ Float64 │ Float64 │
├─────┼─────────────┼─────────┼────────────┼─────────┼─────────┼─────────┤
│ 1   │ 0.02        │ 0.0     │ 1.0        │ 0.02    │ 0.0     │ 0.0     │
│ 2   │ 0.01        │ 0.01    │ 0.98       │ 0.04    │ 0.0     │ 0.0     │
│ 3   │ 0.0         │ 0.02    │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 4   │ 0.0         │ 0.02    │ 0.0        │ 0.0     │ 0.0     │ 1.0     │
│ 5   │ 0.0         │ 1.0     │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 6   │ 0.0         │ 0.01    │ 0.99       │ 0.02    │ 0.0     │ 0.0     │
│ 7   │ 0.0         │ 0.01    │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 8   │ 1.0         │ 0.01    │ 0.99       │ 1.01    │ 0.0     │ 0.0     │
│ 9   │ 0.0         │ 1.0     │ 0.0        │ 0.0     │ 0.01    │ 0.99    │
│ 10  │ 0.0         │ 0.99    │ 0.99       │ 0.02    │ 0.0     │ 0.0     │
│ 11  │ 1.08        │ 0.01    │ 1.0        │ 1.08    │ 0.0     │ 0.0     │
│ 12  │ 0.0         │ 1.08    │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 13  │ 0.89        │ 0.0     │ 1.0        │ 1.97    │ 0.0     │ 0.0     │
│ 14  │ 0.0         │ 1.97    │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 15  │ 1.03        │ 0.0     │ 4.0        │ 0.0     │ 0.0     │ 0.0     │
│ 16  │ 0.0         │ 0.01    │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 17  │ 0.03        │ 0.01    │ 0.98       │ 1.04    │ 0.0     │ 0.0     │
│ 18  │ 0.0         │ 1.02    │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 19  │ 0.0         │ 0.02    │ 2.0        │ 0.0     │ 0.0     │ 0.0     │
```

## Reference

If you use Whale, please cite:

>[Zwaenepoel, A. and Van de Peer, Y., 2019. Inference of Ancient Whole-Genome Duplications and the Evolution of Gene Duplication and Loss Rates. *Molecular biology and evolution*, 36(7), pp.1384-1404.](https://academic.oup.com/mbe/article-abstract/36/7/1384/5475503)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

