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

Set up the model and the data

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
  πr: MvLogNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},FillArrays.Zeros{Float64,1,Tuple{Base.OneTo{Int64}}}}
  πq: Beta{Float64}
  πη: Beta{Float64}
)
```

Run HMC using [`DynamicHMC`](https://github.com/tpapp/DynamicHMC.jl)

```julia
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100,
    warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=3))
@info summarize_tree_statistics(results.tree_statistics)
```

Obtain the posterior distribution

```julia
posterior = transform.(Ref(problem), results.chain)
df = Whale.unpack(posterior)
first(df, 5)
```
```
5×5 DataFrames.DataFrame
│ Row │ λ        │ μ        │ q_1      │ q_2        │ η        │
│     │ Float64  │ Float64  │ Float64  │ Float64    │ Float64  │
├─────┼──────────┼──────────┼──────────┼────────────┼──────────┤
│ 1   │ 0.104749 │ 0.149621 │ 0.602009 │ 0.00971225 │ 0.86146  │
│ 2   │ 0.170998 │ 0.165641 │ 0.203148 │ 0.00327049 │ 0.803779 │
│ 3   │ 0.112158 │ 0.189674 │ 0.854411 │ 0.0824899  │ 0.803101 │
│ 4   │ 0.140092 │ 0.156727 │ 0.168705 │ 0.0675446  │ 0.847692 │
│ 5   │ 0.133129 │ 0.149424 │ 0.2378   │ 0.147512   │ 0.814924 │
```

Obtain reconciled trees sampled from the posterior

```julia
trees = sumtrees(problem, posterior)
```
```
12-element DistributedArrays.DArray{Whale.RecSummary,1,Array{Whale.RecSummary,1}}:
 RecSummary(# unique trees = 21)
 RecSummary(# unique trees = 12)
 RecSummary(# unique trees = 25)
 RecSummary(# unique trees = 45)
 RecSummary(# unique trees = 29)
 RecSummary(# unique trees = 21)
 RecSummary(# unique trees = 14)
 RecSummary(# unique trees = 32)
 RecSummary(# unique trees = 7)
 RecSummary(# unique trees = 8)
 RecSummary(# unique trees = 20)
 RecSummary(# unique trees = 22)
```

Consider the first gene family

```julia
family1 = trees[1].trees
```
```
21-element Array{NamedTuple,1}:
 (freq = 0.52, tree = RecTree(Node(83, 1, )))
 (freq = 0.15, tree = RecTree(Node(83, 1, )))
 (freq = 0.07, tree = RecTree(Node(83, 1, )))
 (freq = 0.04, tree = RecTree(Node(83, 1, )))
 (freq = 0.04, tree = RecTree(Node(83, 1, )))
 (freq = 0.03, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
 (freq = 0.01, tree = RecTree(Node(83, 1, )))
```

and a summary of the expected number of events for each branch

```julia
trees[1].events
```
```
19×6 DataFrames.DataFrame
│ Row │ duplication │ loss    │ speciation │ sploss  │ wgd     │ wgdloss │
│     │ Float64     │ Float64 │ Float64    │ Float64 │ Float64 │ Float64 │
├─────┼─────────────┼─────────┼────────────┼─────────┼─────────┼─────────┤
│ 1   │ 0.03        │ 0.0     │ 0.97       │ 0.06    │ 0.0     │ 0.0     │
│ 2   │ 0.01        │ 0.03    │ 0.99       │ 0.02    │ 0.0     │ 0.0     │
│ 3   │ 0.0         │ 0.01    │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 4   │ 0.0         │ 0.01    │ 0.0        │ 0.0     │ 0.0     │ 1.0     │
│ 5   │ 0.0         │ 1.0     │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 6   │ 0.02        │ 0.03    │ 0.99       │ 0.03    │ 0.0     │ 0.0     │
│ 7   │ 0.0         │ 0.02    │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 8   │ 0.99        │ 0.01    │ 1.0        │ 1.0     │ 0.0     │ 0.0     │
│ 9   │ 0.01        │ 1.0     │ 0.0        │ 0.0     │ 0.0     │ 1.01    │
│ 10  │ 0.02        │ 1.01    │ 0.99       │ 0.04    │ 0.0     │ 0.0     │
│ 11  │ 1.06        │ 0.01    │ 1.0        │ 1.08    │ 0.0     │ 0.0     │
│ 12  │ 0.0         │ 1.08    │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 13  │ 0.82        │ 0.0     │ 1.0        │ 1.9     │ 0.0     │ 0.0     │
│ 14  │ 0.0         │ 1.9     │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 15  │ 1.1         │ 0.0     │ 4.0        │ 0.0     │ 0.0     │ 0.0     │
│ 16  │ 0.0         │ 0.03    │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 17  │ 0.0         │ 0.0     │ 1.0        │ 1.0     │ 0.0     │ 0.0     │
│ 18  │ 0.0         │ 1.0     │ 1.0        │ 0.0     │ 0.0     │ 0.0     │
│ 19  │ 0.0         │ 0.0     │ 2.0        │ 0.0     │ 0.0     │ 0.0     │
```

get the MAP tree as a newick string

```julia
nwstr(family1[1].tree.root)
```
```
"(((PPAT_Pp3c9_4950V3.1_Pp3c9_4950:1.0,loss_8072255387089123919:1.0):1.0,MPOL_Mapoly0036s0119.1_Mapoly0036s0119:1.0):1.0,(SMOE_SMO111G0185.1_SMO111G0185:1.0,(((PABI_PAB00009793.1_PAB00009793:1.0,loss_849280707767022230:1.0):1.0,loss_13036997451547776001:1.0):1.0,((PABI_PAB00012681.1_PAB00012681:1.0,GBIL_Gb_13638:1.0):1.0,((ATRI_ATR0705G185.1_ATR0705G185:1.0,(((CPAP_Cpa.t.sc25.4_Cpa.g.sc25.4:1.0,loss_7444163200399977692:1.0):1.0,OSAT_LOC_Os07g08050.1_LOC_Os07g08050:1.0):1.0,(((ATHA_AT5G48120.1_AT5G48120:1.0,CPAP_Cpa.t.sc25.8_Cpa.g.sc25.8:1.0):1.0,((CPAP_Cpa.t.sc25.5_Cpa.g.sc25.5:1.0,CPAP_Cpa.t.sc25.3_Cpa.g.sc25.3:1.0):1.0,loss_5510716006007477900:1.0):1.0):1.0,loss_9305511805044521356:1.0):1.0):1.0):1.0,loss_10774710300463594448:1.0):1.0):1.0):1.0):1.0):1.0;"
```

## Reference

If you use Whale, please cite:

>[Zwaenepoel, A. and Van de Peer, Y., 2019. Inference of Ancient Whole-Genome Duplications and the Evolution of Gene Duplication and Loss Rates. *Molecular biology and evolution*, 36(7), pp.1384-1404.](https://academic.oup.com/mbe/article-abstract/36/7/1384/5475503)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

