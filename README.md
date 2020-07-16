[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://arzwa.github.io/Whale.jl/dev/index.html)
[![Build Status](https://travis-ci.com/arzwa/Whale.jl.svg?branch=master)](https://travis-ci.com/arzwa/Whale.jl)

# Whale: Bayesian gene tree reconciliation and whole-genome duplication inference by amalgamated likelihood estimation

```julia
#```
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

```julia
#```
```

Whale.jl is a julia library implementing joint inference of gene tree topologies and their reconciliations to a species tree using the **amalgamation** method of Szollosi et al. (2014) to compute the marginalize the reconciliation likelihood over a distribution over tree topologies. Whale implements the duplication-loss (DL) model of gene family evolution as well as a duplication-loss and whole-genome duplication (DLWGD) model (Rabier et al. 2014, Zwaenepoel et al. 2019). The latter can be used for the inference of ancient whole-genome duplications (WGDs) from gene trees while taking into account gene tree and reconciliation uncertainty.

The likelihood routines implemented in Whale support **automatic differentiation** using `ForwardDiff.jl`, allowing for efficient gradient-based Maximum-likelihood estimation and Hamiltonian Monte Carlo (HMC) based Bayesian inference. The library focuses on the Bayesian case, and implements relaxed clock priors to model the evolution of gene duplication and loss rates. Lastly, Whale allows to sample reconciled trees from the posterior distribution or a parameterized DL(+WGD) model using a stochastic backtracking agorithm (as in [ALE](https://github.com/ssolo/ALE)).

Please have a look at the [docs](https://arzwa.github.io/Whale.jl/dev/index.html) for usage instructions and documentation. You might want to get some minimal familiarity with the Julia REPL and its package manager when using Whale, see [the julia docs](https://docs.julialang.org/en/v1/).

Note that the scripts in the `scripts` directory might be helpful to prepare data for Whale analyses.

## Quickstart using Turing and a constant-rates model

```julia
using Whale, NewickTree, Distributions, Turing, DataFrames
```

Get the tree

```julia
t = deepcopy(Whale.extree)
n = length(postwalk(t))  # number of internal nodes
```

```
17
```

Now we add two WGD nodes to the tree. We do this by specifying
the last common ancestor node for the lineages that share the
hypothetical WGD. By default, the added node is halfway between
the specified node and its parent.

```julia
insertnode!(getlca(t, "ATHA", "ATHA"), name="wgd")
insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd")
```

```
0.6125
```

and we obtain a reference model object, here we will use a constant-rates
model

```julia
θ = ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2, 0.1], η=0.9)
r = Whale.RatesModel(θ, fixed=(:p,))
w = WhaleModel(r, t, .1)
```

```
WhaleModel
——————————
⋅Parameterization:
RatesModel with (:p,) fixed
ConstantDLWGD{Float64,Float64}
  λ: Float64 0.1
  μ: Float64 0.2
  q: Array{Float64}((2,)) [0.2, 0.1]
  p: Array{Float64}((0,)) Float64[]
  η: Float64 0.9

⋅Condition:
Whale.RootCondition

⋅Model structure:
19 nodes (9 leaves, 2 WGD nodes)
node_id,wgd_id,distance,Δt,n,subtree
1,0,4.752,0.099,48,"MPOL:4.7520000000000024;"
2,0,4.752,0.099,48,"PPAT:4.7520000000000024;"
3,0,4.457,0.099,45,"SMOE:4.456999999999998;"
4,0,3.178,0.0993,32,"GBIL:3.1779999999999986;"
5,0,3.178,0.0993,32,"PABI:3.1779999999999986;"
6,0,1.555,0.0972,16,"OSAT:1.555;"
7,0,0.5548,0.0925,6,"CPAP:0.5548;"
8,0,0.2774,0.0555,5,"ATHA:0.2774;"
9,0,2.293,0.0997,23,"ATRI:2.293;"
10,0,0.292,0.0584,5,"(MPOL:4.7520000000000024,PPAT:4.7520000000000024):0.292;"
11,0,0.34,0.068,5,"(GBIL:3.1779999999999986,PABI:3.1779999999999986):0.34;"
18,2,0.2774,0.0555,5,"(ATHA:0.2774):0.2774;"
12,0,1.0002,0.0909,11,"(CPAP:0.5548,(ATHA:0.2774):0.2774):1.0002;"
13,0,0.738,0.0922,8,"(OSAT:1.555,(CPAP:0.5548,(ATHA:0.2774):0.2774):1.0002):0.738;"
14,0,0.6125,0.0875,7,"((OSAT:1.555,(CPAP:0.5548,(ATHA:0.2774):0.2774):1.0002):0.738,ATRI:2.293):0.6125;"
19,1,0.6125,0.0875,7,"(((OSAT:1.555,(CPAP:0.5548,(ATHA:0.2774):0.2774):1.0002):0.738,ATRI:2.293):0.6125):0.6125;"
15,0,0.939,0.0939,10,"((GBIL:3.1779999999999986,PABI:3.1779999999999986):0.34,(((OSAT:1.555,(CPAP:0.5548,(ATHA:0.2774):0.2774):1.0002):0.738,ATRI:2.293):0.6125):0.6125):0.939;"
16,0,0.587,0.0978,6,"(SMOE:4.456999999999998,((GBIL:3.1779999999999986,PABI:3.1779999999999986):0.34,(((OSAT:1.555,(CPAP:0.5548,(ATHA:0.2774):0.2774):1.0002):0.738,ATRI:2.293):0.6125):0.6125):0.939):0.587;"
17,0,0.0,0.0,0,"((MPOL:4.7520000000000024,PPAT:4.7520000000000024):0.292,(SMOE:4.456999999999998,((GBIL:3.1779999999999986,PABI:3.1779999999999986):0.34,(((OSAT:1.555,(CPAP:0.5548,(ATHA:0.2774):0.2774):1.0002):0.738,ATRI:2.293):0.6125):0.6125):0.939):0.587):0.0;"

```

next we get the data (we need a model object for that)

```julia
ccd = read_ale(joinpath("example/example-1/ale"), w)
```

```
12-element Array{CCD{UInt16,Float64},1}:
 CCD{UInt16,Float64}(Γ=83, 𝓛=13)
 CCD{UInt16,Float64}(Γ=55, 𝓛=13)
 CCD{UInt16,Float64}(Γ=89, 𝓛=13)
 CCD{UInt16,Float64}(Γ=131, 𝓛=13)
 CCD{UInt16,Float64}(Γ=107, 𝓛=13)
 CCD{UInt16,Float64}(Γ=59, 𝓛=13)
 CCD{UInt16,Float64}(Γ=53, 𝓛=13)
 CCD{UInt16,Float64}(Γ=83, 𝓛=13)
 CCD{UInt16,Float64}(Γ=59, 𝓛=13)
 CCD{UInt16,Float64}(Γ=95, 𝓛=13)
 CCD{UInt16,Float64}(Γ=67, 𝓛=13)
 CCD{UInt16,Float64}(Γ=65, 𝓛=13)
```

Now we define the Turing model

```julia
@model constantrates(model, ccd) = begin
    r  ~ MvLogNormal(ones(2))
    η  ~ Beta(3,1)
    q1 ~ Beta()
    q2 ~ Beta()
    ccd ~ model((λ=r[1], μ=r[2], η=η, q=[q1, q2]))
end

model = constantrates(w, ccd)
chain = sample(model, NUTS(0.65), 100)
pdf = DataFrame(chain)
```

```
50×5 DataFrame
│ Row │ q1         │ q2        │ r[1]      │ r[2]     │ η        │
│     │ Float64    │ Float64   │ Float64   │ Float64  │ Float64  │
├─────┼────────────┼───────────┼───────────┼──────────┼──────────┤
│ 1   │ 0.037111   │ 0.266912  │ 0.128431  │ 0.183775 │ 0.741543 │
│ 2   │ 0.0487045  │ 0.385792  │ 0.129904  │ 0.170465 │ 0.64176  │
│ 3   │ 0.0496272  │ 0.164097  │ 0.144603  │ 0.180296 │ 0.784239 │
│ 4   │ 0.0345101  │ 0.323643  │ 0.122635  │ 0.154142 │ 0.710811 │
│ 5   │ 0.0284716  │ 0.362088  │ 0.109615  │ 0.202335 │ 0.770605 │
│ 6   │ 0.108302   │ 0.1283    │ 0.118246  │ 0.172995 │ 0.743966 │
│ 7   │ 0.059217   │ 0.256584  │ 0.139647  │ 0.219366 │ 0.681216 │
│ 8   │ 0.0363055  │ 0.394341  │ 0.169352  │ 0.198316 │ 0.855772 │
│ 9   │ 0.0192676  │ 0.1184    │ 0.152855  │ 0.176235 │ 0.833965 │
│ 10  │ 0.0366632  │ 0.456505  │ 0.105923  │ 0.147342 │ 0.686557 │
│ 11  │ 0.0162939  │ 0.296075  │ 0.090669  │ 0.128853 │ 0.772351 │
│ 12  │ 0.0964077  │ 0.146934  │ 0.117797  │ 0.146236 │ 0.854984 │
│ 13  │ 0.0709182  │ 0.212994  │ 0.141174  │ 0.19555  │ 0.773449 │
│ 14  │ 0.0664921  │ 0.196798  │ 0.105734  │ 0.159644 │ 0.779898 │
│ 15  │ 0.0698925  │ 0.240373  │ 0.135946  │ 0.177708 │ 0.783805 │
│ 16  │ 0.0821177  │ 0.259253  │ 0.129704  │ 0.138725 │ 0.789484 │
│ 17  │ 0.0149077  │ 0.394679  │ 0.149435  │ 0.191833 │ 0.789952 │
│ 18  │ 0.0365349  │ 0.132897  │ 0.105862  │ 0.168202 │ 0.779643 │
│ 19  │ 0.0381964  │ 0.10196   │ 0.129067  │ 0.150466 │ 0.785236 │
│ 20  │ 0.0409248  │ 0.110145  │ 0.131994  │ 0.146814 │ 0.80246  │
│ 21  │ 0.0296652  │ 0.18325   │ 0.148866  │ 0.166449 │ 0.823314 │
│ 22  │ 0.0277544  │ 0.172357  │ 0.138651  │ 0.172443 │ 0.848504 │
│ 23  │ 0.0593129  │ 0.2854    │ 0.138871  │ 0.197041 │ 0.649383 │
│ 24  │ 0.233831   │ 0.181229  │ 0.118428  │ 0.144591 │ 0.83906  │
│ 25  │ 0.226437   │ 0.167941  │ 0.12146   │ 0.147085 │ 0.868699 │
│ 26  │ 0.123193   │ 0.49048   │ 0.135826  │ 0.143742 │ 0.836659 │
│ 27  │ 0.114718   │ 0.561409  │ 0.127309  │ 0.178098 │ 0.894992 │
│ 28  │ 0.19109    │ 0.297978  │ 0.0930817 │ 0.183433 │ 0.827993 │
│ 29  │ 0.250224   │ 0.374697  │ 0.136148  │ 0.164139 │ 0.799482 │
│ 30  │ 0.0412679  │ 0.232205  │ 0.153507  │ 0.188322 │ 0.572215 │
│ 31  │ 0.043441   │ 0.296557  │ 0.125711  │ 0.151776 │ 0.550242 │
│ 32  │ 0.054329   │ 0.267398  │ 0.154155  │ 0.194713 │ 0.772108 │
│ 33  │ 0.0564164  │ 0.348025  │ 0.120386  │ 0.160935 │ 0.729407 │
│ 34  │ 0.085351   │ 0.245337  │ 0.121627  │ 0.186757 │ 0.812122 │
│ 35  │ 0.100937   │ 0.300133  │ 0.132415  │ 0.152137 │ 0.734229 │
│ 36  │ 0.176164   │ 0.391373  │ 0.151453  │ 0.176427 │ 0.8649   │
│ 37  │ 0.159741   │ 0.265859  │ 0.11495   │ 0.198207 │ 0.580243 │
│ 38  │ 0.0306328  │ 0.22643   │ 0.145496  │ 0.155282 │ 0.898955 │
│ 39  │ 0.0393283  │ 0.183918  │ 0.104192  │ 0.140637 │ 0.923592 │
│ 40  │ 0.124512   │ 0.21717   │ 0.144603  │ 0.207174 │ 0.52091  │
│ 41  │ 0.143652   │ 0.18981   │ 0.141457  │ 0.181508 │ 0.645365 │
│ 42  │ 0.0159798  │ 0.35981   │ 0.130292  │ 0.156823 │ 0.798654 │
│ 43  │ 0.0159798  │ 0.35981   │ 0.130292  │ 0.156823 │ 0.798654 │
│ 44  │ 0.0263202  │ 0.105941  │ 0.130612  │ 0.17267  │ 0.835229 │
│ 45  │ 0.0156378  │ 0.118177  │ 0.16274   │ 0.150978 │ 0.835257 │
│ 46  │ 0.00221182 │ 0.0844978 │ 0.113503  │ 0.180415 │ 0.845886 │
│ 47  │ 0.00379495 │ 0.0606589 │ 0.161163  │ 0.165295 │ 0.866099 │
│ 48  │ 0.0078979  │ 0.0466355 │ 0.155254  │ 0.168107 │ 0.677868 │
│ 49  │ 0.00845649 │ 0.0421507 │ 0.139211  │ 0.185635 │ 0.662196 │
│ 50  │ 0.0137899  │ 0.103724  │ 0.121943  │ 0.167852 │ 0.553981 │
```

We can sample reconciled trees from the posterior using a backtracking algorithm

```julia
fun = (m, x)-> Array(x) |> x->m((λ=x[3], μ=x[4], η=x[5], q=x[1:2]))
tt = TreeTracker(w, ccd[end-1:end], pdf, fun)
trees = track(tt)
```

```
2-element Array{Whale.RecSummary,1}:
 RecSummary(# unique trees = 18)
 RecSummary(# unique trees = 19)
```

## Citation

If you use Whale, please cite:

>[Zwaenepoel, A. and Van de Peer, Y., 2019. Inference of Ancient Whole-Genome Duplications and the Evolution of Gene Duplication and Loss Rates. *Molecular biology and evolution*, 36(7), pp.1384-1404.](https://academic.oup.com/mbe/article-abstract/36/7/1384/5475503)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

