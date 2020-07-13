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
│ Row │ q1        │ q2        │ r[1]      │ r[2]     │ η        │
│     │ Float64   │ Float64   │ Float64   │ Float64  │ Float64  │
├─────┼───────────┼───────────┼───────────┼──────────┼──────────┤
│ 1   │ 0.100277  │ 0.284105  │ 0.12907   │ 0.147672 │ 0.765169 │
│ 2   │ 0.108212  │ 0.22993   │ 0.105061  │ 0.149047 │ 0.772577 │
│ 3   │ 0.206229  │ 0.345464  │ 0.143443  │ 0.187377 │ 0.783382 │
│ 4   │ 0.064777  │ 0.117945  │ 0.14356   │ 0.181902 │ 0.857269 │
│ 5   │ 0.058934  │ 0.204204  │ 0.126282  │ 0.164572 │ 0.886573 │
│ 6   │ 0.0761391 │ 0.191361  │ 0.152139  │ 0.163122 │ 0.764077 │
│ 7   │ 0.0694423 │ 0.213063  │ 0.113278  │ 0.151841 │ 0.684594 │
│ 8   │ 0.0889132 │ 0.206994  │ 0.132818  │ 0.160986 │ 0.593598 │
│ 9   │ 0.0575046 │ 0.209551  │ 0.155446  │ 0.162508 │ 0.93028  │
│ 10  │ 0.0618958 │ 0.213864  │ 0.112169  │ 0.11237  │ 0.924229 │
│ 11  │ 0.0634296 │ 0.225819  │ 0.148662  │ 0.198455 │ 0.81209  │
│ 12  │ 0.0708128 │ 0.236567  │ 0.101629  │ 0.136824 │ 0.803178 │
│ 13  │ 0.0673013 │ 0.227788  │ 0.15505   │ 0.175917 │ 0.805425 │
│ 14  │ 0.0659258 │ 0.30626   │ 0.130522  │ 0.190151 │ 0.808786 │
│ 15  │ 0.0529497 │ 0.225639  │ 0.104923  │ 0.137333 │ 0.819551 │
│ 16  │ 0.135732  │ 0.146017  │ 0.119813  │ 0.252224 │ 0.361208 │
│ 17  │ 0.0881414 │ 0.166479  │ 0.104813  │ 0.171338 │ 0.774859 │
│ 18  │ 0.133193  │ 0.288856  │ 0.107738  │ 0.157817 │ 0.765827 │
│ 19  │ 0.168805  │ 0.522257  │ 0.135307  │ 0.149607 │ 0.84944  │
│ 20  │ 0.129183  │ 0.0989785 │ 0.143206  │ 0.160531 │ 0.717663 │
│ 21  │ 0.0990457 │ 0.179706  │ 0.13224   │ 0.166487 │ 0.674925 │
│ 22  │ 0.0906406 │ 0.345367  │ 0.175905  │ 0.192801 │ 0.666833 │
│ 23  │ 0.0686541 │ 0.267893  │ 0.105106  │ 0.17974  │ 0.632831 │
│ 24  │ 0.0966271 │ 0.24619   │ 0.145608  │ 0.162867 │ 0.749856 │
│ 25  │ 0.0930406 │ 0.242743  │ 0.149312  │ 0.128452 │ 0.764399 │
│ 26  │ 0.0151364 │ 0.264001  │ 0.0976852 │ 0.187369 │ 0.76051  │
│ 27  │ 0.0443372 │ 0.120216  │ 0.143362  │ 0.180102 │ 0.740409 │
│ 28  │ 0.0629253 │ 0.231953  │ 0.146927  │ 0.185072 │ 0.668753 │
│ 29  │ 0.0702792 │ 0.312417  │ 0.117542  │ 0.162183 │ 0.851219 │
│ 30  │ 0.0712238 │ 0.361957  │ 0.0994421 │ 0.172583 │ 0.830653 │
│ 31  │ 0.0553604 │ 0.156084  │ 0.16402   │ 0.162616 │ 0.845261 │
│ 32  │ 0.03755   │ 0.133465  │ 0.0967635 │ 0.165625 │ 0.868945 │
│ 33  │ 0.0442987 │ 0.142669  │ 0.105727  │ 0.121014 │ 0.892444 │
│ 34  │ 0.0317389 │ 0.528653  │ 0.108165  │ 0.158105 │ 0.66762  │
│ 35  │ 0.2464    │ 0.0501615 │ 0.11614   │ 0.15303  │ 0.816129 │
│ 36  │ 0.0195864 │ 0.144296  │ 0.158695  │ 0.189069 │ 0.713249 │
│ 37  │ 0.0152353 │ 0.236284  │ 0.116214  │ 0.118211 │ 0.713732 │
│ 38  │ 0.0533099 │ 0.196647  │ 0.0970081 │ 0.134201 │ 0.78135  │
│ 39  │ 0.06583   │ 0.286232  │ 0.165024  │ 0.203639 │ 0.824427 │
│ 40  │ 0.0853702 │ 0.399527  │ 0.132296  │ 0.239106 │ 0.708532 │
│ 41  │ 0.0734494 │ 0.135726  │ 0.177521  │ 0.196913 │ 0.720482 │
│ 42  │ 0.139825  │ 0.281449  │ 0.125757  │ 0.12231  │ 0.903458 │
│ 43  │ 0.101192  │ 0.238875  │ 0.146082  │ 0.231247 │ 0.604779 │
│ 44  │ 0.226591  │ 0.187946  │ 0.112661  │ 0.141693 │ 0.877773 │
│ 45  │ 0.022282  │ 0.25665   │ 0.125003  │ 0.190338 │ 0.601391 │
│ 46  │ 0.0487091 │ 0.432936  │ 0.120919  │ 0.182472 │ 0.797367 │
│ 47  │ 0.0275356 │ 0.270324  │ 0.136278  │ 0.163526 │ 0.899949 │
│ 48  │ 0.037034  │ 0.341376  │ 0.150644  │ 0.148903 │ 0.893745 │
│ 49  │ 0.043364  │ 0.345141  │ 0.120775  │ 0.179828 │ 0.832589 │
│ 50  │ 0.0320341 │ 0.529367  │ 0.10616   │ 0.105327 │ 0.765065 │
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
 RecSummary(# unique trees = 24)
```

## Citation

If you use Whale, please cite:

>[Zwaenepoel, A. and Van de Peer, Y., 2019. Inference of Ancient Whole-Genome Duplications and the Evolution of Gene Duplication and Loss Rates. *Molecular biology and evolution*, 36(7), pp.1384-1404.](https://academic.oup.com/mbe/article-abstract/36/7/1384/5475503)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

