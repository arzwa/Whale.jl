[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://arzwa.github.io/Whale.jl/dev/index.html)

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
1,0,4.752,0.099,48,"MPOL;"
2,0,4.752,0.099,48,"PPAT;"
3,0,4.457,0.099,45,"SMOE;"
4,0,3.178,0.0993,32,"GBIL;"
5,0,3.178,0.0993,32,"PABI;"
6,0,1.555,0.0972,16,"OSAT;"
7,0,0.5548,0.0925,6,"CPAP;"
8,0,0.2774,0.0555,5,"ATHA;"
9,0,2.293,0.0997,23,"ATRI;"
10,0,0.292,0.0584,5,"(MPOL,PPAT);"
11,0,0.34,0.068,5,"(GBIL,PABI);"
18,2,0.2774,0.0555,5,"(ATHA);"
12,0,1.0002,0.0909,11,"(CPAP,(ATHA));"
13,0,0.738,0.0922,8,"(OSAT,(CPAP,(ATHA)));"
14,0,0.6125,0.0875,7,"((OSAT,(CPAP,(ATHA))),ATRI);"
19,1,0.6125,0.0875,7,"(((OSAT,(CPAP,(ATHA))),ATRI));"
15,0,0.939,0.0939,10,"((GBIL,PABI),(((OSAT,(CPAP,(ATHA))),ATRI)));"
16,0,0.587,0.0978,6,"(SMOE,((GBIL,PABI),(((OSAT,(CPAP,(ATHA))),ATRI))));"
17,0,0.0,0.0,0,"((MPOL,PPAT),(SMOE,((GBIL,PABI),(((OSAT,(CPAP,(ATHA))),ATRI)))));"

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
│ 1   │ 0.286457   │ 0.445689  │ 0.119593  │ 0.138369 │ 0.961406 │
│ 2   │ 0.193428   │ 0.373293  │ 0.167729  │ 0.210796 │ 0.958114 │
│ 3   │ 0.195699   │ 0.320645  │ 0.141413  │ 0.184801 │ 0.950913 │
│ 4   │ 0.0923383  │ 0.213735  │ 0.114784  │ 0.148328 │ 0.748796 │
│ 5   │ 0.0692223  │ 0.180414  │ 0.131483  │ 0.209147 │ 0.814968 │
│ 6   │ 0.0299186  │ 0.289585  │ 0.164709  │ 0.201157 │ 0.641582 │
│ 7   │ 0.183984   │ 0.310952  │ 0.114379  │ 0.164185 │ 0.676394 │
│ 8   │ 0.174934   │ 0.330243  │ 0.154343  │ 0.176929 │ 0.673447 │
│ 9   │ 0.0313897  │ 0.111997  │ 0.163492  │ 0.162126 │ 0.800039 │
│ 10  │ 0.0191264  │ 0.23006   │ 0.121172  │ 0.215723 │ 0.573771 │
│ 11  │ 0.269618   │ 0.317459  │ 0.135594  │ 0.145611 │ 0.903131 │
│ 12  │ 0.351506   │ 0.324884  │ 0.153936  │ 0.181585 │ 0.831297 │
│ 13  │ 0.39337    │ 0.26729   │ 0.122826  │ 0.160032 │ 0.840434 │
│ 14  │ 0.00476708 │ 0.183383  │ 0.150592  │ 0.164572 │ 0.75218  │
│ 15  │ 0.00143002 │ 0.398754  │ 0.119728  │ 0.159892 │ 0.692719 │
│ 16  │ 0.00389907 │ 0.148502  │ 0.124537  │ 0.149829 │ 0.87503  │
│ 17  │ 0.1352     │ 0.239213  │ 0.116472  │ 0.203525 │ 0.651876 │
│ 18  │ 0.160597   │ 0.238104  │ 0.135683  │ 0.125729 │ 0.85198  │
│ 19  │ 0.152838   │ 0.19274   │ 0.129226  │ 0.211959 │ 0.805364 │
│ 20  │ 0.142887   │ 0.179344  │ 0.169125  │ 0.175171 │ 0.819339 │
│ 21  │ 0.138914   │ 0.0715515 │ 0.114672  │ 0.123943 │ 0.770764 │
│ 22  │ 0.1446     │ 0.403111  │ 0.164064  │ 0.199502 │ 0.762231 │
│ 23  │ 0.0745043  │ 0.0303171 │ 0.150231  │ 0.230427 │ 0.71867  │
│ 24  │ 0.0715199  │ 0.0310317 │ 0.103449  │ 0.14885  │ 0.745225 │
│ 25  │ 0.108055   │ 0.0812994 │ 0.159185  │ 0.147697 │ 0.788887 │
│ 26  │ 0.139674   │ 0.141541  │ 0.123652  │ 0.18825  │ 0.769726 │
│ 27  │ 0.128854   │ 0.413309  │ 0.141574  │ 0.177986 │ 0.630874 │
│ 28  │ 0.0315412  │ 0.144975  │ 0.132762  │ 0.148908 │ 0.911938 │
│ 29  │ 0.221379   │ 0.431095  │ 0.150141  │ 0.122182 │ 0.749804 │
│ 30  │ 0.058651   │ 0.501272  │ 0.105984  │ 0.201276 │ 0.751209 │
│ 31  │ 0.0345683  │ 0.14635   │ 0.12151   │ 0.243421 │ 0.704878 │
│ 32  │ 0.0752702  │ 0.146775  │ 0.0809531 │ 0.143457 │ 0.708758 │
│ 33  │ 0.083548   │ 0.151001  │ 0.124222  │ 0.131661 │ 0.722953 │
│ 34  │ 0.03438    │ 0.111451  │ 0.157313  │ 0.13782  │ 0.938985 │
│ 35  │ 0.0457862  │ 0.134558  │ 0.132016  │ 0.208994 │ 0.920305 │
│ 36  │ 0.0425592  │ 0.15593   │ 0.151072  │ 0.16287  │ 0.900561 │
│ 37  │ 0.0240745  │ 0.266235  │ 0.121898  │ 0.175013 │ 0.708972 │
│ 38  │ 0.119974   │ 0.206261  │ 0.146347  │ 0.153309 │ 0.744075 │
│ 39  │ 0.111341   │ 0.314127  │ 0.12233   │ 0.188446 │ 0.782176 │
│ 40  │ 0.0216087  │ 0.264857  │ 0.147608  │ 0.171625 │ 0.653596 │
│ 41  │ 0.0599997  │ 0.273398  │ 0.112378  │ 0.14936  │ 0.882294 │
│ 42  │ 0.0122276  │ 0.256153  │ 0.127648  │ 0.137198 │ 0.842132 │
│ 43  │ 0.0985152  │ 0.0397922 │ 0.122797  │ 0.147081 │ 0.723773 │
│ 44  │ 0.117065   │ 0.0294052 │ 0.159919  │ 0.164727 │ 0.732264 │
│ 45  │ 0.0211677  │ 0.079414  │ 0.139723  │ 0.200216 │ 0.815374 │
│ 46  │ 0.0063972  │ 0.139261  │ 0.133784  │ 0.143644 │ 0.807463 │
│ 47  │ 0.00813118 │ 0.10977   │ 0.137429  │ 0.17592  │ 0.760957 │
│ 48  │ 0.0131041  │ 0.0974858 │ 0.111571  │ 0.142711 │ 0.767873 │
│ 49  │ 0.0661102  │ 0.406109  │ 0.128784  │ 0.171269 │ 0.730852 │
│ 50  │ 0.00523229 │ 0.133181  │ 0.13185   │ 0.16108  │ 0.592545 │
```

We can sample reconciled trees from the posterior using a backtracking algorithm

```julia
fun = (m, x)-> Array(x) |> x->m((λ=x[3], μ=x[4], η=x[5], q=x[1:2]))
tt = TreeTracker(w, ccd[end-1:end], pdf, fun)
trees = track(tt)
```

```
2-element Array{Whale.RecSummary,1}:
 RecSummary(# unique trees = 15)
 RecSummary(# unique trees = 21)
```

## Citation

If you use Whale, please cite:

>[Zwaenepoel, A. and Van de Peer, Y., 2019. Inference of Ancient Whole-Genome Duplications and the Evolution of Gene Duplication and Loss Rates. *Molecular biology and evolution*, 36(7), pp.1384-1404.](https://academic.oup.com/mbe/article-abstract/36/7/1384/5475503)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

