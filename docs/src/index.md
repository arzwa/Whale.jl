
# Introduction

Whale provides tools for genome-wide amalgamated likelihood estimation (ALE) under a DL+WGD model, which is an approach to infer reconciled gene trees and parameters of a model of gene family evolution given a known species tree. The  ALE approach takes into account uncertainty in the gene tree topology by marginalizing over all tree topologies that can be amalgamated from the so-called *conditional clade distribution* (CCD). This CCD can be constructed from a sample of the posterior distribution of tree topologies (which can be obtained using any standard software for Bayesian phylogenetics).

More specifically, this library can be used to

- Statistically test the absence or presence of hypothetical whole-genome duplication (WGD) events in a species phylogeny
- Infer lineage-specific gene duplication and loss rates for a species phylogeny
- Infer high-quality (reconciled) gene trees given a known species tree [cf. Szöllősi *et al.*](https://academic.oup.com/sysbio/article/64/1/e42/1634124)
- All of the above at once

!!! note
    This library implements the DL and DL+WGD models. It does not implement models of gene family evolution that take into account horizontal gene transfer, incomplete lineage sorting or gene conversion.

## Installation

You will need julia-1.x. Fire up a julia REPL by typing `julia` at the command line enter the package manager interface by typing `]`, and run the command

```julia-repl
(v1.1) pkg> add https://github.com/arzwa/Whale.jl
```

## Data preparation

To perform analyses with Whale, you will need  

1. An **ultrametric species tree**, with ideally branch lengths in geological time (since this allows straightforward interpretation of parameter estimates.)
2. A bunch of ALE files, which summarize the **conditional clade distributions** (CCDs) for the same bunch of gene families. These can be obtained from a sample of the posterior distribution of gene trees using the [`ALEobserve`](https://github.com/ssolo/ALE) tool. A pipeline to obtain these from a set of gene family protein fasta files is available at [github](https://github.com/arzwa/whaleprep).

## Quick start

If you're not familiar with `julia`, and you simply want to run analyses as performed for instance in [Zwaenepoel & Van de Peer (2019)](https://academic.oup.com/mbe/advance-article/doi/10.1093/molbev/msz088/5475503) the following scripts will be helpful. If you want to get a more detailed view of the library, please consult the [Manual](@ref).

### Maximum likelihood estimation

I assume that your tree is called `tree_file.nw` and your `.ale` files are in a directory `ccd_dir`. For maximum likelihood estimation with a constant rates model, the following script should work

```julia
using Whale

# data and config
st = SlicedTree("tree_file.nw")
ccd = read_ale("ccd_dir", st)
constant = true  # set to false for branch-wise rates

# inference
constant ? set_constantrates!(st) : nothing
w = WhaleModel(st)
out = mle(w, ccd)
```

Save this script to a file (say `whale-mle.jl`). To run the inference using 16 processors for instance, run `julia -p 16 whale-mle.jl`.

To add WGDs, a `wgd_conf` arguament should be provide to `SLicedTree`, please see the docs for [`SlicedTree`](@ref). See the manual section on [Rate indices](@ref) on how to specify local-clock models.

!!! warning
    Depending on the time scale and data set at hand, you may need to tweak the initial values of the `WhaleModel` to prevent Numerical issues!

!!! warning
    ML estimation with branch-wise rates may result in poor convergence for large species trees.

### Bayesian inference with MCMC

A similar script for Bayesian inference using MCMC looks like

```julia
using Whale

# data and config
st = SlicedTree("tree_file.nw")
ccd = read_ale("ccd_dir", st)
model = IRModel(st)      # independent log-normal rates
# model = GBMModel(st)   # autocorrelated rates
n = 11000

w = WhaleChain(st, model)
chain = mcmc!(w, ccd, n, show_every=10)
```

again, saving this as `whale-bay.jl` and running this with `julia -p <nCPU> whale-bay.jl` will start the MCMC with the likelihood evaluation performed on `nCPU` cores.

!!! warning
    Please take you time to understand the hierarchical model used in Whale and to modify the prior distributions to your data set! See the [Bayesian inference](@ref) section of the manual.
