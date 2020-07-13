
# Introduction

!!! warning
    The latest Whale version is a thorough rewrite of the Whale library, and is still work in progress. For the version as used in Zwaenepoel & Van de Peer (2019), refer to [this release (v0.2)](https://github.com/arzwa/Whale.jl/releases/tag/v0.2). Nevertheless, the current version should be safe to use and is more efficient and convenient (if you know a bit of julia).

Whale provides tools for (genome-wide) **amalgamated likelihood estimation (ALE) under a DL+WGD model**, which is an approach to infer reconciled gene trees and parameters of a model of gene family evolution given a known species tree.

The ALE approach takes into account uncertainty in the gene tree topology by marginalizing over all tree topologies that can be amalgamated from the so-called *conditional clade distribution* (CCD). This CCD can be constructed from a sample of the posterior distribution of tree topologies (which can be obtained using any standard software for Bayesian phylogenetics).

More specifically, this library can be used to

- Statistically test the absence or presence of hypothetical whole-genome duplication (WGD) events in a species phylogeny
- Infer lineage-specific gene duplication and loss rates for a species phylogeny
- Infer high-quality (reconciled) gene trees given a known species tree [cf. Szöllősi *et al.*](https://academic.oup.com/sysbio/article/64/1/e42/1634124)
- All of the above at once

!!! note
    This library implements the DL and DL+WGD models. It does not implement models of gene family evolution that take into account horizontal gene transfer, incomplete lineage sorting or gene conversion.

## Installation

You will need julia-1.x. Fire up a julia REPL by typing `julia` at the command line enter the package manager interface by typing `]`, and execute `add Whale`.

## Data preparation

To perform analyses with Whale, you will need  

1. An **ultrametric species tree**, with ideally branch lengths in geological time (since this allows straightforward interpretation of parameter estimates.)
2. A bunch of ALE files, which summarize the **conditional clade distributions** (CCDs) for the same bunch of gene families. These can be obtained from a sample of the posterior distribution of gene trees using the [`ALEobserve`](https://github.com/ssolo/ALE) tool. A pipeline to obtain these from a set of gene family protein fasta files is available at [github](https://github.com/arzwa/whaleprep).

!!! note
    Gene IDs should be prefixed by the name of the species to which the gene belongs as used in the species tree. For example if *Arabidopsis thaliana* is represented by `ATHA` in the species tree newick file, then the genes should be prefixed with `ATHA_`, e.g. `ATHA_AT1G05000`.

!!! note
    Analyzing CCDs (ALE files) with a very large number of clades or for very large families can be prohibitive computationally. It is therefore generally advisable that large orthogroups are filtered out based on some criterion (for example using the script `orthofilter.py` in the scripts directory of the Whale repository). To filter out families with very large numbers of clades in the CCD (which reflects that there is a lot of uncertainty in the gene tree), the scripts `ccddata.py` and `ccdfilter.py` can be used. This is a rather *ad hoc* filtering procedure, but can be useful to filter out families that trouble the analysis.

!!! warning
    Most analyses in Whale assume that for each family, **there is at least one gene in both clades stemming from the root of the species tree**. The likelihood in Whale is the conditional likelihood under this assumption. This is to rule out the possibility of *de novo* gain of a gene family along a branch of the species tree. The orthogroup data should therefore always be filtered to be in accordance with this criterion. This can also be done using the `orthofilter.py` script.


## References

`Whale.jl` is developed by Arthur Zwaenepoel at the VIB-UGent center for plant
systems biology (bioinformatics & evolutionary genomics group). If you use
Whale, please cite:

>Zwaenepoel, A. and Van de Peer, Y., 2019. Inference of Ancient Whole-Genome Duplications and the Evolution of Gene Duplication and Loss Rates. *Molecular biology and evolution*, 36(7), pp.1384-1404.

The methods in Whale are heavily inspired by previous work done by other researchers. If you use Whale, consider citing the following two particularly important studies:

>[ALE] Szöllősi, G.J., Rosikiewicz, W., Boussau, B., Tannier, E. and Daubin, V., 2013. Efficient exploration of the space of reconciled gene trees. *Systematic biology*, 62(6), pp.901-912.

>[DL+WGD model] Rabier, C.E., Ta, T. and Ané, C., 2013. Detecting and locating whole genome duplications on a phylogeny: a probabilistic approach. *Molecular biology and evolution*, 31(3), pp.750-762.
