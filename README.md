# Whale: whole genome duplication inference by amalgamated likelihood estimation

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

- This library implements the duplication, loss and whole genome duplication (DL + WGD) model for performing joint gene tree - reconciliation inference using amalgamated likelihood estimation (ALE). By using amalgamation, uncertainty in the gene tree topology is taken into account during reconciliation.

- This method, called Whale, can be used to assess WGD hypotheses using gene family phylogenetic trees. It can also be used to estimate branch-specific duplication and loss rates for a species tree under different models of rate evolution.

- To install `Whale`, you will need a julia installation (v1.x). You should fire up the julia REPL (typically by typing `julia`), once in the julia REPL, you should type `]` and enter `add https://github.com/arzwa/Whale.jl`. It should look somewhat like below:

```
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.0.0 (2018-08-08)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

(v1.0) pkg> add https://github.com/arzwa/Whale.jl
   Cloning git-repo `https://github.com/arzwa/Whale.jl`
  Updating git-repo `https://github.com/arzwa/Whale.jl`
  ...
```

You might want to get some minimal familiarity with the Julia REPL and its package manager when using Whale, see [the julia docs](https://docs.julialang.org/en/v1/).

- To do analyses with Whale, you will need (1) a dated species tree, (2) a set of gene families with for each gene family a sample from the posterior distribution of gene trees (bootstrap replicates can also be used in principle), summarized as a *conditional clade distribution* in an `ale` file ([see below](#aleobserve)) and (3) a configuration file.

- The main program is `whale.jl` in the `bin` folder of this repository (it is not a binary file but a julia script, but following traditions I have put it in a bin folder). All analyses are invoked by using

```
julia -p <n_cores> whale.jl <species tree> <ale directory|file|filelist> <config file>
```

- `julia` can have a rather slow startup time, if you plan to use `Whale` a lot, you may want to open a julia session, load the Whale package and do your analyses in the session. However for the typical rather long analyses performed with Whale, you will probably just submit your job to some cluster.

- Below I explain how to use Whale in a maximum-likelihood and Bayesian framework.

## Testing WGD hypotheses by maximum likelihood

Models with and without WGD can be compared by means of a likelihood ratio test or information criterion (such as AIC or BIC). The method allows to estimate branch-specific rates and use arbitrary rate classes for branches of the species tree. To use Whale with the ML approach you will need a config file like `whalemle.conf` in the `example` directory of this repository. This looks something like this:

```
[wgd]
SEED = GBIL,ATHA 3.9 -1.
ANGI = ATRI,ATHA 3.08 -1.
PPAT = PPAT 0.6 -1.

[rates]
ATHA,CPAP = 1 true
PPAT,MPOL = 2 true
GBIL,PABI = 3 true

[ml]
η = 0.66
```

Where the species tree (`example/morris-9taxa.nw`) looks like this

```
((MPOL:4.752,PPAT:4.752):0.292,(SMOE:4.457,(((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);
```

To specify a WGD hypothesis in the config file, for example the seed plant WGD, you have to put a line like the following in your configuration in the `[wgd]` section:

```
SEED = GBIL,ATHA 3.9 -1.
```

Where `SEED` is the name of the WGD, `GBIL,ATHA` reflects the common ancestor node for the largest clade that shares this WGD (*i.e.* the node that is the common ancestor of `ATHA` and `GBIL` in the species tree). `3.9` is the estimated age of this WGD. `-1` indicates that the retention rate for this WGD should be estimated. Specifying a value between 0 and 1 (boundaries included) will fix this retention rate in the analysis (and not estimate it).

To specify branch-wise rates one can use the `[rates]` section. This looks as follows:

```
[rates]
ATHA,CPAP = 1 true
PPAT,MPOL = 2 true
GBIL,PABI = 3 true
```

Here we have specified a rate class (with ID 1) for the clade defined by the common ancestor of `ATHA` and `CPAP`. Similarly for the mosses (with ID 2) and gymnosperms (ID 3). By using the `false` instead of `true` setting, the rate is not defined for the full clade below the specified node but only for the branch leasing to that node.

Other options for the ML method are specified in the `[ml]` section, here you can parametrize the geometric prior distribution on the number of lineages at the root (η). Other parameters that can be set here are `maxiter` (the maximum number of iterations) and `ctol` (the convergence criterion).

You can test the ML methods with the example data in the `example` directory. From the main directory of the repository, execute:

```
julia -p 1 bin/whale.jl example/morris-9taxa.nw example/example-ale example/whalemle.conf
```

The output should be something like this:

```
<lots of stuff>
...
λ = ( 0.245,  0.093,  0.092,  0.106) ; μ = ( 0.345,  0.086,  0.121,  0.176) ; q = ( 0.174,  0.000,  0.418) ; ⤷ log[P(Γ)] = -280.143
Maximum: log(L) = -280.1426
ML estimates (η = 0.66): λ = ( 0.245,  0.093,  0.092,  0.107) ; μ = ( 0.345,  0.086,  0.121,  0.176) ; q = ( 0.174,  0.000,  0.418) ;
out = Results of Optimization Algorithm
 * Algorithm: Nelder-Mead
 * Starting Point: [0.17712481405234004,0.17712481405234004, ...]
 * Minimizer: [0.24459519525102252,0.09337957168782722, ...]
 * Minimum: 2.801426e+02
 * Iterations: 929
 * Convergence: true
   *  √(Σ(yᵢ-ȳ)²)/n < 1.0e-05: true
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 1333
out.minimizer = [0.244595, 0.0933796, 0.0916083, 0.106514, 0.344514, 0.0863275, 0.120583, 0.175975, 0.173878, 9.11622e-7, 0.41785]
out.minimum = 280.1426206365062
```

## Sampling reconciled trees with ML

As in ALE, reconciled trees can be sampled from the dynamic programming matrix by backtracking. To do so, add a section in the config file looking like:

```
[track]
outfile = whaleml-track
N = 100
```

This will, after optimizing the rates to find the MLEs, sample 100 trees for every family and write them, along with some summaries, to some files prefixed with the `outfile` setting.

## Testing WGD hypotheses and inferring branch-wise duplication and loss rates using MCMC

Whale implements a Bayesian approach using MCMC to do posterior inference for the DL + WGD model with branch-wise rates. To do such an analysis, one needs a configuration file like `example/whalebay.conf`, this looks like:

```
[wgd]
SEED = GBIL,ATHA 3.900 -1.
ANGI = ATRI,ATHA 3.080 -1.
MONO = OSAT      0.910 -1.
ALPH = ATHA      0.501 -1.
CPAP = CPAP      0.275 -1.
BETA = ATHA      0.550 -1.
PPAT = PPAT      0.655 -1.

[mcmc]
# priors
rates = gbm         # one of iid|gbm
p_q = 1. 1.         # beta prior on q
p_λ = 0.15 0.5      # LN prior on λ at the root (gbm) or tree wide mean (iid)
p_μ = 0.15 0.5      # LN prior on μ at the root (gbm) or tree wide mean (iid)
p_ν = 0.10          # rate heterogeneity strength parameter
p_η = 4.0  2.0      # prior on η; single param assumes fixed, two params assumes beta prior

# kernel (if arwalk, no other params should be set)
kernel = arwalk

# chain
outfile = whalebay-gbm.csv  # output file for posterior sampled
ngen = 200                  # number of generations to run the chain (set this to > 5000)
freq = 1                    # chain sample frequency
```

For the `[wgd]` section, please refer to the previous section. The `[mcmc]` section is quite self-explanatory (note the comments indicating the priors etc.). The `rates = gbm` setting will result in the autocorrelation (geometric Brownian motion) prior being used, whereas `iid` will result in the independent and identically distributed rates prior being used. Note that the ν parameter has a different meaning in both models. Currently as `kernel` setting only `arwalk` (adaptive random walk) is supported.

You can test the Bayesian approach using the example data by executing:

```
julia -p 1 bin/whale.jl example/morris-9taxa.nw example/example-ale example/whalebay.conf
```

You should see a file `whalebay-gbm.csv` emerge, which looks like this

```
$ head -n3 whalebay-gbm.csv
,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,q1,q2,q3,q4,q5,q6,q7,nu,eta,prior,lhood
1.0,0.09087,0.08779,0.11993,0.09778,0.09058,0.10693,0.10208,0.10039,0.11356,0.11513,0.10401,0.12012,0.11390,0.09216,0.08940,0.06600,0.07432,0.12849,0.12426,0.07258,0.11909,0.13670,0.14608,0.14116,0.15792,0.16911,0.15653,0.13682,0.13521,0.13825,0.17974,0.12830,0.11509,0.10615,0.39608,0.04957,0.62022,0.86409,0.80105,0.37606,0.17825,0.1,0.8749981040217331,91.51772773843761,-300.74884420188476
2.0,0.09087,0.08779,0.06499,0.09778,0.09058,0.10693,0.10208,0.10039,0.11356,0.11513,0.10401,0.12012,0.11390,0.09216,0.08940,0.06600,0.07432,0.12849,0.12426,0.16929,0.11909,0.13670,0.14608,0.14116,0.15792,0.16911,0.15653,0.13682,0.13521,0.13825,0.17974,0.12830,0.11509,0.10615,0.39608,0.04957,0.60998,0.83352,0.29122,0.45863,0.17825,0.1,0.8749981040217331,94.92630520290867,-301.2617513981243
...
```

## Sampling reconciled trees from the posterior

Reconciled trees can be sampled from the posterior by backtracking similar to what is described above for the ML case. However, to do so, a separate program is provided in `bin/track.jl`. This can be run with the following arguments

```
julia -p <n_cpus> track.jl <sptree> <ale> <sample> <burnin> <N> <config> <trees?>
```

Where `sample` is the `csv` file with the samples from the posterior distribution obtained using MCMC (output from `whale.jl`), `N` is the number of trees per family to sample, `config` is the same config file used in the MCMC analyses. When providing a 7th argument all individual trees will be written to files.

## <a name="aleobserve"></a>Getting the CCD (ale) files

Whale requires as input for each gene family a sample from the posterior distribution of topologies, summarized as a conditional clade distribution (CCD). These can be acquired using the program `ALEobserve` from the [ALE software suite](https://github.com/ssolo/ALE). If you have for each gene family a file with on each line a newick tree, you can run  for example `ALEobserve trees.nw burnin=1000` to get a CCD file (discarding the firt 1000 tree as burn-in in this example). **Note** that the gene IDs should be prefixed with the corresponding species name, separated by an `_` character. For example, the gene `AT2G02000` corresponding to the species with ID `ATHA` in the species tree file should be named `ATHA_AT2G02000` in the gene family trees file.

## Extra

Tools for visualizations of species trees and reconciled trees are available in the `PalmTree.jl` package. Some hints on how to used these can be found in the `scripts/coltree.jl` file. A small python program for generating trace plots and marginal posterior densities as histograms is also provided in that directory.

I am aware that `Whale` is not fantastically documented, and this is mainly because the code is still prone to many changes both in design as well as implementation. Since the user base will probably quite small (actually tiny), it might be more efficient for now to handle unclarities by e-mail or the issues section than by providing in-depth docs. **So do not hesitate to contact me for questions, suggestions etc.**
