
# Bayesian inference of reconciled gene trees - Constant rates

We will do inference using NUTS with a constant-rates model. We'll need the
following modules loaded:

```@example rectree
using DynamicHMC, Whale, DistributedArrays, Distributions, Random
using DynamicHMC.Diagnostics
```

We'll use the example data that can be found in the git-repository of Whale,
The associated species tree is already in the Whale module (`extree`)

```@example rectree
wm  = WhaleModel(Whale.extree, Δt=0.1)
addwgd!(wm, wm[6], wm[6].event.t*0.5, rand())
ccd = read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), wm)
```

The data are a bunch of CCDs (conditional clade distributions) obtained using
MrBayes + ALEobserve on a bunch of amino-acid alignments of protein-coding
genes. Note that if julia is running on multiple processes, the `distribute`
call will enable calculation of the likelihood and gradients in parallel.

!!! note
    For a small data set like this, using multiple processes will probably not
    result in a speed-up. It seems that 1 CPU core per 100 gene families is a
    reasonable rule of thumb for a typical 10-taxon species tree.

Now we specify the prior and bundle together prior, model and data into a
`WhaleProblem` object

```@example rectree
prior   = CRPrior(πr=MvNormal(ones(2)), πη=Beta(3,1))
problem = WhaleProblem(wm, ccd, prior)
```

Note our prior choices: we use a bivariate normal prior for the logarithm
of the duplication and loss rate with identity covariance matrix and a
Beta(3,1) prior for the geometric prior on the number of lineages at the root.

!!! note
    We always consider molecular evolutionary rates on a logarithmic scale.
    This is convenient, since virtually all clock models are defined on the
    real line. Furthermore, we see no strong reason why the natural scale
    should be favored anyway for these kinds of parameters.

Now we start the MCMC using the NUTS (No U-turn sampler) implementation in
the wonderful `DynamicHMC` module:
progress  = NoProgressReport()

```@example rectree
results   = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100)
posterior = Whale.transform.(problem.trans, results.chain)
@info summarize_tree_statistics(results.tree_statistics)
```

!!! note
    Now one should do some routine MCMC diagnostics, ensuring there
    are no convergence issues etc. Also typically one would do a lot more
    iterations. Furthermore, I would not generally recommend to turn of the
    progress reporter (this is just turned off for rendering this file).

Whale does not sample reconciled trees during the MCMC, but integrates over
them. We can however sample trees from the acquired posterior using a
stochastic backtracking algorithm. The following will sample exactly one
reconciled tree for each sample of the posterior. The resulting sample of
trees should be an approimate sample from the posterior distribution

```@example rectree
trees    = backtrack(problem, posterior)
rectrees = sumtrees(trees, ccd, wm)
rectrees[1]  # have a look at the first family
```

This shows all reconciled trees observed in the sample, ranked by their
observed frequency (approximate posterior probability).

!!! note
    Not all trees for the same family have an equal number of nodes, since they
    can have distinct numbers of loss nodes.

Now we'll make a plot of the most probable tree for the first gene family.

```@example rectree
using PalmTree, Parameters, Luxor
import Luxor: RGB

rectree = rectrees[1].trees[1].tree
begin
    @unpack root, annot = rectree
    tl = TreeLayout(root)
    PalmTree.cladogram!(tl)

    colfun = (n)->annot[n].label != "loss" ? RGB() : RGB(0.99,0.99,0.99)
    labfun = (k, p)->settext(" $(split(annot[k].name, "_")[1])", p, valign="center")
    credfn = (k, p)->settext(k ∉ tl.leaves ?
        " $(annot[k].cred)" : "", p, valign="center")
    @svg begin
        Luxor.origin(Point(-20,20))
        setfont("Noto sans italic", 11)
        drawtree(tl, color=colfun)
        nodemap(tl, labfun)
        nodemap(tl, credfn)
    end 450 350 #"../assets/cr-rectree.svg"
end
```

![](../assets/cr-rectree.svg)

Here each split is annotated by its observed frequency in the set of
reconciled trees sampled from the posterior. Note that this is the marginal
frequency of the split (as is customary in Bayesian phylogenetics), not of
the subtree.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

