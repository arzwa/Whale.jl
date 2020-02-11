
# Bayesian inference of reconciled gene trees

We first do inference using NUTS with a constant-rates model

```@example rectree
using DynamicHMC, Whale, DistributedArrays, Distributions, Random
```

set-up

```@example rectree
wm      = WhaleModel(Whale.extree, Δt=0.1)
ccd     = distribute(read_ale(joinpath(@__DIR__, "../../../example/example-ale"), wm))
prior   = CRPrior(MvNormal(ones(2)), Beta(3,1), Beta())
problem = WhaleProblem(wm, ccd, prior)
```

MCMC

```@example rectree
progress  = NoProgressReport()
results   = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100, reporter=progress)
posterior = transform.(problem.trans, results.chain)
```

Note, now one should do some routine MCMC diagnostics, ensuring there
are no convergence issues etc. Also typically one would do a lot more
iterations. Furthermore, I would not generally recommend to turn of the
progress reporter (this is just turned off for rendering this file).

Sample a tree (latent state) for each posterior sample

```@example rectree
trees    = backtrack(problem, posterior)
rectrees = sumtrees(trees, ccd, wm)
rectrees[1]  # have a look at the first family
```

This shows all trees observed in the sample, ranked by their observed
frequency.

Now we'll make a plot of the most probable tree for the first gene family

```@example rectree
using PalmTree, Parameters, Luxor
import Luxor: RGB

rectree = rectrees[1][1].tree
begin
    @unpack root, annot = rectree
    tl = TreeLayout(root)
    PalmTree.cladogram!(tl)

    colfun = (n)->annot[n].label != "loss" ? RGB() : RGB(0.99,0.99,0.99)
    labfun = (k, p)->settext(" $(annot[k].name)", p, valign="center")
    credfn = (k, p)->settext(k ∉ tl.leaves ?
        " $(annot[k].cred)" : "", p, valign="center")
    @svg begin
        origin(Point(10,10))
        setfont("Noto sans italic", 11)
        drawtree(tl, color=colfun)
        nodemap(tl, labfun)
        nodemap(tl, credfn)
    end 500 350
end
```

![](../../assets/cr-rectree.svg)

Here each split is annotated by its observed frequency in the set of
reconciled trees sampled from the posterior. Note that this is the marginal
frequency of the split (as is customary in Bayesian phylogenetics), not of
the subtree.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

