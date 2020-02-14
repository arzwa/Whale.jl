# # Bayesian inference of branch-wise duplication and loss rates

# Here an example using a branch-wise rates model is shown. This is
# computationally much more challenging than the constant-rates model with its
# small number of parameters. In general, you would run this on a computing
# cluster, using about 1 CPU core per 100 families.

using DynamicHMC, Whale, DistributedArrays, Distributions, Random

# We'll use the example data that can be found in the git-repository of Whale,
# The associated species tree is already in the Whale module (`extree`)
wm  = WhaleModel(Whale.extree, Δt=0.1)
ccd = DArray(read_ale(joinpath(@__DIR__, "../../../example/example-ale"), wm)[1:2])
ts  = Whale.branchlengths(wm)

# Now we specify the prior and bundle together prior, model and data into a
# `WhaleProblem` object
prior = IRPrior(
    Ψ=[1. 0.; 0. 1.],
    πr=MvNormal(ones(2)),
    πη=Beta(3,1),
    πE=(Normal(1., 0.2), ts))

# We specify a prior on the expected number of lineages at the end of each
# branch per ancestral lineage at the beginning of the branch (`πE`). This
# constrains the duplication and loss rate to be similar.

problem = WhaleProblem(wm, ccd, prior)

# MCMC is *much* more computationally demanding for the branch-wise rates model.

warmup    = DynamicHMC.default_warmup_stages(doubling_stages=3)
results   = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100, warmup_stages=warmup)
posterior = transform.(problem.trans, results.chain)

# Now we should do diagnostics etc. but note that if we were doing the analysis
# 'for real', we should run much longer chains to enable better inferences, and
# if possible run multiple chains to assess convergence (or rather assess
# convergence issues).

using UnicodePlots
λ8 = [x.r[1,8] for x in posterior]
μ8 = [x.r[2,8] for x in posterior]
p = lineplot(λ8)
lineplot!(p, μ8)

# We have induced a strong correlation between duplication and loss rates for
# a given branch by using the `πE` prior, and this has a lot of influence on
# the posterior as we are only looking at a very limited amount of data

scatterplot(λ8, μ8)

# Get the posterior means for the duplication rates
λ = mean([x.r[1,i] for x in posterior, i=1:17], dims=1)

normfun(vec) = (x)->(x - minimum(vec))/-(-(extrema(vec)...))
f = normfun(λ)

using PalmTree, Parameters, Luxor, NewickTree, ColorSchemes
import Luxor: RGB
begin
    NewickTree.isleaf(i::UInt16) = isleaf(wm[i])
    NewickTree.distance(i::UInt16) = wm[i].event.t
    NewickTree.id(i::UInt16) = i
    NewickTree.children(i::UInt16) = wm[i].children
    tl = TreeLayout(wm[1], dim=(150,130))
    colfun = (n)->get(ColorSchemes.viridis, f(λ[n]))
    labfun = (k, p)->haskey(wm.leaves, k) ?
        settext(" $(wm.leaves[k])", p, valign="center") : nothing
    @svg begin
        setline(3)
        Luxor.origin(Point(20,20))
        setfont("Noto sans italic", 11)
        drawtree(tl, color=colfun)
        nodemap(tl, labfun)
    end 220 180 # "../assets/coltree1.svg"
end

# !!! note
#     The NewickTree overloads are really hideous and are necessary because the
#     nodes in the `WhaleModel` do not store direct links to their children...
#     THis should change in the future.

# ![](../assets/coltree1.svg)

# Samples for reconciled trees can be obtained using the stochastic backtracking
# functions.
trees    = backtrack(problem, posterior)
rectrees = sumtrees(trees, ccd, wm)
rectrees[1]  # have a look at the first family

rtrees = [rectrees[1][1].tree, rectrees[1][2].tree]
begin
    @svg begin
        for (rectree, origin) in zip(rtrees, [(5,10), (250,10)])
            @unpack root, annot = rectree
            tl = TreeLayout(root, dim=(230,200))
            PalmTree.cladogram!(tl)
            colfun = (n)->annot[n].label != "loss" ? RGB() : RGB(0.99,0.99,0.99)
            labfun = (k, p)->settext(" $(split(annot[k].name, "_")[1])", p, valign="center")
            credfn = (k, p)->settext(k ∉ tl.leaves ?
                " $(annot[k].cred)" : "", p, valign="center")

            Luxor.origin(Point(origin...))
            setfont("Noto sans italic", 10)
            drawtree(tl, color=colfun)
            nodemap(tl, labfun)
            nodemap(tl, credfn)
        end
    end 550 230 #"../assets/ir-rectree.svg"
end

# ![](../assets/ir-rectree.svg)
