# # Bayesian inference of branch-wise duplication and loss distances

# Here we'll use the methods and models implemented in Whale on a tree that is
# not time calibrated. We set all branch-lengths to one, so that we estimate a
# kind of duplication and loss distance.

using DynamicHMC, Whale, DistributedArrays, Distributions, Random

# We'll use the example data that can be found in the git-repository of Whale,
# The associated species tree is already in the Whale module (`extree`)
tree = "((MPOL:1,PPAT:1):1,(SMOE:1,(((OSAT:1,(ATHA:1,CPAP:1):1):1,ATRI:1):1,(GBIL:1,PABI:1):1):1):1);"
wm  = WhaleModel(tree, Δt=0.1)
ccd = DArray(read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), wm)[1:10])

# Now we specify the prior and bundle together prior, model and data into a
# `WhaleProblem` object
prior = IRPrior(
    Ψ=[1. 0.; 0. 1.],
    πr=MvNormal(ones(2)),
    πη=Normal(0.65, 0.))

# Here we fix the η parameter
prior = Fixedη(prior, wm)

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

using PalmTree, Parameters, Luxor, NewickTree
import Luxor: RGB
begin
    NewickTree.isleaf(i::UInt16) = isleaf(wm[i])
    NewickTree.distance(i::UInt16) = λ[i]
    NewickTree.id(i::UInt16) = i
    NewickTree.children(i::UInt16) = wm[i].children
    tl = TreeLayout(wm[1], dim=(320,230))
    labfun = (k, p)->haskey(wm.leaves, k) ?
        settext(" $(wm.leaves[k])", p, valign="center") : nothing
    @svg begin
        setline(3)
        Luxor.origin(Point(20,20))
        setfont("Noto sans italic", 13)
        drawtree(tl)
        nodemap(tl, labfun)
    end 380 260 # "../assets/coltree1.svg"
end
