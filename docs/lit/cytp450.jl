
using Whale, DynamicHMC, DynamicHMC.Diagnostics, DistributedArrays, Random
using Plots, StatsPlots

# This is a use case I haven't been exploring before, namely large gene
# families. Here we consider a family of about 100 leaves.

base  = joinpath(@__DIR__, "../../example/example-4")
model = WhaleModel(readline(joinpath(base, "tree.nw")), Δt=0.1)
data  = read_ale(joinpath(base, "cytp450.ale"), model)

# Reading in the single CCD is already a very heavy operation. The CCD has about
# 5000 unique clades.
prior = Whale.CRPrior()
problem = WhaleProblem(model, distribute([data]), prior)

# NUTS!
results   = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 1000)
posterior = Whale.transform.(problem.trans, results.chain)
@info summarize_tree_statistics(results.tree_statistics)

# Save the posterior
using CSV
df = Whale.unpack(posterior)
CSV.write(joinpath(base, "cr-post.csv"))

# Make some trace plots
df = Whale.unpack(posterior)
plot(df.λ, label="\\lambda"); plot!(df.μ, label="\\mu")

# Sample reconciled trees
rectrees = sumtrees(problem, posterior)

# Plot the MAP tree
using PalmTree, Parameters, Luxor

rectree = rectrees[1].trees[10].tree
begin
    @unpack root, annot = rectree
    tl = TreeLayout(root, dim=(600, 1200))
    PalmTree.cladogram!(tl)
    colfun = (n)->annot[n].label != "loss" ? RGB() : RGB(0.99,0.99,0.99)
    labfun = (k, p)->settext(" $(split(annot[k].name, "_")[1])", p, valign="center")
    credfn = (k, p)->settext(k ∉ tl.leaves ? " $(round(annot[k].cred, digits=2))" : "", p, valign="center")
    dupfn  = (k, p)->annot[k].label == "duplication" ? box(p, 7, 7, :fill) : nothing
    @svg begin
        Luxor.origin(Point(-20,20))
        setfont("Noto sans italic", 11)
        drawtree(tl, color=colfun)
        nodemap(tl, labfun)
        nodemap(tl, credfn)
        nodemap(tl, dupfn)
    end 650 1250 "docs/src/assets/cytp450-10.svg"
end

# ![](assets/cytp450-1.svg)
