# # Posterior analytics (not by Aristotle) example and MAPS D2
using CSV, Plots, DataFrames, Parameters, Whale, StatsPlots

# ## Whale posterior for MAPS D2 from the 1KP study
# Load the file and parse it
post = CSV.read(joinpath(@__DIR__, "../../example/example-3/hmc-D2.mltrees.csv"))
Whale.parsepost!(post)
dfml = Whale.unpack(post);

# `parsepost!` will parse the dumped posterior into the correct julia data
# structures.

# A function to easily collect trace plots
traces(df; kwargs...) = [plot(x, title=col; kwargs...)
    for (col, x) in eachcol(df, true)]

# ... and let's show some trace plots using it:
ps = traces(dfml, grid=false, legend=false, xticks=false, yticks=false,
    color=:black, linewidth=0.2, title_loc=:left, titlefont=7)
plot(ps..., size=(700,600))

# The same for marginal distributions (we could define a plot recipe to combine
# these with the trace plots)
marginalhists(df; kwargs...) = [stephist(x, title=col; kwargs...)
    for (col, x) in eachcol(df, true)]

ps = marginalhists(dfml, grid=false, legend=false, yticks=false,
    color=:black, linewidth=1, title_loc=:left, titlefont=7, alpha=0.1, fill=true)
plot(ps..., size=(700,600))

# For this example, which is a reanalysis of a gymnosperm data set from the 1KP
# paper (using ML trees, so no gene tree uncertainty is taken into account), we
# clearly see a consistent inflation of the loss rate relative to the duplication
# rate. This is rather suspicious, since, if we would take this at face value, it
# would entail persistent genome contraction, which is no known feature of
# gymnosperm evolution. It seems the birth-death process is wandering of in rather
# unlikely regions of parameter space.

# ## Using MrBayes tree samples (accounting for tree uncertainty)

post = CSV.read(joinpath(@__DIR__, "../../example/example-3/hmc-D2.mbtrees.csv"))
Whale.parsepost!(post)
dfmb = Whale.unpack(post);
ps = traces(dfmb, grid=false, legend=false, xticks=false, yticks=false,
    color=:black, linewidth=0.2, title_loc=:left, titlefont=7)
plot(ps..., size=(700,600))

# We'll compare the marginal posterior distributions between the Whale results
# based on ML trees and those based on CCDs.
Base.startswith(s::Symbol, prefix::String) = startswith(string(s), prefix)
ps = []
for (col, x) in eachcol(dfml, true)
    title =
        startswith(col, "r1") ? "\\lambda$(split(string(col), "_")[2])" :
        startswith(col, "r2") ? "\\mu$(split(string(col), "_")[2])" :
        col == :η ? "\\eta" : col
    p = stephist(x, grid=false, legend=false, yticks=false, title=title,
        color=:black, linewidth=1, title_loc=:left, titlefont=7, normalize=true)
    stephist!(p, dfmb[:,col], color=:blue, fill=true, alpha=0.3, normalize=true)
    startswith(col, "q") ? xlims!(0, 1) : nothing
    push!(ps, p)
end
plot(ps..., size=(800,600))

# This is very interesting, it seems taking into account uncertainty in gene
# trees has a huge effect... There seems to be one branch (3, one of the branches
# emanating from the root) where we get higher loss rates than in the ML trees case,
# but for all other branches we find more reasonable estimates for the CCD
# based analysis. Interestingly, also the support for WGD hypotheses is very
# sensitive to the different data sets, with the Gymnosperm WGD going from extreme
# support to q ≈ 0.

# !!! note
#     This is of course not suitable as a proxy for MAPS, as MAPS uses
#     a BSV-based cut-off and some taxon-occupancy thresholds as well... **Turns
#     out in the 1KP analyses no BSV threshold was used?**

# !!! note
#     It would be best to validate (using ML for instance) the new implementation
#     on trees with uncertainty as well (some random swaps etc?) before
#     we go on with this.

using StatsPlots

ps = [plot(), plot(), plot(), plot()]
for (p, df) in zip(ps[1:2:4], [dfml, dfmb])
    for (col, x) in eachcol(df, true)
        !startswith(col, "r") ? continue : nothing
        color = startswith(col, "r1") ? :black : :salmon
        density!(p, x, grid=false, legend=false, yticks=false, title="",
            color=color, linewidth=1, fill=true, fillalpha=0.2)
    end
end

for (p, df) in zip(ps[2:2:4], [dfml, dfmb])
    for (col, x) in eachcol(df, true)
        !startswith(col, "r") ? continue : nothing
        color = startswith(col, "r1") ? :black : :salmon
        density!(p, log10.(x), grid=false, legend=false, yticks=false, title="",
            color=color, linewidth=1, fill=true, fillalpha=0.2)
    end
end
plot(ps..., layout=(2,2), size=(500,400))

# This plot shows it perhaps even more clearly. ML tree based estimates seem to
# be inflated, at least relative to the CCD based estimates... There is still one
# loss rate that is suspiciously high in the CCD-based estimates, which is for
# one of the branches coming from the root. It would be cool to test this on
# simulated data.

# A relevant question is whether similar big differences are observed under a
# constant rates model.

# ## Sampling reconciled trees

# Let's investigate some reconciled gene trees sampled from the posterior.
# Set up the `WhaleModel` (as it was used for inference).
using NewickTree
wm = WhaleModel(readline(joinpath(@__DIR__, "../../example/example-3/sp.nw")), Δt=0.01)
for (i, n) in sort(wm.nodes)
    isroot(n) || isroot(wm[n.parent]) || isleaf(n) ?
        continue : addwgd!(wm, n, n.event.t*0.501, rand())
end

# Read some sample CCD files and define the `WhaleProblem`
ccd = read_ale(joinpath(@__DIR__, "../../example/example-3/D2-sample-ale"), wm)
prior = IRPrior(Ψ=[1. 0.; 0. 1.])
problem = WhaleProblem(wm, ccd, prior)

# get the posterior in the right data structure
posterior = Whale.df2vec(post)
@time recsum = sumtrees(problem, posterior)

# Some families have very high probability MAP trees, others are associated with
# a rather vague posteroir distribution over tree topologies.
ps = []
for rsum in recsum
    @unpack trees = rsum
    push!(ps, bar([trees[i].freq for i=1:min(10, length(trees))],
        legend=false, grid=false, color=:white))
end
plot(ps..., xticks=false, ylims=(-0.05,1), size=(700,400))

# The `RecSummary` objects also contain a summary of the events for each species
# tree branch observed in the family.
recsum[2].events

# !!! note
#     These are posterior mean values, for instance the duplication column shows
#     for each branch in the species tree the average number of duplications
#     observed in the posterior distribution of reconciled trees (we should
#     add some quantiles or standard deviations).

# We can get a summary of the reconciliations across all families, recording the
# average number of duplications, losses, etc. for each species tree branch for
# the full genome.
sumry = Whale.sumevents(recsum)

# This is similar to what ALE outputs, but in our case these can be interpreted
# as posterior means.

# MAPS only looks at internal branches, and has no WGD model. To get a MAPS-like
# output from the Whale results, we add the WGD-reconciled duplications to their
# species tree branches and remove leaf nodes and the root.
function whale2maps(sumry, wm)
    df = deepcopy(sumry)
    df[!,:node] = 1:size(df)[1]
    for n in Whale.getwgd(wm)
        df[id(Whale.nonwgdchild(wm[n], wm)),:duplication] +=
            df[n,:wgd] + df[n,:duplication]
    end
    todel = [[1]; collect(keys(wm.leaves)) ; Whale.getwgd(wm)]
    deleterows!(df, sort(todel))
    select!(df, Not([:wgd, :wgdloss]))
    df[!,:rowsum] = map(x->sum(x), eachrow(df))
    df
end

mapslike = whale2maps(sumry, wm)

# Now check it
plot(mapslike[!,:duplication] ./ mapslike[!,:rowsum],
    color="black", grid=false, size=(400,200), legend=false,
    xlabel="Node", ylabel="P")

# This looks very much like the results using MAPS, but this is likely
# partly coincidental, since we're only looking at 25 gene families

# We can also plot trees
using PalmTree, Luxor
import Luxor: RGB
species = Dict("dzq"=>"Pinus", "iov"=>"Pseudotsuga", "gge"=>"Cedrus",
             "xtz"=>"Araucaria", "sgt"=>"Ginkgo", "jvs"=>"Equisetum",
             "smo"=>"Sellaginella", ""=>"")
function draw(tree)
    @unpack root, annot = tree
    tl = TreeLayout(root, dim=(350, 260))
    PalmTree.cladogram!(tl)

    colfun = (n)->annot[n].label != "loss" ? RGB() : RGB(0.99,0.99,0.99)
    labfun = (k, p)->settext(" $(species[(split(annot[k].name, "_")[1])])",
        p, valign="center")
    credfn = (k, p)->settext(k ∉ tl.leaves ?
        " $(round(annot[k].cred, digits=2))" : "", p, valign="center")
    dupfn  = (k, p)->begin
        if annot[k].label == "duplication"
            box(p, 5, 5, :fill)
        elseif annot[k].label == "wgd" || annot[k].label == "wgdloss"
            star(p, 6.0, 5, 0.4, 0, :fill)
        end
    end
    @svg begin
        origin(Point(-20,20))
        setfont("Noto sans italic", 11)
        PalmTree.drawtree(tl, color=colfun)
        nodemap(tl, labfun)
        nodemap(tl, credfn)
        nodemap(tl, dupfn)
    end 400 300 #"docs/src/assets/D2-rectree1.svg"
end

rsum = recsum[7]
draw(rsum.trees[19].tree)

# ![](../assets/D2-rectree1.svg)

bar([rsum.trees[i].freq for i=1:length(rsum.trees)], color=:white,
    grid=false, legend=false, xlabel="Tree", ylabel="P")


# ## Comparing with MAPS

# Above we performed an interesting comparison between probabilistic reconciliation
# using ML trees as input and using distributions over trees as input. While the
# MAPS approach is of course related to the former in that it does not take into
# account gene tree uncertainty explicitly, it is still quite different as it is
# not model-based.

# !!! note
#     I was able to reproduce the results of the 1KP paper, so now I'm sure what
#     exactly the input data looks like. These are unrooted trees, with no bootstrap
#     support values or branch lengths.

# To enable a comparison with MAPS, we should get an idea of the number of
# duplications reconciled to each branch of the species tree in the posterior
# reconciled trees from Whale.

# ## Constraining duplication and loss rates

# A way to constrain this without sacrificing the flexibility of separate duplication
# and loss rates is by using a prior on the expected degree of expansion/contraction
# on each branch. For the ML trees however, it seems HMC becomes very slow, probably
# because the data is pulling the parameters into regions of parameter space that
# are very unlikely under the prior (is this a feature of HMC ?)
