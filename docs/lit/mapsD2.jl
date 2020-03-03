# # Posterior analytics (not by Aristotle) example and MAPS D2
using CSV, Plots, DataFrames, Parameters, Whale, StatsPlots

# ## Whale posterior for MAPS D2 from the 1KP study
# Load the file and parse it
base = joinpath(@__DIR__, "../../example/example-3/")

# ## Using MrBayes tree samples (accounting for tree uncertainty)

dfmb = CSV.read(joinpath(base, "hmc-D2.mb.csv"))
dfml = CSV.read(joinpath(base, "hmc-D2.ml.csv"))

# We'll compare the marginal posterior distributions between the Whale results
# based on ML trees and those based on CCDs.
Base.startswith(s::Symbol, prefix::String) = startswith(string(s), prefix)
ps = []
gr()
for (col, x) in eachcol(dfml, true)
    title =
        startswith(col, "r_1") ? "\\lambda$(split(string(col), "_")[3])" :
        startswith(col, "r_2") ? "\\mu$(split(string(col), "_")[3])" : col
    p = stephist(x, grid=false, legend=false, label="ML trees",
        yticks=false, title=title, color=:salmon, title_loc=:left, titlefont=7,
        normalize=true, fill=true, alpha=0.5)
    stephist!(p, dfmb[:,col], color=:firebrick, fill=true, alpha=0.5,
        normalize=true, label="CCDs")
    startswith(col, "q") ? xticks!(p, 0:0.2:1) : nothing
    startswith(col, "q") ? xlims!(0, max(0.2, round(maximum(x), digits=1))) : nothing
    push!(ps, p)
end
plot(ps..., size=(800,600))
savefig("/home/arzwa/vimwiki/presentations/lm0320/assets/posterior-ml-ccd.pdf")

# For this example, which is a reanalysis of a gymnosperm data set from the 1KP
# paper (using ML trees, so no gene tree uncertainty is taken into account), we
# clearly see a consistent inflation of the loss rate relative to the duplication
# rate. This is rather suspicious, since, if we would take this at face value, it
# would entail persistent genome contraction, which is no known feature of
# gymnosperm evolution. It seems the birth-death process is wandering of in rather
# unlikely regions of parameter space.

# This is very interesting, it seems taking into account uncertainty in gene
# trees has a huge effect... There seems to be one branch (3, one of the branches
# emanating from the root) where we get higher loss rates than in the ML trees case,
# but for all other branches we find more reasonable estimates for the CCD
# based analysis. Interestingly, also the support for WGD hypotheses is very
# sensitive to the different data sets, with the Gymnosperm WGD going from extreme
# support to q ≈ 0.

# !!! note
#     We should fix η as in the simulations by Li *et al.*

# !!! note
#     This is of course not suitable as a proxy for MAPS, as MAPS uses
#     a BSV-based cut-off and some taxon-occupancy thresholds as well... **Turns
#     out in the 1KP analyses no BSV threshold was used?**

# !!! note
#     It would be best to validate (using ML for instance) the new implementation
#     on trees with uncertainty as well (some random swaps etc?) before
#     we go on with this.

using StatsPlots, LaTeXStrings

pyplot()
ps = [plot(), plot()]
for (i,(p, df)) in enumerate(zip(ps, [dfml, dfmb]))
    for (col, x) in eachcol(df, true)
        !startswith(col, "r") ? continue : nothing
        color = startswith(col, "r_1") ? :black : :firebrick
        lab   = startswith(col, "r_1") ? L"\lambda" : L"\mu"
        density!(p, x, grid=false, legend=false, yticks=false, title="",
            color=color, linewidth=1, fill=true, fillalpha=0.2, label=lab)
        xlabel!(p, L"\log \theta")
    end
    i == 1 ? title!("ML", title_loc=:left, titlefont=9) :
        title!("CCD (MrBayes)", title_loc=:left, titlefont=9)
end
plot(ps..., layout=(2,1), size=(500,400), xlims=(-5,5))
savefig("/home/arzwa/vimwiki/presentations/lm0320/assets/whale-ml-mb-rates.pdf")

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
wm = WhaleModel(readline(joinpath(base, "sp.nw")), Δt=0.01)
for (i, n) in sort(wm.nodes)
    isroot(n) || isleaf(n) ?
        continue : addwgd!(wm, n, n.event.t*0.5, rand())
end

# Read some sample CCD files and define the `WhaleProblem`
ccd = read_ale(joinpath(base, "D2-sample-ale"), wm)
problem = WhaleProblem(wm, ccd, IRPrior())

# get the posterior in the right data structure
posterior = Whale.df2vec(post)

# implementation changed, now we need log-scale
logr(tup) = (; [k=> (k==:r ? log.(v) : v) for (k,v) in pairs(tup)]...)
posterior = [logr(x) for x in posterior]
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

# We can also plot trees
using PalmTree, Luxor, ColorSchemes
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
    end 400 300 "docs/src/assets/D2-rectree1.svg"
end

rsum = recsum[7]
draw(rsum.trees[21].tree)

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
# reconciled trees from Whale. This we obtain using the `sumtrees` and
# `sumevents` methods. We further add the duplication events reconciled to
# WGD nodes to the total number of duplication events on the relevant branch
# to obtain the expected number of duplications on a species tree branch. Lastly,
# MAPS does not consider leaf branches, so we remove these.
function whale2maps(sumry, wm)
    df = DataFrame(Matrix(sumry), names(sumry))
    df[:,:node] = 1:size(df)[1]
    todel = [[1]; collect(keys(wm.leaves))]
    for n in Whale.getwgd(wm)
        id(wm[n]) > size(df)[1] ? continue : nothing
        df[id(Whale.nonwgdchild(wm[n], wm)),:duplication] +=
            df[n,:wgd] + df[n,:duplication]
        push!(todel, id(wm[n]))
    end
    deleterows!(df, sort(todel))
    select!(df, Not([:wgd, :wgdloss]))
    df
end

# Note, Notung reformats the species tree such that a ladder-tree will be
# represented as `((((a,b)n2,c)n4,d)n6,e)` so that the order of the internal
# node labels is the reverse of the pre-order used in Whale.
function notung2maps(df)
    dff = df[startswith.(df[:,:Column1], "n"),:]
    dff[!,:node] = map(x->parse(Int, x[2:end]), dff[!,:Column1])
    sort(dff, :node)[1:end-1,:]
end

# Now we can try to compare
mapsdf   = CSV.read(joinpath(base, "maps-D2.csv"))
maps0df  = CSV.read(joinpath(base, "maps-D2-mt0.csv"))
notungdf = notung2maps(CSV.read(joinpath(base, "notung-D2.csv")))
mbsumdf  = whale2maps(CSV.read(joinpath(base, "hmc-D2-mbtrees.recsum.csv")), wm)
mlsumdf  = whale2maps(CSV.read(joinpath(base, "hmc-D2-mltrees.recsum.csv")), wm)

maps    = mapsdf[!,:Duplication]
maps0   = maps0df[!,:Duplication]
whalemb = reverse(mbsumdf[!,:duplication])
whaleml = reverse(mlsumdf[!,:duplication])
notung  = notungdf[!,:Dups]

# Now plot the comparison
# pyplot()  # doesn't look too nice on GR
using Plots, ColorSchemes
groupedbar([maps maps0 notung whaleml whalemb],
    color=reshape(get(ColorSchemes.viridis, 0.2:0.2:1), (1,5)),
    label=["MAPS" "MAPS (mt=0)" "Notung" "Whale (ML)" "Whale (CCD)"],
    xlabel="Species tree node", ylabel="# duplication events",
    bar_width=0.7, bar_position=:dodge, size=(600,300),
    grid=false, legend=:topleft)
savefig("/home/arzwa/vimwiki/presentations/lm0320/assets/reccomp.pdf")

# Look at that! We infer more duplication events with Whale for all but the
# first node. This is unexpected, since LCA reconciliation in general results
# in the inference of *more* duplication and loss events. It seems the
# 'taxon-occupancy filter' has some role in this, but it does not explain the
# observations either, since even when excluding this filter, we do infer
# more duplications for the Whale analyses. Maybe the rooting issues could
# cause this?

# The comparison with Notung is shocking. NOTUNG, as expected, infers *a lot*
# more events than Whale. MAPS, which performs some kind of LCA reconciliation
# somehow leads to very different and unexpected results...

# However all methods infer similar relative numbers of duplications on different
# branches relative

using PalmTree, Parameters, Luxor, NewickTree
import Luxor: RGB
begin
    NewickTree.isleaf(i::UInt16) = isleaf(wm[i])
    NewickTree.distance(i::UInt16) = wm[i].event.t
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
    end 380 260 "/home/arzwa/tmp/D2.tree"
end
