
# Posterior analytics (not by Aristotle) example and MAPS D2

```@example mapsD2
using CSV, Plots, DataFrames
```

We first define some functions to parse the dumped posterior file.

```@example mapsD2
function parsepost!(df)
    f = (x) -> eval.(Meta.parse.(x))
    for (col, x) in eachcol(df, true)
        df[!,col] = try; f(x); catch; x; end
    end
end

function unpack(df)
    cols = []
    for (col, x) in eachcol(df, true)
        push!(cols, unpack.(x, col))
    end
    DataFrame(merge.(cols...))
end

unpack(x::T, sym::Symbol) where T<:Real = (; sym=>x)
unpack(x::Matrix, sym::Symbol) = (; [Symbol("$(sym)$(i)_$(j)")=>x[i,j]
    for i=1:size(x)[1], j=1:size(x)[2]]...)
unpack(x::Vector, sym::Symbol) =
    (; [Symbol("$(sym)$(i)")=>x[i] for i=1:length(x)]...)
```

## Whale posterior for MAPS D2 from the 1KP study
Load the file and parse it

```@example mapsD2
post = CSV.read(joinpath(@__DIR__, "../../example/example-3/hmc-D2.mltrees.csv"))
parsepost!(post);
dfml = unpack(post)
```

A function to easily collect trace plots

```@example mapsD2
traces(df; kwargs...) = [plot(x, title=col; kwargs...)
    for (col, x) in eachcol(df, true)]
```

... and let's show some trace plots using it:

```@example mapsD2
ps = traces(dfml, grid=false, legend=false, xticks=false, yticks=false,
    color=:black, linewidth=0.2, title_loc=:left, titlefont=7)
plot(ps..., size=(700,600))
```

The same for marginal distributions (we could define a plot recipe to combine
these with the trace plots)

```@example mapsD2
marginalhists(df; kwargs...) = [stephist(x, title=col; kwargs...)
    for (col, x) in eachcol(df, true)]

ps = marginalhists(dfml, grid=false, legend=false, yticks=false,
    color=:black, linewidth=1, title_loc=:left, titlefont=7, alpha=0.1, fill=true)
plot(ps..., size=(700,600))
```

For this example, which is a reanalysis of a gymnosperm data set from the 1KP
paper (using ML trees, so no gene tree uncertainty is taken into account), we
clearly see a consistent inflation of the loss rate relative to the duplication
rate. This is rather suspicious, since, if we would take this at face value, it
would entail persistent genome contraction, which is no known feature of
gymnosperm evolution. It seems the birth-death process is wandering of in rather
unlikely regions of parameter space.

## Using MrBayes tree samples (accounting for tree uncertainty)

```@example mapsD2
post = CSV.read(joinpath(@__DIR__, "../../example/example-3/hmc-D2.mbtrees.csv"))
parsepost!(post)
dfmb = unpack(post);
ps = traces(dfmb, grid=false, legend=false, xticks=false, yticks=false,
    color=:black, linewidth=0.2, title_loc=:left, titlefont=7)
plot(ps..., size=(700,600))
```

We'll compare the marginal posterior distributions between the Whale results
based on ML trees and those based on CCDs.

```@example mapsD2
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
```

This is very interesting, it seems taking into account uncertainty in gene
trees has a huge effect... There seems to be one branch (3, one of the branches
emanating from the root) where we get higher loss rates than in the ML trees case,
but for all other branches we find more reasonable estimates for the CCD
based analysis. Interestingly, also the support for WGD hypotheses is very
sensitive to the different data sets, with the Gymnosperm WGD going from extreme
support to q ≈ 0.

!!! note
    This is of course not suitable as a proxy for MAPS, as MAPS uses
    a BSV-based cut-off and some taxon-occupancy thresholds as well...

!!! note
    It would be best to validate (using ML for instance) the new implementation
    on trees with uncertainty as well (some random swaps etc?) before
    we go on with this.

## Sampling reconciled trees

Let's investigate some reconciled gene trees sampled from the posterior.
Set up the `WhaleModel` (as it was used for inference).

```@example mapsD2
using Whale
wm = WhaleModel(readline(joinpath(@__DIR__, "../../example/example-3/sp.nw")), Δt=0.01)
wgds = [(["dzq","iov"], 0.005),
        (["dzq","gge"], 0.045),
        (["dzq","xtz"], 0.015),
        (["dzq","sgt"], 0.050),
        (["dzq","jvs"], 0.015)]
for (lca, t) in wgds
    node = Whale.lcanode(wm, lca)
    addwgd!(wm, wm[node], t, rand())
end
```

Read some sample CCD files and define the `WhaleProblem`

```@example mapsD2
ccd = read_ale(joinpath(@__DIR__, "../../example/example-3/D2-sample-ale"), wm)
prior = IRPrior(Ψ=[1. 0.; 0. 1.])
problem = WhaleProblem(wm, ccd, prior)
```

get the posterior in the right data structure

```@example mapsD2
df2vec(df) = [(; [x=>y[x] for x in names(y)]...) for y in eachrow(df)]
posterior = df2vec(post)

trees    = backtrack(problem, posterior)
rectrees = sumtrees(trees, ccd, wm)
```

Some families have very high probability MAP trees, others are associated with
a rather vague posteroir distribution over tree topologies.

```@example mapsD2
ps = []
for t in rectrees
    push!(ps, bar([t[i].freq for i=1:min(10, length(t))], legend=false, grid=false, color=:white))
end
plot(ps..., xticks=false, ylims=(-0.05,1), size=(700,400))
```

We can also plot trees

```@example mapsD2
using PalmTree, Parameters, Luxor
import Luxor: RGB
function draw(tree)
    @unpack root, annot = tree
    tl = TreeLayout(root, dim=(380, 260))
    PalmTree.cladogram!(tl)

    colfun = (n)->annot[n].label != "loss" ? RGB() : RGB(0.99,0.99,0.99)
    labfun = (k, p)->settext(" $(split(annot[k].name, "_")[1])", p, valign="center")
    credfn = (k, p)->settext(k ∉ tl.leaves ?
        " $(annot[k].cred)" : "", p, valign="center")
    @svg begin
        origin(Point(-20,20))
        setfont("Noto sans italic", 11)
        PalmTree.drawtree(tl, color=colfun)
        nodemap(tl, labfun)
        nodemap(tl, credfn)
    end 400 300 "/tmp/rectree.svg"
end

draw(rectrees[7][1].tree)
```

## Constraining duplication and loss rates

A way to constrain this without sacrificing the flexibility of separate duplication
and loss rates is by using a prior on the expected degree of expansion/contraction
on each branch.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

