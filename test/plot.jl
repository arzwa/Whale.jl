using Pkg; Pkg.activate(@__DIR__)
using Whale, NewickTree, Parameters, ForwardDiff
using Test, Random, Distributed
using Plots, Measures

data = joinpath("example/example-1/ale")
t = deepcopy(Whale.extree)
n = length(postwalk(t))
r = DLWGD(λ=ones(n), μ=ones(n), η=0.9)
w = WhaleModel(r, t, 0.05, maxn=5)
ccd = read_ale(data, w)
l = logpdf!(w, ccd[1])

G = Whale.backtrack(w, ccd[1])

plot(w, G)


function rectangles!(p, l; kwargs...)
    for (k,v) in l
        plot!(Shape(v); kwargs...)
    end
    p
end

d = Dict("ATHA"=>"A. thaliana", "MPOL"=>"M. polymorpha", "PPAT"=>"P. patens",
         "SMOE"=>"S. moellendorffii", "OSAT"=>"O. sativa", "CPAP"=>"C. papaya",
         "ATRI"=>"A. trichopoda", "GBIL"=>"G. biloba", "PABI"=>"P. abies")
l = Whale.species_layout(getroot(w), ypad=2)
g = Whale.genetree_layout(w, G, l[1])
p = plot(grid=false, legend=false, framestyle=:none)
rectangles!(p, l[1], color=:lightgray, linecolor=:lightgray)
rectangles!(p, l[2], color=:lightgray, alpha=0.3, linecolor=:lightgray, linealpha=0.3)
for (k,v) in g
    plot!(p, v, color=:black)
    if !isroot(k)
        plot!(p, [v[2], g[parent(k)][1]], color=:black)
    end
end
for n in getleaves(getroot(w))
    cs = l[1][id(n)]
    x = (cs[end][1] + cs[1][1])/2
    y = cs[1][2]
    #annotate!(p, x, y-5, text(d[name(n)], 10, :center, "helvetica oblique", rotation=15))
    annotate!(p, x, y-2, text(name(n), 10, :top, "helvetica"))
end
plot(p, ylim=(-5,Inf))

savefig("/home/arzwa/vimwiki/phd/img/whale/examplerec.pdf")
