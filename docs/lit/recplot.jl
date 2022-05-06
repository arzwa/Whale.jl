using Pkg; Pkg.activate(@__DIR__)
using Whale, NewickTree, Plots, Measures
theme(:wong2)

# load data and set up model
t = nw"(((A:0.3,B:0.3):0.5,(C:0.6,D:0.6):0.2):0.2,E:1.);"
n = length(postwalk(t))
r = ConstantDLWGD(λ=0.8, μ=0.9, η=0.9)
w = WhaleModel(r, t, 0.01)

# simulate a couple of trees
trees = Whale.dlsimbunch(t, r, 100, condition=:root)
aledir = Whale.aleobserve(trees)
ccd = read_ale(aledir, w)
l = logpdf!(w, ccd)

# make a plot for the 16th
ps = map(1:6) do i
    G = Whale.backtrack(w, ccd[16])
    plot(w, G, sscale=50.)
end 
plot(ps..., size=(700,400))

