using Pkg; Pkg.activate("./test")
using Parameters
using NewickTree
using BenchmarkTools
# using ForwardDiff
# using Optim
include("_model.jl")
include("_ccd.jl")
include("_core.jl")

wm = WhaleModel(extree)
ccd = CCD("./example/example-ale/OG0004533.fasta.nex.treesample.ale", wm)

f(x) = -logpdf(wm(ConstantRates(exp(x[1]), exp(x[2]), 0.85)), ccd)
g = (x) -> ForwardDiff.gradient(f, x)
g!(G, x) = G .= g(x)

init = randn(2)
results = optimize(f, g!, init, show_trace=true)
@show exp.(results.minimizer)
