using Distributed
@everywhere using Pkg
@everywhere Pkg.activate("/home/arzwa/dev/Whale.jl/")
@everywhere using Whale
using NewickTree, BenchmarkTools

using Pkg; Pkg.activate("/home/arzwa/dev/Whale.jl/")
using Whale, NewickTree, BenchmarkTools

tree = "/home/arzwa/research/wmi/sptree/gnepine.nw"
ale  = "/home/arzwa/research/wmi/whale/ale-500"

t = readnw(readline(tree))
n = length(postwalk(t))
l = (n+1) ÷ 2
r = RatesModel(ConstantDLWGD(λ=0.1, μ=0.1, q=Float64[], η=0.9, p=zeros(l)))
w = WhaleModel(r, t, Δt=0.05)

# DArray
ccd = read_ale(ale, w, true)
@btime logpdf!(w, ccd)
@btime logpdf(w, ccd)

# julia> @btime logpdf!(w, ccd)
#   114.674 ms (1799 allocations: 132.23 KiB)
# -20852.716187197897
#
# julia> @btime logpdf(w, ccd)
#   177.842 ms (1797 allocations: 132.20 KiB)
# -20852.716187197897

# Threads
ccd = read_ale(ale, w, false)
@btime logpdf!(w, ccd)
@btime logpdf(w, ccd)

# julia> @btime logpdf!(w, ccd)
#   112.510 ms (20 allocations: 2.23 KiB)
# -20852.716187197904
#
# julia> @btime logpdf(w, ccd)
#   161.785 ms (24851 allocations: 541.82 MiB)
# -20852.716187197904
