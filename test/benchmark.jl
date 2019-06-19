using PhyloTrees
using Distributions
using Distributed
using DataFrames
using DistributedArrays
using Optim
using Printf
using BenchmarkTools
using Random
import PhyloTrees: isleaf
import ProgressMeter: @showprogress
import Distributions: logpdf, @check_args, rand, eltype
import DocStringExtensions: TYPEDSIGNATURES, SIGNATURES, TYPEDEF

Random.seed!(1234)

base = "/home/arzwa/Whale.jl/src_/"

include("$base/slicedtree.jl")
include("$base/ccd.jl")
include("$base/core3.jl")
#include("$base/core2.jl")
#include("$base/core.jl")

# benchmark data
wgdconf = Dict(
    "PPAT" => ("PPAT", 0.655, -1.0),
    "CPAP" => ("CPAP", 0.275, -1.0),
    "BETA" => ("ATHA", 0.55, -1.0),
    "ANGI" => ("ATRI,ATHA", 3.08, -1.0),
    "SEED" => ("GBIL,ATHA", 3.9, -1.0),
    "MONO" => ("OSAT", 0.91, -1.0),
    "ALPH" => ("ATHA", 0.501, -1.0))
tree = readtree("/home/arzwa/Whale.jl/example/morris-9taxa.nw")
st = SlicedTree(tree, wgdconf)
ccd = read_ale("/home/arzwa/Whale.jl/example/example-ale/", st)
x = ccd[1]
λ = rand(length(Set(collect(values(st.rindex)))))
μ = rand(length(Set(collect(values(st.rindex)))))
q = rand(length(st.qindex))
η = 0.9

#=
Benchmarks on the first ccd in the example directory [carey]

version in core.jl:
BenchmarkTools.Trial:
  memory estimate:  645.59 KiB
  allocs estimate:  6229
  --------------
  minimum time:     821.537 μs (0.00% GC)
  median time:      867.854 μs (0.00% GC)
  mean time:        953.005 μs (7.81% GC)
  maximum time:     4.132 ms (76.56% GC)
  --------------
  samples:          5234
  evals/sample:     1


version in core2.jl
BenchmarkTools.Trial:
  memory estimate:  643.72 KiB
  allocs estimate:  6109
  --------------
  minimum time:     831.670 μs (0.00% GC)
  median time:      847.098 μs (0.00% GC)
  mean time:        945.498 μs (9.29% GC)
  maximum time:     4.336 ms (79.37% GC)
  --------------
  samples:          5276
  evals/sample:     1


version in core3.jl (allows AD...)
BenchmarkTools.Trial:
  memory estimate:  630.78 KiB
  allocs estimate:  6074
  --------------
  minimum time:     814.420 μs (0.00% GC)
  median time:      832.352 μs (0.00% GC)
  mean time:        951.050 μs (7.90% GC)
  maximum time:     6.666 ms (84.16% GC)
  --------------
  samples:          5242
  evals/sample:     1


Benchmarking of AD
BenchmarkTools.Trial:
  memory estimate:  22.42 MiB
  allocs estimate:  40091
  --------------
  minimum time:     8.939 ms (17.46% GC)
  median time:      9.261 ms (17.71% GC)
  mean time:        9.363 ms (17.79% GC)
  maximum time:     11.018 ms (19.08% GC)
  --------------
  samples:          534
  evals/sample:     1
=#

# core1
m = WhaleParams(λ, μ, q, η)
w = WhaleModel(st, m)
@time logpdf(w, x)
@benchmark logpdf(w, x)

# core2
w = WhaleModel(st, λ, μ, q, η)
@time logpdf(w, x)
@benchmark logpdf(w, x)
@code_warntype whale!(x, st, w.λ, w.μ, w.q, w.η, w.ε, w.ϕ, w.cond, -1)
@time whale!(x, st, w.λ, w.μ, w.q, w.η, w.ε, w.ϕ, w.cond, -1)

# AD tests
function flogpdf(w, x)
    n = nrates(w.S)
    return (v) -> logpdf(WhaleModel(st, v[1:n], v[n+1:2n],
        v[2n+1:end-1], v[end]), x, matrix=false)
end
v = [λ ; μ ; q; η]
f = flogpdf(w, x)
g = v -> ForwardDiff.gradient(f, v)
g(v)
