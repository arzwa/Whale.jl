# # Backtracking

# The backtracking algorithm is recursive and fairly memory-intensive. It seems
# that using an approach with DArrays is most efficient
using Distributed; #addprocs(2)
@everywhere using Pkg; @everywhere Pkg.activate("docs")
@everywhere using Whale
using BenchmarkTools, DataFrames

t = Whale.extree
θ = ConstantDLWGD(λ=0.1, μ=0.1)
r = Whale.RatesModel(θ, fixed=(:p,:η))
w = WhaleModel(r, t, .1)

pdf = DataFrame(exp.(randn(10,2) ./ 10))
fun = (m, x)-> Array(x) |> x->m((λ=x[1], μ=x[2], q=Float64[]))

ccd = read_ale("example/example-1/ale/", w, true)
tt = TreeTracker(w, ccd, pdf, fun)
trees = track(tt, outdir="/tmp/testdir2")
ev, sm = Whale.summarize(trees)

ccd = read_ale("example/example-1/ale/", w, true)
tt = TreeTracker(w, ccd, pdf, fun)
@btime trees = track(tt, progress=false)
# 706.764 ms (1806 allocations: 128.83 KiB)

ccd = read_ale("example/example-1/ale/", w, false)
tt = TreeTracker(w, ccd, pdf, fun)
@btime trees = track(tt, progress=false)
# 758.362 ms (95973 allocations: 6.19 MiB)

ccd = read_ale("example/example-1/ale/", w, false)
tt = TreeTracker(w, ccd, pdf, fun)
@btime trees = Whale.track_threaded(tt, progress=false)
# 869.151 ms (14053222 allocations: 869.97 MiB)
