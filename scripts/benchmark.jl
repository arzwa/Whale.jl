using Whale
using PhyloTrees

st = Whale.example_tree()
ccd = read_ale("/home/arzwa/Whale.jl/example/example-ale/", st)
x = ccd[1]
λ = rand(length(Set(collect(values(st.rindex)))))
μ = rand(length(Set(collect(values(st.rindex)))))
q = rand(length(st.qindex))
η = 0.9

# core2/3
w = WhaleModel(st, λ, μ, q, η)
@time logpdf(w, x)
@benchmark logpdf(w, x)
@code_warntype Whale.whale!(x, st, w.λ, w.μ, w.q, w.η, w.ε, w.ϕ, w.cond, -1)
@time Whale.whale!(x, st, w.λ, w.μ, w.q, w.η, w.ε, w.ϕ, w.cond, -1)

# AD tests
@everywhere getparams(v, n) = v[1:n], v[n+1:2n], v[2n+1:end-1], v[end]

function flogpdf(w, x)
    n = nrates(w.S)
    return (v) -> logpdf(WhaleModel(st, getparams(v, n)...), x)
end
v = [λ ; μ ; q; η]
f = flogpdf(w, ccd)
cfg = ForwardDiff.GradientConfig(nothing, v)
g = v -> ForwardDiff.gradient(f, v, cfg)
g(v)


D = distribute(ccd)
logpdf(w, D)

#=
Benchmarks on the first ccd in the example directory [carey]
using logpdf(w, x)

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
  memory estimate:  644.58 KiB
  allocs estimate:  6095
  --------------
  minimum time:     833.437 μs (0.00% GC)
  median time:      852.538 μs (0.00% GC)
  mean time:        965.943 μs (8.44% GC)
  maximum time:     93.400 ms (98.92% GC)
  --------------
  samples:          5163
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
