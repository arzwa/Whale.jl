
# version april 18 (not with new RatesModel interface etc.)
using Whale, BenchmarkTools
wm  = WhaleModel(Whale.extree, Δt=0.05, minn=5)
# addwgd!(wm, wm[6], wm[6].event.t*0.5, rand())
ccd = read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), wm)

# basic logpdf
# julia> @btime logpdf!(wm, ccd[1])
# 155.765 μs (27 allocations: 512 bytes)
# -37.54937082193779

# julia> @btime logpdf!(wm, ccd)
#   2.001 ms (251 allocations: 5.28 KiB)
# -398.8334480105326

# julia> @btime logpdf(wm, ccd)
#   2.285 ms (582 allocations: 4.61 MiB)
# -398.8334480105326

# logdensity and gradient
using LogDensityProblems, TransformVariables, Distributions
prior = CRPrior(πr=MvNormal(ones(2)), πη=Beta(3,1))
p = WhaleProblem(wm, ccd, prior)
@btime LogDensityProblems.logdensity_and_gradient(p, zeros(dimension(p.trans)))
# julia> @btime LogDensityProblems.logdensity_and_gradient(p, zeros(dimension(p.trans)))
#   6.213 ms (1373 allocations: 12.14 MiB)
# (-406.6068662236971, [98.6796940684234, -215.63104722343087, -1.000342284714084, 1.4186221848682221])

# new version 18 april
r = RatesModel(ConstantDLG(λ=1.0, μ=1.0, η=0.9))
w = WhaleModel(r, t, Δt=0.05, minn=5)

@btime logpdf!(w, ccd[1])
#   123.982 μs (13 allocations: 288 bytes)
# -37.54937082193779

@btime logpdf!(w, ccd)
#   1.847 ms (63 allocations: 2.38 KiB)
# -398.8334480105326

@btime logpdf(w, ccd)
#   2.186 ms (382 allocations: 4.61 MiB)
# -398.8334480105326
