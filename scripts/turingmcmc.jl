# Whale MCMC engine
# The specific nature of the problem (especially the tree structure) makes
# inference using generic engines (Turing, Mamba) rather cumbersome.

#=
ν ~ InverseGamma
η ~ Beta
q ~ [Beta]
θ ~ MvLogNormal
λ ~ GBM
μ ~ GBM

How can we implement this in e.g. Turing, with the same speed-ups as the custom
implementation?

We might need to define a different Sampler object for the rates that uses the
partial recomputation scheme. This should implement adaptation.

Similarly for η
=#
# Turing =======================================================================
using Distributed
using Distributions
@everywhere using Turing
if nworkers() <= 1
    addprocs(2)
end
@everywhere using Whale

# worked!
@model gbmwhale(x) = begin
    ν ~ InverseGamma(10.)
    η ~ Beta(10, 2)
    λ ~ MvLogNormal(ones(nrates(st)))
    μ ~ MvLogNormal(ones(nrates(st)))
    x ~ WhaleModel(st, λ, μ, typeof(λ[1])[], η)
end

# worked!
@model gbmwhale(x) = begin
    ν ~ InverseGamma(10.)
    η ~ Beta(10, 2)
    q = Vector{Real}(undef, nwgd(st))
    for i in eachindex(q)
        q[i] ~ Beta(1, 1)
    end
    λ ~ MvLogNormal(ones(nrates(st)))
    μ ~ MvLogNormal(ones(nrates(st)))
    x ~ WhaleModel(st, λ, μ, float.(q), η)
end

# worked, this is effectively an iid model
using Whale
using Turing
st = Whale.example_tree()
ccd = read_ale("/home/arzwa/Whale.jl/example/example-ale/", st, d=false)

@model iidwhale(x) = begin
    ν ~ InverseGamma(100.)
    η ~ Beta(10, 2)
    q = Vector{Real}(undef, nwgd(st))
    for i in eachindex(q)
        q[i] ~ Beta(1, 1)
    end
    r ~ Exponential(0.2)
    θ ~ MvLogNormal([log(r), log(r)], [.5 0.45 ; 0.45 0.5])
    λ ~ MvLogNormal(repeat([log(θ[1])], nrates(st)), ones(nrates(st)))
    μ ~ MvLogNormal(repeat([log(θ[2])], nrates(st)), ones(nrates(st)))
    x ~ [WhaleModel(st, λ, μ, float.(q), η)]  # vectorized
    # x ~ WhaleModel(st, λ, μ, float.(q), η)  # parallel, issues!
end
turingmodel = iidwhale(repeat(ccd, 10))
turingmodel = iidwhale(ccd)
chain = sample(turingmodel, HMC(1000, 0.0001, 1))

#= 3 threads
@time chain = sample(turingmodel, HMC(100, 0.1, 1))
┌ Info: Finished 100 sampling steps in 46.671314065 (s)
│   typeof(h.metric) = AdvancedHMC.Adaptation.UnitEuclideanMetric
│   typeof(τ) = AdvancedHMC.StaticTrajectory{AdvancedHMC.Leapfrog{Float64}}
│   EBFMI(Hs) = 16.499714169371913
└   mean(αs) = 0.8158860524748017
 46.862336 seconds (99.98 M allocations: 106.590 GiB, 18.90% gc time)
=#

#= 1 thread
julia> @time chain = sample(turingmodel, HMC(100, 0.1, 1))
┌ Info: Finished 100 sampling steps in 51.37785391 (s)
│   typeof(h.metric) = AdvancedHMC.Adaptation.UnitEuclideanMetric
│   typeof(τ) = AdvancedHMC.StaticTrajectory{AdvancedHMC.Leapfrog{Float64}}
│   EBFMI(Hs) = 179.2183623077022
└   mean(αs) = 1.3633824551121218e-7
 51.577726 seconds (99.98 M allocations: 106.590 GiB, 20.44% gc time)

 julia> @time chain = sample(turingmodel, HMC(100, 0.01, 1))
┌ Info: Finished 100 sampling steps in 456.630230858 (s)
│   typeof(h.metric) = AdvancedHMC.Adaptation.UnitEuclideanMetric
│   typeof(τ) = AdvancedHMC.StaticTrajectory{AdvancedHMC.Leapfrog{Float64}}
│   EBFMI(Hs) = 1.7207417000657395
└   mean(αs) = 0.9950987740345328
457.968410 seconds (954.16 M allocations: 1.029 TiB, 19.67% gc time)

=#

chain = sample(turingmodel, SGLD(1000, 0.1))
chain = sample(turingmodel, NUTS(1000, 0.65))

@model gbmwhale(x) = begin
    ν ~ InverseGamma(10.)
    η ~ Beta(10, 2)
    q = Vector{Real}(undef, nwgd(st))
    for i in eachindex(q)
        q[i] ~ Beta(1, 1)
    end
    r ~ Exponential(1.0)
    θ ~ MvLogNormal([log(r), log(r)], [1. 0.9 ; 0.9 1.])
    λ ~ GBM(st, θ[1], ν)
    μ ~ GBM(st, θ[2], ν)
    x ~ WhaleModel(st, λ, μ, float.(q), η)
end
turingmodel = gbmwhale(distribute(ccd))
chain = sample(turingmodel, MH(1000))
chain = sample(turingmodel, HMC(10, 0.1, 10))


@model mwe(x) = begin
    σ ~ InverseGamma(1.)
    μ ~ Normal()
    x ~ [Normal(μ, σ)]  # logpdf evaluation should be in parallel
end
m = mwe(randn(10000))
c = sample(m, HMC(1000, 0.01, 10))


c = sample(turingmodel, MH(1000))

#=
Problem: AD + Distributed computing
Solution: Compute gradient using mapreduce
New problem: how to use this gradient function in Turing?
=#
