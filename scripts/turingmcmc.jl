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

using Whale
using Turing
st = Whale.example_tree()
ccd = read_ale("example/example-ale/", st, d=false)

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
@model iidwhale(x) = begin
    ν ~ InverseGamma(10.)
    η ~ Beta(10, 2)
    q = Vector{Real}(undef, nwgd(st))
    for i in eachindex(q)
        q[i] ~ Beta(1, 1)
    end
    r ~ Exponential(1.0)
    θ ~ MvLogNormal([log(r), log(r)], [1. 0.9 ; 0.9 1.])
    λ ~ MvLogNormal(repeat([log(θ[1])], nrates(st)), ones(nrates(st)))
    μ ~ MvLogNormal(repeat([log(θ[2])], nrates(st)), ones(nrates(st)))
    x ~ [WhaleModel(st, λ, μ, float.(q), η)]  # vectorized
    # x ~ WhaleModel(st, λ, μ, float.(q), η)  # parallel, issues!
end
turingmodel = iidwhale(ccd)
chain = sample(turingmodel, HMC(1000, 0.1, 10))
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
