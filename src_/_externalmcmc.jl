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
using Turing
@model gbmwhale(x) = begin
    ν ~ InverseGamma(10.)
    η ~ Beta(10, 2)
    q = zeros(nwgd(st))
    for i in eachindex(q)
        q[i] ~ Beta(1, 1)
    end
    r ~ Exponential(1.0)
    θ ~ MvLogNormal([log(r), log(r)], [1. 0.9 ; 0.9 1.])
    λ ~ GBM(st, θ[1], ν)
    μ ~ GBM(st, θ[2], ν)
    x ~ WhaleModel(st, λ, μ, q, η)
end
turingmodel = gbmwhale(distribute(ccd))
chain = sample(turingmodel, MH(1000))
chain = sample(turingmodel, NUTS(1000, 200, 0.5))
chain = sample(turingmodel, Gibbs(1000, HMC(2, 0.1, 5, :θ, :r, :q, :η, :ν),
                              MH(1, :λ, :μ)))

# So this works for the MH sampler; can't get it to work for the samplers that
# require AD...

# I should probably implement samplers myself, like the AMWG? I can then use
# the compositional approach. So I should implement e.g. AMWG(:λ), and make sure
# that when Turing comes to sampling the rates, this dedicated sampler is used?

# Mamba ========================================================================
# seems to only work for Floats etc.
using Mamba
model = Model(
    ν = Logical(() -> 0.1, false),
    η = Stochastic(() -> Beta(4, 2)),
    r = Stochastic(() -> Exponential(1.0)),
    θ = Stochastic(1,
        () -> MvLogNormal([log(r), log(r)], [1.0 0.9; 0.9 1.0]), false),
    λ0 = Logical((θ)->θ[1], false),
    μ0 = Logical((θ)->θ[2], false),
    λ = Stochastic(1, (λ0, ν) -> GBM(st, λ0, ν)),
    μ = Stochastic(1, (μ0, ν) -> GBM(st, μ0, ν)),
    q = Stochastic(1, () -> UnivariateDistribution[
        Beta(1,1) for i=1:nwgd(st)]),
    x = Stochastic(1, (λ, μ, q, η) -> WhaleModel(st, λ, μ, q, η)))

scheme1 = [AMM([:ν, :η, :r, :θ, :λ, :μ, :q])]
scheme2 = [Mamba.NUTS([:ν, :η, :r, :θ, :λ, :μ, :q])]
initd = Dict(
    :ν => 0.1,
    :η => 0.9,
    :r => 0.5,
    :θ => [0.5, 0.5],
    :λ => rand(GBM(st, 0.5, 0.1)),
    :μ => rand(GBM(st, 0.5, 0.1)),
    :q => rand(nwgd(st)),
    :x=>[x])

data = Dict{Symbol,Any}(:x=>x)
setsamplers!(model, scheme2)
sim2 = mcmc(model, data, [initd], 10000, burnin=250, thin=2, chains=1)
# ==============================================================================
