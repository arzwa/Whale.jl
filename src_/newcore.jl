#= It should be possible to use Mamba as MCMC engine, since we are actually not
doing a very special MCMC algorithm like e.g. in tree inference (where the
tree structure requires a lot of non-standard stuff in the algorithms). In the
(WH)ALE case however, the MCMC itself is more amenable to implementation in
external MCMC engines.

What we need would be the following:

- A struct analogous to a UnivariateDistribution but where the variate is a
CCD. This 'distribution' object should have a `logpdf` method which returns
the likelihood of a CCD given S, λ, μ, q, ... (which are the 'parameters' of
the distribution)

- Similar structs for the rate priors (GBM, IID).

That will be some work to implement but it's a nice opportunity to have alook
at the core code again and we might get a more efficient/reliable/maintainable
result in the end.

More tricky is how to recompute all those speedups like the partial recompute
etc.? By storing the parameter values associated with the last computation in
the CCD object? Perhaps a separate CCDArray type would work for thoe purposes

XXX PROBLEM: when Mamba is exectuted with julia in parallel, it runs chains in
parallel, but I want the likelihood to be computed in parallel.
=#

# XXX: OK here I start some rewrites with new structs etc. to make the code
# generally better and allow MCMC with Mamba (at least in principle).

#= Mamba sketch; I'm starting to like Turing.jl better...
model = Model(
    ν = Logical(() -> 0.1, false)
    η = Stochastic(() -> Beta(4, 2))
    θ = Stochastic(1,
        () -> MvNormal([log(0.2), log(0.2)], [0.5 0.25; 0.25 0.5]), false)
    λ0 = Logical((θ)->exp(θ[1]), false)
    μ0 = Logical((θ)->exp(θ[2]), false)
    λ = Stochastic(1, (λ0, ν, tree) -> GeometricBrownianMotion(tree, λ0, ν))
    μ = Stochastic(1, (μ0, ν, tree) -> GeometricBrownianMotion(tree, μ0, ν))
    q = Stochastic(1, (tree) -> UnivariateDistribution[
        Beta(1,1) for i=1:length(tree.qindex)])
    X = Stochastic(1, (λ, μ, q, η, tree) -> WhaleSamplingDist(λ, μ, q, η, tree,
        "oib"))
)
=#

#= Turing.jl
@model Whale(X, S) = begin
    # X is the observed data (a DArray of CCDs?)
    # S is the sliced species tree
    ν = 0.1
    η ~ Beta
    θ ~ MultivariateNormal
    l = θ[1]
    m = θ[2]
    λ ~ GeometricBrownianMotion(l, ν, S)
    μ ~ GeometricBrownianMotion(m, ν, S)
    q = tzeros(length(S.qindex))
    for i=1:length(S.qindex)
        q[i] ~ Beta(1., 1.)
    end
    X ~ WhaleSamplingDist(λ, μ, q, η, S, "oib")
end =#



struct WhaleSamplingDist # <: DiscreteUnivariateDistribution
    λ::Array{Float64}
    μ::Array{Float64}
    q::Array{Float64}
    η::Float64
    S::SlicedTree
    cond::String
end


# this should automatically result in partial recomputation when applicable...
# somewhere (maybe in the SlicedTree) there should be a `lastnode` field or so
# or if the update is in a fixed order, we might do it more clever?
logpdf(d::WhaleSamplingDist, x::CCD) = alepdf(x, d.S, d.λ, d.μ, d.q, d.cond)

# P({CCD}|θ, SlicedTree)


conf = read_whaleconf("./example/whalebay.conf")
tree = readtree("/home/arzwa/Whale.jl/example/morris-9taxa.nw")
st = getslicedtree(tree, conf)
