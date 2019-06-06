OK here I start some rewrites with new structs etc. to make the code
generally better and allow MCMC with Mamba (at least in principle).

Mamba sketch; I'm starting to like Turing.jl better...
```julia
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
```

Turing.jl
```julia
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
end
```

Every CCD object stores it's own rec. matrix Where should the extinction and
propagation probabilities be stored? I guess at the WhaleModel level, because
they result from applying the WhaleParams to the tree?

A way to clean up further is to provide methods for the CCD, instead of using the `.` to access fields. like `get_triples(x::CCD, γ::Int64)` etc.
