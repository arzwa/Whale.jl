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
etc.? By storing the parameter values associted with the lat computation in
the CCD object? Perhaps a separate CCDArray type would work for thoe purposes =#

# XXX: OK here I start some rewrites with new structs etc. to make the code
# generally better and allow MCMC with Mamba (at least in principle).

struct SlicedTree <: Arboreal
    tree::Tree
    index::Dict{Symbol,Dict{Int64,Int64}}  # {:wgd => Dict, :λ => Dict, ...}
    leaves::Dict{Int64,String}
    clades::Dict{Int64,Set{Int64}}
    border::Array{Int64}  # postorder of species tree branches
    slices::Dict{Int64,Array{Float64,1}}
end

struct WhaleSamplingDist <: DiscreteUnivariateDistribution
    λ::Array{Float64}
    μ::Array{Float64}
    q::Array{Float64}
    S::SlicedTree
    cond::String
end

# this should automatically result in partial recomputation when applicable...
# somewhere (maybe in the SlicedTree) there should be a `lastnode` field or so
logpdf(d::WhaleSamplingDist, x::CCD) = alepdf(x, d.S, d.λ, d.μ, d.q, d.cond)
