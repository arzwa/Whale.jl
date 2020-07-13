# Potentially worth to open issues in DynamicHMC:
# - save intermediate chains (so that we can run a chain indefinitely, and stop
#   it when we're happy)

# We could have a separate RatesModel interface for hyperparameters of a prior,
# and define a struct that combines RatesModels into one

"""
    WhaleProblem
"""
struct WhaleProblem{T,M,V,I}
    data ::Union{Nothing,CCDArray{T,I}}  # NOTE require DArray
    model::WhaleModel{T,M,I}
    prior::V
end

function fand∇f(f::Function, x, cfg)
    result = DiffResults.GradientResult(x)
    result = ForwardDiff.gradient!(result, f, x, cfg)
    DiffResults.value(result), DiffResults.gradient(result)
end

function fand∇f(prior::Prior, t, x)
    fun = (x) -> logpdf(prior, t(x))
    cfg = ForwardDiff.GradientConfig(fun, x, ForwardDiff.Chunk{length(x)}())
    return fand∇f(fun, x, cfg)
end

function fand∇f(trans::TransformTuple, x)
    fun = (x) -> transform_and_logjac(trans, x)[2]
    cfg = ForwardDiff.GradientConfig(fun, x, ForwardDiff.Chunk{length(x)}())
    return fand∇f(fun, x, cfg)
end

# for sampling from the prior alone
fand∇f(wm::WhaleModel, data::Nothing, x) = 0., zeros(dimension(wm.rates.trans))

# parallel computation of ℓ and ∇ℓ, all derivation is within parallel processes
# and the partial values are accumulated on the main process
function fand∇f(wm::WhaleModel, data::CCDArray, x)
    y = [@spawnat i _grad(wm, localpart(data), x) for i in procs(data)]
    result = fetch.(y)
    acc = foldl(+, result)
    acc[1], acc[2:end]
end

function _grad(wm::WhaleModel, data::Vector, x)
    function fun(x)
        model = wm(x)  # sets the model
        mapreduce(u->logpdf(model, u), +, data)
    end
    cfg = ForwardDiff.GradientConfig(fun, x, ForwardDiff.Chunk{length(x)}())
    vcat(fand∇f(fun, x, cfg)...)
end

function LogDensityProblems.logdensity_and_gradient(p::WhaleProblem, x)
    @unpack model, prior, data = p
    ℓ, ∇ℓ = fand∇f(model, data, x)
    π, ∇π = fand∇f(prior, model.rates, x)
    J, ∇J = fand∇f(model.rates.trans, x)
    # @show J, ∇J
    return (ℓ + π + J)::Float64, (@. ∇ℓ + ∇π + ∇J)::Vector{Float64}
end

LogDensityProblems.capabilities(::Type{<:WhaleProblem}) =
    LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.dimension(p::WhaleProblem) = dimension(p.model.rates.trans)

TransformVariables.transform(p::WhaleProblem, x) =
    transform(p.model.rates.trans, x)

function track(p::WhaleProblem, post;
        progress=true, outdir::String="")
    @unpack model, data = p
    # NOTE eachrow on vector of namedtuples produces for each element (i.e.
    # namedtuple) a subarray with that element.
    tt = TreeTracker(model, data, post, (model, x)->model(x[1]))
    track_distributed(tt, progress=progress, outdir=outdir)
end
