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

# NOTE: consider removing these functions, as there is no information that is
# not contained in the summarized trees
# backtrack(p::WhaleProblem, posterior) =
#     backtrack(p.model, p.data, posterior, p.model.rates)
#
# function backtrack(wm, ccd, posterior, rates)
#     function bt(x)
#         wmm = wm(x)
#         logpdf!(wmm, ccd)
#         Array(backtrack(wmm, ccd))
#     end
#     permutedims(hcat(map(bt, posterior)...))
# end
#
# function sumtrees(p::WhaleProblem, posterior)
#     # NOTE: this will do the whole backtracking + sumarizing routine in the
#     # inner (parallel) loop. This avoids storing a huge array (N × n) of
#     # reconciled trees
#     @unpack model, data = p
#     function track_and_sum(ccd)
#         trees = Array{RecNode,1}(undef, length(posterior))
#         for (i,x) in enumerate(posterior)
#             wmm = model(x)
#             logpdf!(wmm, ccd)
#             trees[i] = backtrack(wmm, ccd)
#         end
#         sumtrees(trees, ccd, model)
#     end
#     map(track_and_sum, data)
# end
