struct RootCondition <: SamplingCondition end
struct NonExtinctCondition <: SamplingCondition end

struct NowhereExtinctCondition{T} <: SamplingCondition
    s::Vector{T}  # signs for inclusion - exclusion
    NowhereExtinctCondition(model, T=Float64) =
        new{T}(treepgf_allbinary_sign(model))
end

condition(wm::WhaleModel) = condition(wm, wm.condition)

# log probability of non-extinction
function condition(wm::WhaleModel, ::NonExtinctCondition)
    @unpack η = getθ(wm.rates, root(wm))
    log(1. -geompgf(η, getϵ(root(wm))))
end

# log probability of non extinction in both clades stemming from the root
function condition(wm::WhaleModel, ::RootCondition)
    @unpack η = getθ(wm.rates, root(wm))
    f, g = children(root(wm))
    ϵr = geompgf(η, getϵ(root(wm)))
    ϵf = geompgf(η, getϵ(f))
    ϵg = geompgf(η, getϵ(g))
    p = one(η) - ϵf - ϵg + ϵr
    p > zero(p) ? log(p) : -Inf
end

function condition(wm::WhaleModel, e::NowhereExtinctCondition{T}) where T
    p = treepgf_allbinary(wm)
    # probability of at least one count at each leaf
    log(one(T) - sum((p .* e.s)[1:end-1]))
end

# log probability of non-extinction everywhere? naive implementation
# function condition(wm::WhaleModel{T}, ::NowhereCondition, bound=10) where T
#     𝑃 = zeros(T, bound, length(wm.order))
#     p = one(T)
#     function walk(n)
#         _pvec!(𝑃, wm, n)
#         for c in children(n) walk(c) end
#         if isleaf(n)
#             p *= sum(𝑃[2:end, id(n)])
#         end
#         return
#     end
#     walk(root(wm))
#     return log(p)
# end
#
# function _pvec!(𝑃, model, n)
#     if isroot(n)
#         @unpack η = getθ(model.rates, n)
#         𝑃[:,id(n)] = [0. ; pdf.(Geometric(η), 0:size(𝑃)[1]-2)]
#     else
#         @unpack λ, μ = getθ(model.rates, n)
#         t = distance(n)
#         bound = size(𝑃)[1]
#         matrix = [tp(i, j, t, λ, μ) for i=0:bound-1, j=0:bound-1]
#         p = matrix' * 𝑃[:,id(parent(n))]
#         𝑃[:,id(n)] .= p /sum(p)
#     end
# end
