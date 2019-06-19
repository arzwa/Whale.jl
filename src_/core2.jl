# Arthur Zwaenepoel - 2019
# some helper types
const PDict{T} = Dict{Int64,Array{T,1}} where T<:Real

Base.getindex(d::PDict{T}, e::Int64, i::Int64) where T<:Real = d[e][i]

Base.setindex!(d::PDict{T}, x::T, e::Int64, i::Int64) where T<:Real =
    d[e][i] = x

"""
    $(TYPEDEF)

The full Whale model, containing both the sliced species tree, parameters and
fields for extinction and propagation probabilities.
"""
struct WhaleModel{T<:Real,CCD} <: DiscreteUnivariateDistribution
    S::SlicedTree
    λ::Array{T}
    μ::Array{T}
    q::Array{T}
    η::T
    ε::PDict{T}
    ϕ::PDict{T}
    cond::String

    function WhaleModel(S::SlicedTree, λ::Array{T}, μ::Array{T},
            q::Array{T}=T[], η::T=0.9, cond="oib") where {T<:Real}
        ε = get_ε(S, λ, μ, q, η)
        ϕ = get_ϕ(S, λ, μ, q, η, ε)
        new{T,CCD}(S, λ, μ, q, η, ε, ϕ, cond)
    end
end

eltype(w::WhaleModel{_,T}) where {_,T} = T

function Base.show(io::IO, w::WhaleModel)
    write(io, "Tree of $(ntaxa(w.S)) taxa with $(nrates(w.S)) rate classes")
    write(io, " and $(nwgd(w.S)) WGDs")
end

function get_ε(s::SlicedTree, λ, μ, q, η)
    ε = PDict{typeof(η)}(e => zeros(nslices(s, e)) for e in s.border)
    for e in s.border
        if isleaf(s.tree, e)
            ε[e, 1] = 0.
        elseif haskey(s.qindex, e)
            qe = q[s[e, :q]]
            f = childnodes(s.tree, e)[1]
            ε_wgd = ε[f][nslices(s, f)]
            ε[e, 1] = qe * ε_wgd^2 + (1-qe) * ε_wgd
        else
            f, g = childnodes(s.tree, e)
            ε[e, 1] = ε[f, nslices(s, f)] * ε[g, nslices(s, g)]
        end
        if isroot(s.tree, e)
            return ε
        end
        for i in 2:nslices(s, e)
            λe = λ[s[e, :λ]]
            μe = μ[s[e, :μ]]
            ε[e, i] = ε_slice(λe, μe, s[e, i], ε[e, i-1])
        end
    end
end

ε_slice(λ, μ, t, ε) = isapprox(λ, μ, atol=1e-5) ?
    1. + (1. - ε)/(μ * (ε - 1.) * t - 1.) :
        (μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ

function get_ϕ(s::SlicedTree, λ, μ, q, η, ε::PDict)
    ϕ = PDict{typeof(η)}(e => zeros(nslices(s, e)) for e in s.border)
    for e in s.border[1:end-1]
        λe = λ[s[e, :λ]]
        μe = μ[s[e, :μ]]
        ϕ[e, 1] = 1.
        for i in 2:nslices(s, e)
            ϕ[e, i] = ϕ_slice(λe, μe, s[e, i], ε[e, i-1])
        end
    end
    return ϕ
end

function ϕ_slice(λ, μ, t, ε)
    if isapprox(λ, μ, atol=1e-5)
        return 1. / (μ * (ε - 1.) * t - 1.)^2
    else
        x = exp((μ - λ)*t)
        a = x * (λ - μ)^2
        b = λ - (x * μ)
        c = (x - 1.) * λ * ε
        return a / (b + c)^2
    end
end

check_args(m::WhaleModel) =
    all([m.λ; m.μ] .> 0) && all(1. .>= [[m.η]; m.q] .>= 0)

function logpdf(m::WhaleModel, x::CCD, node::Int64=-1)
    if ~check_args(m) ; return -Inf; end
    return x.Γ == -1 ? 0. :
        whale!(x, m.S, m.λ, m.μ, m.q, m.η, m.ε, m.ϕ, m.cond, node::Int64)
end

# main whale algorithm
function whale!(x::CCD, s, λ, μ, q, η, ε, ϕ, cond::String, node::Int64)
    branches = (node == -1) ? s.border[1:end-1] : get_parentbranches(s, node)
    set_tmpmat!(x, s, branches)

    for e in branches  # skip the root branch
        qnode = haskey(s.qindex, e)
        sleaf = isleaf(s, e)
        λe = λ[s[e, :λ]]
        μe = μ[s[e, :μ]]

        for γ in x.clades
            !(x.species[γ] ⊆ s.clades[e]) ? (continue) : nothing
            γleaf = isleaf(x, γ)

            for i in 1:nslices(s, e)
                # speciation or WGD node
                if i == 1
                    if γleaf && x.m3[γ] == e
                        setp!(x, e, γ, i, 1.0)
                    elseif !(sleaf || qnode)
                        f, g = childnodes(s.tree, e)
                        if !γleaf
                            Π_speciation!(x, e, γ, f, g)
                        end
                        Π_loss!(x, e, γ, ε, f, g)
                    elseif qnode
                        qe = q[s[e, :q]]
                        f = childnodes(s.tree, e)[1]
                        if !γleaf
                            Π_wgd_retention!(x, e, γ, qe, f)
                        end
                        Π_wgd_non_retention!(x, e, γ, qe, f)
                        Π_wgd_loss!(x, e, γ, qe, ε[f][end], f)
                    end

                # internal of branch (slice boundary)
                else
                    Δt = s[e, i]
                    x.tmpmat[e, γ, i] += ϕ[e, i] * x.tmpmat[e, γ, i-1]
                    if !γleaf
                        Π_duplication!(x, e, i, γ, Δt, λe, μe)
                    end
                end
            end
        end
    end
    whale_root!(x, s, ε, η)
    cond_lhood(x, s, ε, η, cond)
end

function cond_lhood(x::CCD, s::SlicedTree, ε, η, cond::String)
    root = s.border[end]
    f, g = childnodes(s.tree, root)
    #if cond == "oib"
    return oib(x, root, f, g, ε, η)
    #end
end

function oib(x::CCD, e::Int64, f::Int64, g::Int64, ε::PDict, η::Float64)
    ε_root = geometric_extinctionp(ε[e, 1], η)
    ε_left = geometric_extinctionp(ε[f][end], η)
    ε_rght = geometric_extinctionp(ε[g][end], η)
    nf = 1 - ε_left - ε_rght + ε_root
    x.tmpmat[e, x.Γ, 1] > 0. && nf > 0. ? log(x.tmpmat[e, x.Γ, 1]/nf) : -Inf
end

geometric_extinctionp(ε::Float64, η::Float64) = η * ε / (1 - (1 - η)*ε)

function whale_root!(x::CCD, s::SlicedTree, ε::PDict, η::Float64)
    root = s.border[end]
    f, g = childnodes(s.tree, root)
    ε0 = ε[root, 1]
    η_ = 1.0/(1. - (1. - η) * ε0)^2
    for γ in x.clades
        γleaf = isleaf(x, γ)
        x.tmpmat[root, γ, 1] = 0.
        Π_loss!(x, root, γ, ε, f, g)
        if ~γleaf
            Π_speciation!(x, root, γ, f, g)
        end
        x.tmpmat[root, γ, 1] *= η_
        if ~γleaf
            Π_root!(x, root, γ, η, ε0)
        end
    end
    x.tmpmat[root, x.Γ, 1] *= η
end

function Π_root!(x::CCD, root::Int64, γ::Int64, η, ε0)
    p = 0.
    for (γ1, γ2, count) in x.m2[γ]
        p += x.ccp[(γ, γ1, γ2)] * x.tmpmat[root, γ1, 1] * x.tmpmat[root, γ2, 1]
    end
    p *= (1. - η) * (1. - (1. - η) * ε0)
    x.tmpmat[root, γ, 1] += p
end

function Π_speciation!(x::CCD, e::Int64, γ::Int64, f::Int64, g::Int64)
    p = 0.
    for (γ1, γ2, count) in x.m2[γ]
        p += x.ccp[(γ, γ1, γ2)] * x.tmpmat[f][γ1, end] * x.tmpmat[g][γ2, end]
        p += x.ccp[(γ, γ1, γ2)] * x.tmpmat[g][γ1, end] * x.tmpmat[f][γ2, end]
    end
    x.tmpmat[e, γ, 1] += p
end

function Π_loss!(x::CCD, e::Int64, γ::Int64, ε, f::Int64, g::Int64)
    x.tmpmat[e, γ, 1] += x.tmpmat[f][γ, end] * ε[g][end] +
        x.tmpmat[g][γ, end] * ε[f][end]
end

function Π_wgd_retention!(x::CCD, e::Int64, γ::Int64, q, f::Int64)
    p = 0.
    for (γ1, γ2, count) in x.m2[γ]
        p += x.ccp[(γ, γ1, γ2)] * x.tmpmat[f][γ1,end] * x.tmpmat[f][γ2,end]
    end
    x.tmpmat[e, γ, 1] += p * q
end

function Π_wgd_non_retention!(x::CCD, e::Int64, γ::Int64, q, f::Int64)
    x.tmpmat[e, γ, 1] += (1-q) * x.tmpmat[f][γ, end]
end

function Π_wgd_loss!(x::CCD, e::Int64, γ::Int64, q, ε, f::Int64)
    x.tmpmat[e, γ, 1] += 2 * q * ε * x.tmpmat[f][γ, end]
end

function Π_duplication!(x::CCD, e::Int64, i::Int64, γ::Int64, Δt, λe, μe)
    p = 0.
    for (γ1, γ2, count) in x.m2[γ]
        p += x.ccp[(γ, γ1, γ2)] * x.tmpmat[e, γ1, i-1] * x.tmpmat[e, γ2, i-1]
    end
    # p12 = p_transition_kendall(2, Δt, λ, μ)
    # return p * p12
    #= NOTE: it would be more correct to use the BD transition probability to go
    from one lineage to two lineages. In practice it doesn't make a
    difference as long as the slice lengths (Δt) are  short enough (∼ [λ/10,
    λ/5]) and using the approximation seems to counteract some bias. =#
    x.tmpmat[e, γ, i] += p * λe * Δt
end

function set_tmpmat!(x::CCD, s::SlicedTree, branches::Array{Int64})
    for b in [branches; s.border[end]]
        x.tmpmat[b] = zeros(length(x.clades), length(s.slices[b]))
    end
end
