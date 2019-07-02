# Arthur Zwaenepoel - 2019
# some helper types
# Distributed CCD array
const CCDArray = DArray{CCD,1,Array{CCD,1}}
const CCDSub = SubArray{CCD,0,Array{CCD,1},Tuple{Int64},false}

# tree probability dict
const PDict{T} = Dict{Int64,Array{T,1}} where T<:Real

Base.getindex(d::PDict{T}, e::Int64, i::Int64) where T<:Real = d[e][i]
Base.setindex!(d::PDict{T}, x::T, e::Int64, i::Int64) where T<:Real =
    d[e][i] = x

"""
    WhaleModel{T<:Real,CCD}

The full Whale model, containing both the sliced species tree, parameters and
fields for extinction and propagation probabilities.
"""
struct WhaleModel{T<:Real,CCD} <: DiscreteUnivariateDistribution
    S::SlicedTree
    λ::Array{T,1}
    μ::Array{T,1}
    q::Array{T,1}
    η::T
    ε::PDict{T}
    ϕ::PDict{T}
    cond::String

    function WhaleModel(S::SlicedTree, λ::Array{T,1}, μ::Array{T,1},
            q::Array{T,1}=T[], η::T=0.9, cond="oib") where {T<:Real}
        @check_args(WhaleModel, all([λ; μ] .>= 0) && all(1. .>= [[η]; q] .>= 0))
        ε = get_ε(S, λ, μ, q, η)
        ϕ = get_ϕ(S, λ, μ, q, η, ε)
        new{T,CCD}(S, λ, μ, q, η, ε, ϕ, cond)
    end
end

function WhaleModel(S::SlicedTree, x::Vector{T}, η=0.9, c="oib") where T<:Real
    n = nrates(S)
    WhaleModel(S, x[1:n], x[n+1:2n], x[2n+1:end], promote(η, x[1])[1], c)
end

# initializer
function WhaleModel(S::SlicedTree, λ=0.2, μ=0.2, q=0.2, η=0.9, cond="oib")
    n = nrates(S)
    m = nwgd(S)
    WhaleModel(S, repeat([λ], n), repeat([μ], n), repeat([q], m), η, cond)
end

asvector1(w::WhaleModel) = [w.λ ; w.μ ; w.q]
asvector2(w::WhaleModel) = [w.λ ; w.μ ; w.q; [w.η]]

eltype(w::WhaleModel{_,T}) where {_,T} = T

Base.show(io::IO, w::WhaleModel) = show(io, w, (:λ, :μ, :q, :η))

function logpdf(m::WhaleModel, x::CCD, node::Int64=-1; matrix=false)
    if x.Γ == -1
        return 0.
    else
        l, M = whale!(x, m.S, m.λ, m.μ, m.q, m.η, m.ε, m.ϕ, m.cond, node::Int64)
        matrix ? set_tmpmat!(x, M) : nothing
        return l
    end
end

function get_ε(s::SlicedTree, λ, μ, q, η)
    ε = PDict{typeof(η)}(e => zeros(nslices(s, e)) for e in s.border)
    for e in s.border
        if isleaf(s.tree, e)
            ε[e][1] = 0.
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
        ϕ[e][1] = 1.
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

# main whale algorithm
function whale!(x::CCD, s::SlicedTree, λ::Array{T}, μ::Array{T}, q::Array{T},
        η::T, ε::PDict{T}, ϕ::PDict{T}, cond::String, node=-1) where T<:Real
    branches = (node == -1) ? s.border : get_parentbranches(s, node)
    M = DPMat{typeof(η)}()
    init_matrix!(M, x, s, branches)

    for e in s.border[1:end-1]  # skip the root branch
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
                        M[e][γ, i] = 1.0
                    elseif !(sleaf || qnode)
                        f, g = childnodes(s.tree, e)
                        if !γleaf
                            Π_speciation!(M, x, e, γ, f, g)
                        end
                        Π_loss!(M, x, e, γ, ε, f, g)
                    elseif qnode
                        qe = q[s[e, :q]]
                        f = childnodes(s.tree, e)[1]
                        if !γleaf
                            Π_wgd_retention!(M, x, e, γ, qe, f)
                        end
                        Π_wgd_non_retention!(M, x, e, γ, qe, f)
                        Π_wgd_loss!(M, x, e, γ, qe, ε[f][end], f)
                    end

                # internal of branch (slice boundary)
                else
                    Δt = s[e, i]
                    M[e][γ, i] += ϕ[e, i] * M[e][γ, i-1]
                    if !γleaf
                        Π_duplication!(M, x, e, i, γ, Δt, λe, μe)
                    end
                end
            end
        end
    end
    whale_root!(M, x, s, ε, η)
    l = cond_lhood(M, x, s, ε, η, cond)
    return l, M
end

function cond_lhood(M, x::CCD, s::SlicedTree, ε, η, cond::String)
    root = s.border[end]
    f, g = childnodes(s.tree, root)
    #if cond == "oib"
    return oib(M, x, root, f, g, ε, η)
    #end
end

function oib(M, x::CCD, e::Int64, f::Int64, g::Int64, ε, η)
    ε_root = geometric_extinctionp(ε[e, 1], η)
    ε_left = geometric_extinctionp(ε[f][end], η)
    ε_rght = geometric_extinctionp(ε[g][end], η)
    nf = 1 - ε_left - ε_rght + ε_root
    M[e, x.Γ, 1] > 0. && nf > 0. ? log(M[e, x.Γ, 1]/nf) : -Inf
end

geometric_extinctionp(ε, η) = η * ε / (1 - (1 - η)*ε)

function whale_root!(M, x::CCD, s::SlicedTree, ε, η)
    root = s.border[end]
    f, g = childnodes(s.tree, root)
    ε0 = ε[root, 1]
    η_ = 1.0/(1. - (1. - η) * ε0)^2
    for γ in x.clades
        γleaf = isleaf(x, γ)
        M[root][γ, 1] = 0.
        Π_loss!(M, x, root, γ, ε, f, g)
        if ~γleaf
            Π_speciation!(M, x, root, γ, f, g)
        end
        M[root][γ, 1] *= η_
        if ~γleaf
            Π_root!(M, x, root, γ, η, ε0)
        end
    end
    M[root, x.Γ, 1] *= η
end

function Π_root!(M, x::CCD, root::Int64, γ::Int64, η, ε0)
    p = 0.
    for (γ1, γ2, count) in x.m2[γ]
        p += x.ccp[(γ, γ1, γ2)] * M[root, γ1, 1] * M[root, γ2, 1]
    end
    p *= (1. - η) * (1. - (1. - η) * ε0)
    M[root, γ, 1] += p
end

function Π_speciation!(M, x::CCD, e::Int64, γ::Int64, f::Int64, g::Int64)
    p = 0.
    for (γ1, γ2, count) in x.m2[γ]
        p += x.ccp[(γ, γ1, γ2)] * M[f][γ1, end] * M[g][γ2, end]
        p += x.ccp[(γ, γ1, γ2)] * M[g][γ1, end] * M[f][γ2, end]
    end
    M[e, γ, 1] += p
end

function Π_loss!(M, x::CCD, e::Int64, γ::Int64, ε, f::Int64, g::Int64)
    M[e, γ, 1] += M[f][γ, end] * ε[g][end] + M[g][γ, end] * ε[f][end]
end

function Π_wgd_retention!(M, x::CCD, e::Int64, γ::Int64, q, f::Int64)
    p = 0.
    for (γ1, γ2, count) in x.m2[γ]
        p += x.ccp[(γ, γ1, γ2)] * M[f][γ1,end] * M[f][γ2,end]
    end
    M[e, γ, 1] += p * q
end

function Π_wgd_non_retention!(M, x::CCD, e::Int64, γ::Int64, q, f::Int64)
    M[e, γ, 1] += (1-q) * M[f][γ, end]
end

function Π_wgd_loss!(M, x::CCD, e::Int64, γ::Int64, q, ε, f::Int64)
    M[e, γ, 1] += 2 * q * ε * M[f][γ, end]
end

function Π_duplication!(M, x::CCD, e::Int64, i::Int64, γ::Int64, Δt, λe, μe)
    p = 0.
    for (γ1, γ2, count) in x.m2[γ]
        p += x.ccp[(γ, γ1, γ2)] * M[e, γ1, i-1] * M[e, γ2, i-1]
    end
    # XXX instead of λe*Δt? p_transition_kendall(2, Δt, λe, μe)
    M[e, γ, i] += p * λe * Δt
end

function init_matrix!(M::DPMat{T}, x::CCD, s::SlicedTree, bs) where T<:Real
    for b in s.border
        if b in bs
            M[b] = zeros(T, length(x.clades), length(s.slices[b]))
        else
            M[b] = x.tmpmat[b]
        end
    end
end

set_tmpmat!(x::CCD, M::DPMat) = x.tmpmat = M

logpdf(m::WhaleModel, x::Array{CCD,1}, node::Int64=-1) =
    sum(logpdf.(m, x))

# DistributedArrays parallelism
logpdf(m::WhaleModel, x::CCDArray, node::Int64=-1) =
    mapreduce((x)->logpdf(m, x, node), +, x)

set_recmat!(D::CCDArray) = ppeval(_set_recmat!, D)

function _set_recmat!(x::CCDSub)
    x[1].recmat = x[1].tmpmat
    return 0.
end

gradient(m::WhaleModel, x::CCDArray) = mapreduce((x)->gradient(m, x), +, x)

function gradient(m::WhaleModel, x::CCD)
    v = asvector1(m)
    f = (u) -> logpdf(WhaleModel(m.S, u), x)
    g = ForwardDiff.gradient(f, v)
    return g[:, 1]
end

# utils
"""
    describe(w::WhaleModel)

Get a detailed description of a particular `WhaleModel`.
"""
function describe(w::WhaleModel)
    println("Leaves")
    println("======")
    for (n, l) in w.S.leaves
        println("$n \t→ $l")
    end
    println("Rates (λ, μ)")
    println("============")
    for n in w.S.border
        i = w.S.rindex[n]
        l = join(w.S.clades[n], ",")
        print("$n \t| λ, μ = $(w.λ[i]), $(w.μ[i])")
        println("\t| ($l)")
    end
    println("WGDs (q)")
    println("========")
    for (n, v) in w.S.qindex
        println("$n, q = $(w.q[v])")
    end
    println("Other")
    println("=====")
    println("   η = $(w.η)")
    println("cond = $(w.cond)")
end
