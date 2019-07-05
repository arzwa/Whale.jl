# MCMC as in original Whale implementation (AMWG)
# there are many targets for optimization here, but perhaps not very important
# as the likelihood routine is still dominating

# Model (priors)
# ==============
# XXX optimize by making Model a proper subtype of Distribution → type stability
abstract type Model end

const Prior = Union{<:Distribution,Array{<:Distribution,1},<:Real}
const State = Dict{Symbol,Union{Vector{Float64},Float64}}

Base.getindex(x::State, s::Symbol, i::Int64) = x[s][i]

# HACK: for constant parameters
Distributions.logpdf(d::Float64, x::Float64) = 0.

# GBM model
"""
    GBMModel(st::SlicedTree, ν::T, η::T, λ::T, μ::T, q::T) where
        T::Union{<:Distribution,Array{<:Distribution,1},<:Real}

Hierarchical model for Whale using a GBM prior.
"""
struct GBMModel <: Model
    ν::Prior
    η::Prior
    λ::Prior
    μ::Prior
    q::Prior
end

GBMModel(st::SlicedTree, ν=InverseGamma(15), η=Beta(10, 1), λ=Exponential(1),
    μ=Exponential(1), q=[Beta(1,1) for i=1:nwgd(st)]) = GBMModel(ν, η, λ, μ, q)

function Distributions.logpdf(m::GBMModel, x::State, st::SlicedTree, args...)
    x_ = deepcopy(x)
    for (k, v) in args; x_[k] = v ; end
    p = 0.
    for f in fieldnames(typeof(m))
        if f == :q
            p += sum(logpdf.(m.q, x_[:q]))
        elseif f == :λ || f == :μ
            p += logpdf(getfield(m, f), x_[f][1])
            p += logpdf(GBM(st, x_[f][1], x_[:ν]), x_[f])
        else
            p += logpdf(getfield(m, f), x_[f])
        end
    end
    return p
end

function Base.rand(m::GBMModel, st)
    x = State()
    for f in fieldnames(typeof(m))
        d = getfield(m, f)
        x[f] = typeof(d) <: Real ? d : (
            typeof(d) <: AbstractArray ? rand.(d) : rand(d))
    end
    x[:λ] = rand(GBM(st, x[:λ], x[:ν]))
    x[:μ] = rand(GBM(st, x[:μ], x[:ν]))
    return x
end

# IRModel
"""
    IRModel(st::SlicedTree, ν::T, η::T, λ::T, μ::T, q::T) where
        T::Union{<:Distribution,Array{<:Distribution,1},<:Real}

Hierarchical model for Whale using a independent rates prior.
"""
struct IRModel <: Model
    ν::Prior
    η::Prior
    λ::Prior
    μ::Prior
    q::Prior
end

IRModel(st::SlicedTree, ν=InverseGamma(15), η=Beta(10, 1), λ=Exponential(1),
    μ=Exponential(1), q=[Beta(1,1) for i=1:nwgd(st)]) = IRModel(ν, η, λ, μ, q)

function Distributions.logpdf(m::IRModel, x::State, st::SlicedTree, args...)
    x_ = deepcopy(x)
    for (k, v) in args; x_[k] = v ; end
    p = 0.
    for f in fieldnames(typeof(m))
        if f == :q
            p += sum(logpdf.(m.q, x_[:q]))
        elseif f == :λ || f == :μ
            p += logpdf(getfield(m, f), x_[f,1])
            v = repeat([log(x_[f,1])], nrates(st)-1)
            p += logpdf(MvLogNormal(v, x_[:ν]), x_[f][2:end])
        else
            p += logpdf(getfield(m, f), x_[f])
        end
    end
    return p
end

function Base.rand(m::IRModel, st)
    x = State()
    for f in fieldnames(typeof(m))
        d = getfield(m, f)
        x[f] = typeof(d) <: Real ? d : (
            typeof(d) <: AbstractArray ? rand.(d) : rand(d))
    end
    x[:λ] = rand(MvLogNormal(repeat([log(x[:λ])], nrates(st)), x[:ν]))
    x[:μ] = rand(MvLogNormal(repeat([log(x[:μ])], nrates(st)), x[:ν]))
    return x
end

# Samplers
# ========
mutable struct Sampler
    accepted::Int64
    tuneinterval::Int64
    kernel::Normal{Float64}
end

Sampler(σ::Float64, tuning=20) = Sampler(0, tuning, Normal(0., σ))

Base.rand(spl::Sampler) = rand(spl.kernel)
Base.rand(spl::Sampler, n::Int64) = rand(spl.kernel, n)

const Samplers = Dict{Symbol,Union{Vector{Sampler},Sampler}}
Base.getindex(spl::Samplers, s::Symbol, i::Int64) = spl[s][i]

function get_samplers(x::State, σ=0.05)
    s = Samplers()
    for (k, v) in x
        s[k] = typeof(v) <: AbstractArray ?
            [Sampler(σ) for i=1:length(v)] : Sampler(σ)
    end
    return s
end

function adapt!(spl::Sampler, gen::Int64, target=0.25, bound=5., δmax=0.25)
    gen == 0 ? (return) : nothing
    δn = min(δmax, 1/√(gen/spl.tuneinterval))
    α = spl.accepted / spl.tuneinterval
    lσ = α > target ? log(spl.kernel.σ) + δn : log(spl.kernel.σ) - δn
    lσ = abs(lσ) > bound ? sign(lσ) * bound : lσ
    spl.kernel = Normal(0., exp(lσ))
    spl.accepted = 0
end

# Chain
"""
    WhaleChain(st::SlicedTree, π::Model)

Chain object for performing MCMC under various hierarchical models defined in
Whale.
"""
mutable struct WhaleChain{T<:Model}
    S::SlicedTree
    state::State
    prior::T
    samplers::Samplers
    gen::Int64
    wgdbranches::Array{Tuple{Int64,Int64}}
    df::DataFrame

    WhaleChain(s::SlicedTree, π::T, x::State, spl::Samplers) where T<:Model =
        new{T}(s, x, π, spl, 0, wgdbranches(s), DataFrame())
end

WhaleChain(st, π, x) = WhaleChain(st, π, x, get_samplers(x))
WhaleChain(st, π) = WhaleChain(st, π, init_state(π, st))
WhaleChain(st) = WhaleChain(st, GBMModel(st))
WhaleModel(w::WhaleChain) = WhaleModel(w.S, w[:λ], w[:μ], w[:q], w[:η])

Base.getindex(w::WhaleChain, s::Symbol) = w.state[s]
Base.getindex(w::WhaleChain, s::Symbol, i::Int64) = w.state[s][i]
Base.setindex!(w::WhaleChain, x, s::Symbol) = w.state[s] = x
Base.setindex!(w::WhaleChain, x, s::Symbol, i::Int64) = w.state[s][i] = x

Distributions.logpdf(w::WhaleChain, args...) =
    logpdf(w.prior, w.state, w.S, args...)

function MCMCChains.Chains(w::WhaleChain)
    X = reshape(Matrix(w.df), (size(w.df)...,1))[:, 2:end, :]
    return Chains(X, [string(x) for x in names(w.df)][2:end])
end

function init_state(prior, st)
    x = rand(prior, st)
    x[:π] = -Inf
    x[:l] = -Inf
    return x
end

function getstate(st::SlicedTree, row)
    λ = Vector(row[getnames(row, "λ")])
    μ = Vector(row[getnames(row, "μ")])
    q = Vector(row[getnames(row, "q")])
    η = row[:η]
    return WhaleModel(st, λ, μ, q, η)
end

getnames(row, s) = [x for x in names(row) if startswith(string(x), s)]

function init!(w::WhaleChain, D::CCDArray)
    w[:π] = logpdf(w)
    w[:l] = logpdf(WhaleModel(w), D, matrix=true)
    Whale.set_recmat!(D)
end

# MCMC algorithm
"""
    mcmc!(w::WhaleChain, D::CCDArray, n::Int64, args...; kwargs...)

Perform `n` generations of MCMC sampling for a `WhaleChain` given a bunch of
observed CCDs.
"""
function mcmc!(w::WhaleChain, D::CCDArray, n::Int64, args...;
        show_trace=true, show_every=10, backtrack::Bool=true)
    init!(w, D)
    for i=1:n
        cycle!(w, D, args...)
        w.gen += 1
        log_mcmc(w, stdout, show_trace, show_every)
        backtrack ? backtrack!(D, WhaleModel(w)) : nothing
    end
    Chains(w)
end

function cycle!(w::WhaleChain, D::CCDArray, args...)
    !(:ν in args) ? sample_ν!(w) : nothing
    !(:η in args) ? sample_η!(w, D) : nothing
    gibbs_sweep!(w, D)
    q_sweep!(w, D)
    wgd_sweep!(w, D)
end

function log_mcmc(w, io, show_trace, show_every)
    if w.gen == 1
        s = w.state
        x = [typeof(v)<:AbstractArray ? ["$k$i" for i in 1:length(v)] :
                k for (k,v) in s]
        x = vcat("gen", x...)
        w.df = DataFrame(zeros(0,length(x)), [Symbol(k) for k in x])
        show_trace ? write(io, join(x, ","), "\n") : nothing
    end
    x = vcat(w.gen, [x for x in values(w.state)]...)
    push!(w.df, x)
    if show_trace && w.gen % show_every == 0
        write(io, join(x, ","), "\n")
    end
    flush(stdout)
end

function sample_ν!(x::WhaleChain)
    spl = x.samplers[:ν]
    ν_ = x[:ν] + rand(spl)
    ν_ < 0. ? (return) : nothing
    p = logpdf(x, :ν=>ν_)
    a = p - x[:π]
    if log(rand()) < a
        x[:ν] = ν_
        x[:π] = p
        spl.accepted += 1
    end
    x.gen % spl.tuneinterval == 0 ? adapt!(spl, x.gen) : nothing
    return
end

function sample_η!(x::WhaleChain, D::CCDArray)
    spl = x.samplers[:η]
    η_ = reflect(x[:η] + rand(spl))
    p = logpdf(x, :η=>η_)
    l = logpdf(WhaleModel(x.S, x[:λ], x[:μ], x[:q], η_), D, 1, matrix=true)
    a = p + l - x[:π] - x[:l]
    if log(rand()) < a
        x[:η] = η_
        x[:π] = p
        x[:l] = l
        spl.accepted += 1
        set_recmat!(D)
    end
    x.gen % spl.tuneinterval == 0 ? adapt!(spl, x.gen) : nothing
    return
end

function gibbs_sweep!(x::WhaleChain, D::CCDArray)
    for i in x.S.border
        # WGD branches have the same rates before and after WGD
        haskey(x.S.qindex, i) ? continue : nothing
        spl = x.samplers[:λ, i]
        λᵢ = x[:λ, i] + rand(spl)
        μᵢ = x[:μ, i] + rand(spl)
        λᵢ < 0. || μᵢ < 0. ? continue : nothing
        λ_ = deepcopy(x[:λ]); μ_ = deepcopy(x[:μ])
        λ_[i] = λᵢ; μ_[i] = μᵢ
        p = logpdf(x, :λ=>λ_, :μ=>μ_)
        l = logpdf(WhaleModel(x.S, λ_, μ_, x[:q], x[:η]), D, i, matrix=true)
        a = p + l - x[:π] - x[:l]
        if log(rand()) < a
            x[:λ, i] = λᵢ
            x[:μ, i] = μᵢ
            x[:π] = p
            x[:l] = l
            spl.accepted += 1
            set_recmat!(D)
        end
        x.gen % spl.tuneinterval == 0 ? adapt!(spl, x.gen) : nothing
    end
end

function wgd_sweep!(x::WhaleChain, D::CCDArray)
    for (b, i) in x.wgdbranches
        idx = x.S.qindex[b]
        splr = x.samplers[:λ, i]
        splq = x.samplers[:q, idx]
        λᵢ = x[:λ, i] + rand(splr)
        μᵢ = x[:μ, i] + rand(splr)
        λᵢ < 0. || μᵢ < 0. ? continue : nothing
        qᵢ = reflect(x[:q, idx] + rand(splq))
        λ_ = deepcopy(x[:λ]); μ_ = deepcopy(x[:μ]); q_ = deepcopy(x[:q])
        λ_[i] = λᵢ; μ_[i] = μᵢ; q_[idx] = qᵢ
        p = p = logpdf(x, :λ=>λ_, :μ=>μ_, :q=>q_)
        l = logpdf(WhaleModel(x.S, λ_, μ_, q_, x[:η]), D, i, matrix=true)
        a = p + l - x[:π] - x[:l]
        # NOTE: no adaptation based on this proposal; because both q and rates
        if log(rand()) < a
            x[:λ, i] = λᵢ
            x[:μ, i] = μᵢ
            x[:q, idx] = qᵢ
            x[:π] = p
            x[:l] = l
            set_recmat!(D)
        end
    end
end

function q_sweep!(x::WhaleChain, D::CCDArray)
    for (b, i) in x.S.qindex
        spl = x.samplers[:q, i]
        qᵢ = reflect(x[:q, i] + rand(spl))
        q_ = deepcopy(x[:q])
        q_[i] = qᵢ
        p = p = logpdf(x, :q=>q_)
        l = logpdf(WhaleModel(x.S, x[:λ], x[:μ], q_, x[:η]), D, b, matrix=true)
        a = p + l - x[:π] - x[:l]
        if log(rand()) < a
            x[:q, i] = qᵢ
            x[:π] = p
            x[:l] = l
            spl.accepted += 1
            set_recmat!(D)
        end
        x.gen % spl.tuneinterval == 0 ? adapt!(spl, x.gen) : nothing
    end
end

function reflect(x::Float64, a::Float64=0., b::Float64=1.)
    while !(a <= x <= b)
        x = x < a ? 2a - x : x
        x = x > b ? 2b - x : x
    end
    return x
end
