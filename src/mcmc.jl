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

- `ν`: autocorrelation strength (variance of geometric Brownian motion)
- `η`: parameter of geometric prior on the number of genes at the root of the species tree S
- `λ`: duplication rate at the root of S
- `μ`: loss rate at the root of S
- `q`: retention rates

Example: InverseGamma prior on `ν`, Beta prior on `η`, Exponential priors on `λ`
and `μ` at the root and Beta(1,1) priors on the retention rates `q`.

```julia-repl
julia> m = GBMModel(st, InverseGamma(15), Beta(10, 1), Exponential(1), Exponential(1), [Beta(1,1) for i=1:nwgd(st)]);

julia> rand(m, st)  # sample a state from the prior
Dict{Symbol,Union{Float64, Array{Float64,1}}} with 5 entries:
  :ν => 0.0771695
  :μ => [0.498764, 0.500195, 0.459161, 0.615959, 0.482646, 0.502002, 0.402976, 0.354524, 0.378785, …
  :λ => [0.760383, 0.733029, 0.696493, 0.592979, 0.718899, 1.01307, 0.642244, 0.639109, 0.600577, 0…
  :η => 0.915222
  :q => [0.761553, 0.729741, 0.116805, 0.278579, 0.200221, 0.293774, 0.869646]
```
"""
struct GBMModel <: Model
    ν::Prior
    η::Prior
    λ::Prior
    μ::Prior
    q::Prior
end

GBMModel(st::SlicedTree, ν=Exponential(0.5), η=Beta(10, 1), λ=Exponential(1),
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

Hierarchical model for Whale using a independent rates prior. As
[`GBMModel`](@ref) but `ν` is the variance of the log-Normal prior on the
branch-wise rates.

Example:
```julia-repl
julia> m = IRModel(st, InverseGamma(10), Beta(6,3));

julia> m.ν
InverseGamma{Float64}(
invd: Gamma{Float64}(α=10.0, θ=1.0)
θ: 1.0
)

julia> m.λ
Exponential{Float64}(θ=1.0)

julia> rand(m, st)
Dict{Symbol,Union{Float64, Array{Float64,1}}} with 5 entries:
  :ν => 0.0955299
  :μ => [0.456988, 0.43639, 0.626645, 0.464031, 0.540543, 0.629841, 0.574996, 0.498506, 0.428841, 0…
  :λ => [0.498921, 0.69414, 0.54769, 0.533595, 0.521085, 0.515748, 0.534503, 0.594045, 0.583829, 0.…
  :η => 0.486934
  :q => [0.970779, 0.965934, 0.788127, 0.405944, 0.827755, 0.253859, 0.395734]
```
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

Sampler(σ::Float64, tuning=50) = Sampler(0, tuning, Normal(0., σ))

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
    s[:ψ] = Sampler(σ)
    return s
end

function adapt!(spl::Sampler, gen::Int64, target=0.25, bound=5., δmax=0.25)
    gen == 0 ? (return) : nothing
    δn = min(δmax, 1. /√(gen/spl.tuneinterval))
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

```julia-repl
julia> w = WhaleChain(st, IRModel(st))
WhaleChain{IRModel}(SlicedTree(9, 17, 7))

julia> w[:q]  # current state's q values
7-element Array{Float64,1}:
 0.4068351791361286
 0.5893350551453437
 0.21472854312451684
 0.8068526003250731
 0.05239284527812649
 0.8432325466244709
 0.9006557436550706

julia> w[:μ, 3]  # current state of μ for branch 3
0.6578486200316909

julia> logpdf(w)  # prior logpdf
59.40417835345352

julia> logpdf(w, :ν=>0.1)  # prior logpdf, but with ν = 0.1
58.49660562613573

julia> logpdf(w, :η=>0.2)  # prior logpdf, but with η = 0.2
45.22412219063766

julia> WhaleModel(w)  # WhaleModel corresponding to current state of the chain
WhaleModel{Float64,CCD}(
λ: [0.440648, 0.486472, 0.412741, 0.432659, 0.392268, 0.485483, 0.485181, 0.422687, 0.431686, 0.477667, 0.405392, 0.414238, 0.442341, 0.390893, 0.429505, 0.41379, 0.435038]
μ: [0.635544, 0.575146, 0.657849, 0.657189, 0.626361, 0.668117, 0.619245, 0.600366, 0.682475, 0.570257, 0.58954, 0.576526, 0.6966, 0.729826, 0.561087, 0.614903, 0.668041]
q: [0.406835, 0.589335, 0.214729, 0.806853, 0.0523928, 0.843233, 0.900656]
η: 0.966691254252368
```
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
Base.display(io::IO, w::WhaleChain) = print("$(typeof(w))($(w.S))")
Base.show(io::IO, w::WhaleChain) = write(io, "$(typeof(w))($(w.S))")

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

function mcmc!(w::WhaleChain, n::Int64, args...; show_trace=true, show_every=10)
    @warn "No data provided, sampling from the prior"
    mcmc!(w, distribute(CCD[get_dummy_ccd()]), n, args...,
        show_trace=show_trace, show_every=show_every, backtrack=false)
end

function cycle!(w::WhaleChain, D::CCDArray, args...)
    !(:ν in args) ? sample_ν!(w) : nothing
    !(:η in args) ? sample_η!(w, D) : nothing
    gibbs_sweep!(w, D)
    q_sweep!(w, D)
    wgd_sweep!(w, D)
    length(D) == 1 ? allrates!(w, D) : nothing  # better mixing when spling π
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
        idx = x.S.rindex[i]
        spl = x.samplers[:λ, idx]
        λᵢ = x[:λ, idx] + rand(spl)
        μᵢ = x[:μ, idx] + rand(spl)
        λᵢ < 0. || μᵢ < 0. ? continue : nothing
        λ_ = deepcopy(x[:λ]); μ_ = deepcopy(x[:μ])
        λ_[idx] = λᵢ; μ_[idx] = μᵢ
        p = logpdf(x, :λ=>λ_, :μ=>μ_)
        l = logpdf(WhaleModel(x.S, λ_, μ_, x[:q], x[:η]), D, i, matrix=true)
        a = p + l - x[:π] - x[:l]
        if log(rand()) < a
            x[:λ, idx] = λᵢ
            x[:μ, idx] = μᵢ
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
        idx = x.S.rindex[i]
        qidx = x.S.qindex[b]
        splr = x.samplers[:λ, idx]
        splq = x.samplers[:q, qidx]
        λᵢ = x[:λ, idx] + rand(splr)
        μᵢ = x[:μ, idx] + rand(splr)
        λᵢ < 0. || μᵢ < 0. ? continue : nothing
        qᵢ = reflect(x[:q, qidx] + rand(splq))
        λ_ = deepcopy(x[:λ]); μ_ = deepcopy(x[:μ]); q_ = deepcopy(x[:q])
        λ_[idx] = λᵢ; μ_[idx] = μᵢ; q_[qidx] = qᵢ
        p = logpdf(x, :λ=>λ_, :μ=>μ_, :q=>q_)
        l = logpdf(WhaleModel(x.S, λ_, μ_, q_, x[:η]), D, i, matrix=true)
        a = p + l - x[:π] - x[:l]
        # NOTE: no adaptation based on this proposal; because both q and rates
        if log(rand()) < a
            x[:λ, idx] = λᵢ
            x[:μ, idx] = μᵢ
            x[:q, qidx] = qᵢ
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
        p = logpdf(x, :q=>q_)
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

function allrates!(x::WhaleChain, D::CCDArray)
    spl = x.samplers[:ψ]
    λ_ = x[:λ] .+ rand(spl)
    μ_ = x[:μ] .+ rand(spl)
    if any(x -> (x <= 0.), λ_) || any(x -> (x <= 0.), μ_)
        return
    end
    p = logpdf(x, :λ=>λ_, :μ=>μ_)
    l = logpdf(WhaleModel(x.S, λ_, μ_, x[:q], x[:η]), D, matrix=true)
    a = p + l - x[:π] - x[:l]
    if log(rand()) < a
        x[:λ] = λ_
        x[:μ] = μ_
        x[:π] = p
        x[:l] = l
        spl.accepted += 1
        set_recmat!(D)
    end
    x.gen % spl.tuneinterval == 0 ? adapt!(spl, x.gen) : nothing
end

function reflect(x::Float64, a::Float64=0., b::Float64=1.)
    while !(a <= x <= b)
        x = x < a ? 2a - x : x
        x = x > b ? 2b - x : x
    end
    return x
end
