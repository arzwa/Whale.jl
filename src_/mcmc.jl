# Dedicated MCMC engine for Whale
# TODO function MCMCChains.Chains(chain::WhaleChain)
# maybe make it less flexible and simpler, one model, one algorithm, one
# parameter vector?

abstract type Model end

const ParamValue = Union{Array{<:Real},Real}

const Prior = Union{Distribution,Array{<:Distribution,1},Number}

mutable struct Sampler{T}
    accepted::Int64
    tuneinterval::Int64
    kernel::Distribution
end

Sampler(σ::Real, tuning=10) = Sampler{Real}(0, tuning, Normal(0., σ))

Sampler(Σ::Array{<:Real,2}, tuning=10) = Sampler{Array{Real,1}}(
    0, tuning, MvNormal(Σ))

struct GBMModel <: Model
    ν::Prior
    η::Prior
    r::Prior
    q::Prior
    θ::Function
    λ::Function
    μ::Function
end

GBMModel(d::Dict) = GBMModel(d[:ν], d[:η], d[:r], d[:q], d[:θ], d[:λ], d[:μ])

GBMModel(st::SlicedTree) = GBMModel(fullgbmmodel(st))

fullgbmmodel(st::SlicedTree) = Dict{Symbol,Any}(
    :ν => InverseGamma(10.),
    :η => Beta(10, 2),
    :r => Exponential(0.2),
    :q => [Beta(1, 1) for i=1:nwgd(st)],
    :θ => (r) -> MvLogNormal([log(r), log(r)], [1. 0.9 ; 0.9 1.]),
    :λ => (θ, ν) -> GBM(st, θ[1], ν),
    :μ => (θ, ν) -> GBM(st, θ[2], ν))

mutable struct WhaleChain
    S::SlicedTree
    data::CCDArray
    model::Model
    state::Dict{Symbol,ParamValue}
    samplers::Dict{Symbol,Sampler}
    df::DataFrame
    gen::Int64
end

function WhaleChain(S::SlicedTree, data::CCDArray, model=GBMModel(S))
    state = rand(model)
    state[:prior] = -Inf
    state[:lhood] = -Inf
    samplers = get_defaultsamplers(model, S)
    #df = initdf(model)
    df = initdf(state)
    w = WhaleChain(S, data, model, state, samplers, df, 0)
    evaluate_prior!(w)
    evaluate_lhood!(w)
    w.df[:lhood][1] = w.state[:lhood]
    w.df[:prior][1] = w.state[:prior]
    return w
end

function get_defaultsamplers(m::GBMModel, s::SlicedTree)
    simulations = zeros(100, nwgd(s)+2*nrates(s))
    for i = 1:100
        p = rand(m)
        simulations[i,:] = [p[:λ] ; p[:μ] ; p[:q]]
    end
    Σinit = cov(simulations)
    Σinit = ((Σinit + Σinit')/2 + I)/100000
    println(Σinit)
    mvsampler = Sampler(Σinit)
    return Dict(
        :ν => Sampler(0.05),
        :η => Sampler(0.1),
        :r => Sampler(0.1),
        :θ => Sampler([0.1 0.05 ; 0.05 0.1]),
        :q => mvsampler,
        :λ => mvsampler,
        :μ => mvsampler)
end

function rand(model::GBMModel)
    state = Dict{Symbol,Any}()
    for v in [:ν, :η, :r]
        p = getfield(model, v)
        typeof(p) <: Distribution ? state[v] = rand(p) : state[v] = p
    end
    state[:q] = zeros(length(model.q))
    for i in eachindex(model.q)
        state[:q][i] = rand(model.q[i])
    end
    state[:θ] = rand(model.θ(state[:r]))
    state[:λ] = rand(model.λ(state[:θ], state[:ν]))
    state[:μ] = rand(model.λ(state[:θ], state[:ν]))
    return state
end

function get_tmp_state(chain::WhaleChain, args...)
    state = copy(chain.state)
    for (k, v) in args
        state[k] = v
    end
    return state
end

function evaluate_prior(chain::WhaleChain, model::GBMModel, args...)
    p = 0.0
    state = get_tmp_state(chain, args...)
    p += evaluate_prior(model.ν, state[:ν])
    p += evaluate_prior(model.η, state[:η])
    p += evaluate_prior(model.r, state[:r])
    for (i, d) in enumerate(model.q)
        p += evaluate_prior(d, state[:q][i])
    end
    p += evaluate_prior(model.θ(state[:r]), state[:θ])
    p += evaluate_prior(model.λ(state[:θ], state[:ν]), state[:λ])
    p += evaluate_prior(model.μ(state[:θ], state[:ν]), state[:μ])
    return p
end

function evaluate_prior!(chain::WhaleChain)
    p = evaluate_prior(chain, chain.model)
    setstate!(chain, :prior=>p)
end

evaluate_prior(chain::WhaleChain, args...) =
    evaluate_prior(chain, chain.model, args...)

evaluate_prior(d::Vector, x::Vector) = 0.

evaluate_prior(d::Number, x::Number) = 0.

evaluate_prior(d::Distribution, x) = logpdf(d, x)

evaluate_prior(d::Array{Distribution}, x::Vector) = sum(
    [logpdf(d[i],x[i]) for i in eachindex(x)])

function evaluate_lhood(chain, node, args...)
    state = get_tmp_state(chain, args...)
    return logpdf(WhaleModel(chain.S, state), chain.data, node)
end

evaluate_lhood!(chain, node=-1) = chain.state[:lhood] =
    logpdf(WhaleModel(chain.S, chain.state), chain.data, node)

# updates that only involve evluation of the prior
function sample_prior!(chain, v::Symbol)
    @assert v in [:ν, :r, :θ]
    s = getsampler(chain, v)
    v_ = getstate(chain, v) .+ rand(s.kernel)
    any(v_ .< 0.) ? (return) : nothing
    π_ = evaluate_prior(chain, v=>v_)
    ar = π_ - getstate(chain, :prior)
    if log(rand()) < ar
        setstate!(chain, v=>v_)
        setstate!(chain, :prior=>π_)
        s.accepted += 1
    end
    if chain.gen % s.tuneinterval == 0
        adapt!(s, chain.gen)
    end
end

# update for η
function sample_η!(chain)
    s = getsampler(chain, :η)
    η_ = getstate(chain, :η) + rand(s.kernel)
    π_ = evaluate_prior(chain, :η=>η_)
    l_ = evaluate_lhood(chain, chain.S.border[end], :η=>η_)
    r = l_ + π_ - (getstate(chain, :lhood) + getstate(chain, :prior))
    if log(rand()) < r
        setstate!(chain, :η=>η_)
        setstate!(chain, :prior=>π_)
        setstate!(chain, :lhood=>l_)
        s.accepted += 1
    end
    if chain.gen % s.tuneinterval == 0
        adapt!(s, chain.gen)
    end
end

# Gibbs sweep over branches (for AMWG approach)
function branch_gibbs!(chain)
    state = getstate(chain, :λ, :μ, :q)
    n = nrates(chain.S); m = 2n
    for (branch, ps) in chain.branchparams

    end
end

# Multivariate normal sampler for rates
function sample_rates!(chain)
    s = getsampler(chain, :λ)
    r = getstate(chain, :λ, :μ, :q) + rand(s.kernel)
    any(r .< 0.) ? (return) : nothing
    n = nrates(chain.S)
    π_ = evaluate_prior(chain, :λ=>r[1:n], :μ=>r[n+1:2n], :q=>r[2n+1:end])
    l_ = evaluate_lhood(chain, -1, :λ=>r[1:n], :μ=>r[n+1:2n], :q=>r[2n+1:end])
    ar = l_ + π_ - (getstate(chain, :lhood) + getstate(chain, :prior))
    if log(rand()) < ar
        setstate!(chain, :λ=>r[1:n], :μ=>r[n+1:2n], :q=>r[2n+1:end])
        setstate!(chain, :prior=>π_)
        setstate!(chain, :lhood=>l_)
        s.accepted += 1
    end
    if chain.gen % s.tuneinterval == 0
        adapt!(s, chain.gen, chain.df)
    end
end

function cycle!(chain::WhaleChain)
    chain.gen += 1
    sample_prior!(chain, :ν)
    sample_prior!(chain, :r)
    sample_prior!(chain, :θ)
    sample_η!(chain)
    sample_rates!(chain)
    d = unvector(chain.state)
    push!(chain.df, d)
    chain.gen % 10 == 0 ? log_generation(chain) : nothing
    println(chain.state[:ν], ", ", chain.state[:q][1], ", ", chain.state[:λ][4])
end

getsampler(chain, v) = chain.samplers[v]

getstate(chain, v::Symbol) = chain.state[v]

getstate(chain, args...) = vcat([chain.state[v] for v in args]...)

function setstate!(chain, args...)
    for (k, v) in args
        chain.state[k] = v
    end
end

initdf(state) = DataFrame(sort(unvector(state))...)

function unvector(state)
    d = Dict{Symbol,Real}()
    for (k, v) in state
        if typeof(v) <: AbstractArray
            for i in eachindex(v)
                d[Symbol("$k$i")] = v[i]
            end
        else
            d[k] = v
        end
    end
    return d
end

function adapt!(s::Sampler, gen::Int, df::DataFrame)
    λ = df[[x for x in sort(names(df)) if startswith(string(x), "λ")]]
    μ = df[[x for x in sort(names(df)) if startswith(string(x), "μ")]]
    q = df[[x for x in sort(names(df)) if startswith(string(x), "q")]]
    Σn = cov(Matrix(hcat(λ, μ, q)))
    s.kernel = newkernel(s.kernel, length(λ), s.accepted/s.tuneinterval, Σn)
    s.accepted = 0.0
end

function adapt!(s::Sampler, gen::Int)
    ar = s.accepted/s.tuneinterval
    s.accepted = 0
end

function newkernel(kernel::MultivariateDistribution, n, ar::Float64, Σn,
        target=2.38, β=0.2)
    d = size(Σn)[1]
    if !isposdef(Σn*target^2/d)
        return kernel
    end
    # TODO make variance for q larger
    return MixtureModel([
        MvNormal(zeros(d),Σn*target^2/d),
        MvNormal(zeros(d), [zeros(2n) .+ 0.0001; zeros(d-2n) .+ 0.01])],
        [1.0-β, β])
end

function log_generation(chain)
    @printf "%.3f," chain.state[:ν]
    @printf "%.3f," chain.state[:η]
end
