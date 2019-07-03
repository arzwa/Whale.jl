# Arthur Zwaenepoel - 2018
# Another rewrite for the Adaptive MCMC
# We take an approach where every proposal kernel is an object for itself that stores its
# associated. I think that can be a general setting to test different updates/adaptations/...
abstract type ProposalSettings end
abstract type ChainSettings end

# chain stuff
"""
    Chain(init, S, slices, rate_index; fixed_rates=Int64[])
The MCMC chain, keeping states and some data.
"""
mutable struct Chain <: ChainSettings
    lhood::Float64
    prior::Float64
    S::SpeciesTree
    slices::Slices
    ri::Dict{Int64,Int64}
    fixed_rates::Array{Int64}
    wgds::Dict{Int64,Array{Int64}}
    state::Dict{String,Array{Float64}}

    function Chain(init, S, slices, ri; fixed_rates::Array{Int64}=Int64[])
        wgds = get_wgds(S)
        new(NaN, NaN, S, slices, ri, fixed_rates, wgds, init)
    end
end

"""
    get_wgds(S)
Get an index mapping branches with WGDs to the ID s of the WGDs happning on these branches.
"""
function get_wgds(S)
    wgds = Dict{Int64,Array{Int64}}()
    for (node, v) in S.wgd_index
        x = Whale.non_wgd_child(S, node)
        haskey(wgds, x) ? push!(wgds[x], v) : wgds[x] = [v]
    end
    return wgds
end

# Proposal & algorithm stuff
"""
    UvProposal(σ)
    UvProposal(a, b)
A single univariate proposal kernel.
"""
mutable struct UvProposal
    kernel::Distribution{Univariate,Continuous}
    accepted::Real

    UvProposal(σ) = new(Normal(0., σ), 0.)
    UvProposal(a, b) = new(Uniform(a, b), 0.)
end

"""
    UvAdaptiveProposals(S; σθ=0.02, σq=0.2, σν=0.2, ση=0.2)
Univariate adaptive proposal kernels.
"""
struct UvAdaptiveProposals <: ProposalSettings
    λ::Dict{Int64,UvProposal}
    μ::Dict{Int64,UvProposal}
    q::Dict{Int64,UvProposal}
    ν::UvProposal
    η::UvProposal
    ψ::UvProposal  # for the all_branches update

    function UvAdaptiveProposals(S; σθ=0.1, σq=0.2, σν=0.2, ση=0.1, σψ=0.1)
        λ = Dict(n => UvProposal(σθ) for n in keys(S.tree.nodes) if !haskey(S.wgd_index, n))
        μ = Dict(n => UvProposal(σθ) for n in keys(S.tree.nodes) if !haskey(S.wgd_index, n))
        q = Dict(n => UvProposal(σq) for n in values(S.wgd_index))
        new(λ, μ, q, UvProposal(σν), UvProposal(ση), UvProposal(σψ))
    end
end

"""
    mcmc!(ccd, chain, proposal, prior, ngen, freq; fname="")
Basic (non-adaptive) MCMC sampler.
"""
function mcmc!(ccd, chain, prop, prior, ngen, freq; fname::String="")
    @info "Distributing over $(length(workers())) workers"
    D = distribute(ccd)
    Whale.evaluate_prior!(chain, prior)
    Whale.evaluate_lhood!(D, chain, prior)
    for i in 1:ngen
        cycle!(D, chain, prop, prior)
        mod(i, freq) == 0 || i == 1 ? Whale.log_generation(chain, i, fname=fname) : nothing
    end
end

"""
    amcmc!(ccd, chain, proposal, prior, ngen, freq; fname="", bsize=25)
Adaptive MCMC sampler.
"""
function amcmc!(ccd, chain, prop, prior, ngen, freq; fname::String="", bsize::Int64=50)
    @info "Distributing over $(length(workers())) workers"
    D = distribute(ccd)
    Whale.evaluate_prior!(chain, prior)
    Whale.evaluate_lhood!(D, chain, prior)
    batch = 1
    for i in 1:ngen
        cycle!(D, chain, prop, prior)
        mod(i, freq) == 0 || i == 1 ? Whale.log_generation(chain, i, fname=fname) : nothing
        if i % bsize == 0
            adapt!(prop, batch, bsize)
            batch += 1
        end
    end
end

function cycle!(D, chain, prop, prior::GeometricBrownianMotion)
    if !(prior.fixed_ν)
        ν_update!(chain, prop, prior)
    end
    if !(prior.fixed_η)
        η_update!(D, chain, prop, prior)
    end
    θ_update!(D, chain, prop, prior)
    q_update!(D, chain, prop, prior)
    branch_update!(D, chain, prop, prior)
    θ_update_all!(D, chain, prop, prior)
end

function cycle!(D, chain, prop, prior::IidRates)
    if !(prior.fixed_η)
        η_update!(D, chain, prop, prior)
    end
    r_update!(chain, prop, prior)
    θ_update!(D, chain, prop, prior)
    q_update!(D, chain, prop, prior)
    branch_update!(D, chain, prop, prior)
    θ_update_all!(D, chain, prop, prior)
end

# updates
"""
    r_update(chain, prop, prior)
Updates the mean of the iid rates prior. Note that this parameter is stored in
λ[1] and μ[1] respectively.
"""
function r_update!(chain::ChainSettings, prop::ProposalSettings, prior::IidRates)
    λ = chain.state["λ"]  # we store the means of the iid rates priors in λ₁ and μ₁
    μ = chain.state["μ"]  # since these have an analogous role as λ₁ and μ₁ in the gbm prior
    λ_ = [λ[1] + rand(prop.λ[1].kernel) ; λ[2:end]]
    μ_ = [μ[1] + rand(prop.μ[1].kernel) ; μ[2:end]]
    if λ_[1] <= 0. || μ_[1] <= 0.
        return
    end
    # likelihood isn't changed
    π = Whale.evaluate_prior(chain.S, λ_, μ_, chain.state, prior)
    r = π - chain.prior
    if log(rand()) < r
        update_chain!(chain, 1, λ_[1], μ_[1], π, chain.lhood)
        update_prop!(prop, :λ, 1)
        update_prop!(prop, :μ, 1)
    end
end

"""
    θ_update!(D::DArray{CCD}, chain, proposal, prior)
Rates [θ = (λᵢ, μᵢ)] update. Iterates over all branches and updates pairs of rates
based on their individual proposal kernels.
"""
function θ_update!(D, chain, prop, prior)
    for i in 1:length(chain.state["λ"])
        i in chain.fixed_rates ? continue : nothing
        state = chain.state
        λ = state["λ"]
        μ = state["μ"]
        q = q_ = state["q"]
        λᵢ = state["λ"][i] + rand(prop.λ[i].kernel)
        μᵢ = state["μ"][i] + rand(prop.μ[i].kernel)
        λᵢ <= 0. || μᵢ <= 0. ? continue : nothing
        λ_ = [λ[1:i-1] ; λᵢ ; λ[i+1:end]]
        μ_ = [μ[1:i-1] ; μᵢ ; μ[i+1:end]]

        # get the prior for the rates
        π = Whale.evaluate_prior(chain.S, λ_, μ_, q_, state, prior)

        # get the likelihood by partial re-evaluation of the dynamic programming matrix
        l = Whale.evaluate_lhood!(D, λ_, μ_, q_, i, chain, prior)

        # log acceptance probability
        r = l + π - (chain.lhood + chain.prior)
        if log(rand()) < r
            update_chain!(chain, i, λᵢ, μᵢ, π, l)
            update_prop!(prop, :λ, i)
            update_prop!(prop, :μ, i)
            Whale.set_recmat!(D)
        end
    end
end

"""
    q_update!(D, chain, proposal, prior)
"""
function q_update!(D::DArray{CCD}, chain, prop, prior)
    for (n, i) in chain.S.wgd_index
        state = chain.state ; q = chain.state["q"]
        qᵢ = Whale.reflect(state["q"][i] + rand(prop.q[i].kernel))

        q_ = [q[1:i-1] ; qᵢ ; q[i+1:end]]
        π = Whale.evaluate_prior(chain.S, state["λ"], state["μ"], q_, state, prior)
        l = Whale.evaluate_lhood!(D, state["λ"], state["μ"], q_, n, chain, prior)
        # XXX: 23/12/2018 the node was incorrect! we gave i instead of n to the lhood function!

        r = l + π - (chain.lhood + chain.prior)
        if log(rand()) < r
            chain.state["q"][i] = qᵢ
            chain.prior = π
            chain.lhood = l
            update_prop!(prop, :q, i)
            Whale.set_recmat!(D)
        end
    end
end

"""
    ν_update!(chain, prop, prior::GeometricBrownianMotion)
Update of the ν parameter governing the correlation fo rates in the GBM prior.
Does not evaluate the likelihood so it's very cheap.
"""
function ν_update!(chain, prop, prior::GeometricBrownianMotion)
    ν_ = chain.state["ν"][1] + rand(prop.ν.kernel)
    π = Whale.evaluate_prior(chain, ν_, prior)
    r = π - chain.prior
    if log(rand()) < r
        chain.state["ν"][1] = ν_
        chain.prior = π
        update_prop!(prop, :ν)
    end
end

"""
    η_update!(chain, prop, prior::GeometricBrownianMotion)
Update of the η parameter.
"""
function η_update!(D, chain, prop, prior::PriorSettings)
    η_ = chain.state["η"][1] + rand(prop.η.kernel)
    π_ = logpdf(prior.prior_η, chain.state["η"][1])
    π  = logpdf(prior.prior_η, η_)
    l  = Whale.evaluate_lhood!(D, η_, chain, prior)
    r = l + π - (chain.lhood + π_)
    if log(rand()) < r
        chain.state["η"][1] = η_
        Whale.evaluate_prior!(chain, prior)
        update_prop!(prop, :η)
    end
end

"""
    branch_update!(D::DArray{CCD}, chain, proposal, prior)
Update λᵢ, μᵢ, 𝒒ᵢ jointly ∀ branch i with one or more WGDs with retention
rates 𝒒ᵢ based on their respective individual proposal kernels. There is no
adaptation for this proposal, since the proposal kernels are adapted based on
the target in the `θ_update!()` step.
"""
function branch_update!(D, chain, prop, prior)
    for (branch, wgds) in chain.wgds
        state = chain.state
        λ = state["λ"]
        μ = state["μ"]
        q = state["q"]
        λᵢ = state["λ"][branch] + rand(prop.λ[branch].kernel)
        μᵢ = state["μ"][branch] + rand(prop.μ[branch].kernel)
        λᵢ <= 0 || μᵢ <= 0 ? continue : nothing
        λ_ = [λ[1:branch-1] ; λᵢ ; λ[branch+1:end]]
        μ_ = [μ[1:branch-1] ; μᵢ ; μ[branch+1:end]]
        q_ = [i in wgds ? reflect(q[i] + rand(prop.q[i].kernel)) :
            q[i] for i in 1:length(q)]
        # XXX also a fault here, there was `q[i] in wgds ?` instead of `i in wgd ?`

        # get the prior for the rates
        π = Whale.evaluate_prior(chain.S, λ_, μ_, q_, state, prior)

        # get the likelihood by partial re-evaluation of the dynamic programming matrix
        l = Whale.evaluate_lhood!(D, λ_, μ_, q_, branch, chain, prior)

        # log acceptance probability
        r = l + π - (chain.lhood + chain.prior)
        if log(rand()) < r
            @debug "Branch update accepted!"
            update_chain!(chain, branch, λᵢ, μᵢ, π, l)
            chain.state["q"][:] = q_[:]
            Whale.set_recmat!(D)
        end
    end
end

"""
    θ_update_all!(D::DArray{CCD}, chain, proposal, prior)
Update all λ and μ jointly, with one update for all λ's and an independent update
from the same kernel (ψ kernel) for all μ's.
"""
function θ_update_all!(D, chain, prop, prior)
    λ_ = chain.state["λ"] .+ rand(prop.ψ.kernel)
    μ_ = chain.state["μ"] .+ rand(prop.ψ.kernel)
    for i in chain.fixed_rates
        λ_[i] = chain.state["λ"][i]
        μ_[i] = chain.state["μ"][i]
    end
    q_ = chain.state["q"]
    if any(x -> (x <= 0.), λ_) || any(x -> (x <= 0.), μ_) || any(x -> (x < 0. || x > 1.), q_)
        return
    end
    π = Whale.evaluate_prior(chain.S, λ_, μ_, q_, chain.state, prior)
    l = Whale.evaluate_lhood!(D, λ_, μ_, q_, chain, prior)
    r = l + π - (chain.lhood + chain.prior)
    if log(rand()) < r
        @debug "All branch update accepted!"
        chain.state["λ"] = λ_
        chain.state["μ"] = μ_
        chain.state["q"] = q_
        chain.prior = π
        chain.lhood = l
        update_prop!(prop, :ψ)
        Whale.set_recmat!(D)
    end
end

function update_chain!(chain, branch::Int64, λ::Float64, μ::Float64, π::Float64, l::Float64)
    chain.state["λ"][branch] = λ
    chain.state["μ"][branch] = μ
    chain.prior = π
    chain.lhood = l
end

update_prop!(prop, field::Symbol) = getfield(prop, field).accepted += 1
update_prop!(prop, field::Symbol, branch::Int64) = getfield(prop, field)[branch].accepted += 1

function adapt!(prop, batch, bsize;
        target=0.25, bound=5., δmax=0.25, branch=[:λ, :μ, :q], hyper=[:ψ, :η, :ν])
    δn = min(δmax, 1/√batch)
    # branch-specific (dict) kernels
    for d in branch
        for (branch, p) in getfield(prop, d)
            @printf "%s%3d" d branch
            α = p.accepted / bsize
            p.kernel = adapt_kernel(p.kernel, α, δn, target, bound)
            p.accepted = 0
        end
    end
    # hyperparameter/global kernels
    for d in hyper
        @printf "%s   " d
        p = getfield(prop, d)
        α = p.accepted / bsize
        p.kernel = adapt_kernel(p.kernel, α, δn, target, bound)
        p.accepted = 0
    end
end

function adapt_kernel(kernel::Distributions.Normal{Float64}, α, δn, target, bound)
    lσ = α > target ? log(kernel.σ) + δn : log(kernel.σ) - δn
    lσ = abs(lσ) > bound ? sign(lσ) * bound : lσ
    @info "α = $(@sprintf("%.2f",α)): $(@sprintf("%.4f",kernel.σ)) → $(@sprintf("%.4f",exp(lσ)))"
    return Normal(0., exp(lσ))
end

# likelihood evaluation DArray
# lower level functions implemented in `parallel.jl`
"""
    evaluate_lhood!(D::DArray, chain::ChainSettings, prior::PriorSettings)
Evaluate the likelihood for the current state of the chain, stores it in
the recmat field! Modifies the chain.lhood field.
"""
function evaluate_lhood!(D::DArray, chain::ChainSettings, prior::PriorSettings)
    L = evaluate_lhood!(D, chain.S, chain.slices, chain.state["λ"], chain.state["μ"],
            chain.state["q"], chain.state["η"][1], chain.ri)
    chain.lhood = L
    set_recmat!(D)
end

"""
    evaluate_lhood!(D::DArray, η::Float64, chain::ChainSettings, prior::PriorSettings)
    evaluate_lhood!(D::DArray, λ::Array{Float64}, μ::Array{Float64},
            node::Int64, chain::ChainSettings, prior::PriorSettings)
    evaluate_lhood!(D::DArray, λ::Array{Float64}, μ::Array{Float64},
            chain::ChainSettings, prior::PriorSettings)
    evaluate_lhood!(D::DArray, q::Array{Float64}, node::Int64,
            chain::ChainSettings, prior::PriorSettings)
    evaluate_lhood!(D::DArray, λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64},
            node::Int64, chain::ChainSettings, prior::PriorSettings)
    evaluate_lhood!(D::DArray, λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64},
            chain::ChainSettings, prior::PriorSettings)
Evaluate the likelihood when only rates have changed. Does not modify the chain.
"""
function evaluate_lhood!(D::DArray, η::Float64, chain::ChainSettings,
        prior::PriorSettings)
    return evaluate_root!(D, chain.S, chain.slices,  chain.state["λ"],
        chain.state["μ"], chain.state["q"], η, chain.ri)
end

function evaluate_lhood!(D::DArray, λ::Array{Float64}, μ::Array{Float64},
        node::Int64, chain::ChainSettings, prior::PriorSettings)
    return evaluate_partial!(D, node, chain.S, chain.slices, λ, μ,
        chain.state["q"], chain.state["η"][1], chain.ri)
end

function evaluate_lhood!(D::DArray, λ::Array{Float64}, μ::Array{Float64},
        chain::ChainSettings, prior::PriorSettings)
    return evaluate_lhood!(D, chain.S, chain.slices, λ, μ, chain.state["q"],
        chain.state["η"][1], chain.ri)
end

function evaluate_lhood!(D::DArray, q::Array{Float64}, node::Int64,
        chain::ChainSettings, prior::PriorSettings)
    return evaluate_partial!(D, node, chain.S, chain.slices, chain.state["λ"],
        chain.state["μ"], q, chain.state["η"][1], chain.ri)
end

function evaluate_lhood!(D::DArray, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, node::Int64, chain::ChainSettings,
        prior::PriorSettings)
    return evaluate_partial!(D, node, chain.S, chain.slices, λ, μ, q,
        chain.state["η"][1], chain.ri)
end

function evaluate_lhood!(D::DArray, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, chain::ChainSettings, prior::PriorSettings)
    return evaluate_lhood!(D, chain.S, chain.slices, λ, μ, q,
        chain.state["η"][1], chain.ri)
end

# logging
function log_generation(chain::ChainSettings)
    state = chain.state
    print(join([@sprintf "%7.5f" x for x in state["λ"][2:4]], ",") * ",…,")
    print(join([@sprintf "%7.5f" x for x in state["μ"][2:4]], ",") * ",…,")
    print(join([@sprintf "%7.5f" x for x in state["q"]], ",") * ",…,")
    haskey(state, "ν") ? (@printf "%7.5f," state["ν"][1]) : nothing
    @printf "%7.5f," state["η"][1]
    @printf "%7.5f,%7.5f\n" chain.prior chain.lhood
    flush(stdout)
end

function log_header(f, nr, nwgd, state)
    write(f, ",")
    write(f, join(["l"*string(i) for i in 1:nr], ","), ",")
    write(f, join(["m"*string(i) for i in 1:nr], ","), ",")
    nwgd > 0 ? write(f, join(["q"*string(i) for i in 1:nwgd], ",") * ",") : nothing
    haskey(state, "ν") ? write(f, "nu,") : nothing
    write(f, "eta,prior,lhood\n")
end

function log_generation(chain, gen; fname::String="")
    log_generation(chain)
    if length(fname) == 0 ; return ; end
    open(fname, "a+") do f
        state = chain.state
        nwgd = length(state["q"])
        nr = length(state["λ"])
        gen == 1 ? log_header(f, nr, nwgd, state) : nothing
        write(f, "$gen,")
        write(f, join([@sprintf "%7.5f" x for x in state["λ"]], ",") * ",")
        write(f, join([@sprintf "%7.5f" x for x in state["μ"]], ",") * ",")
        nwgd > 0 ? write(f, join([@sprintf "%7.5f" x for x in state["q"]], ",") * ",") : nothing
        haskey(state, "ν") ? write(f, string(state["ν"][1]), ",") : nothing
        write(f, string(state["η"][1]), ",")
        write(f, string(chain.prior), ",", string(chain.lhood), "\n")
    end
end

# reflection
function reflect(x::Float64, a::Float64, b::Float64)
    while !(a <= x <= b)
        x = x < a ? 2a - x : x
        x = x > b ? 2b - x : x
    end
    return x
end

reflect(x)::Float64 = reflect(x, 0., 1.)

# show the chain
function Base.show(io::IO, chain::ChainSettings)
    println(io, "Metropolis-Hastings MCMC chain")
    println(io, "state:")
    println(io, " .. ̄λ = ", sum(chain.state["λ"])/length(chain.state["λ"]))
    println(io, " .. ̄̄μ = ", sum(chain.state["μ"])/length(chain.state["μ"]))
    println(io, " .. q = ", chain.state["q"])
    println(io, " .. η = ", chain.state["η"][1])
    if haskey(chain.state, "ν")
        println(io, " .. ν = ", chain.state["ν"][1])
    end
    println(io, "prior: ", chain.prior)
    println(io, "lhood: ", chain.lhood)
end
