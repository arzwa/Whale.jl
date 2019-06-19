mutable struct AMWG{T} <: InferenceAlgorithm
    n_iters::Int
    proposals::Dict{Symbol,Any}
    space::Set{T}
end

function AMWG(n_iters::Int, space...)
    new_space = Set()
    proposals = Dict{Symbol,Any}()

    # parse random variables with their hypothetical proposal
    for element in space
        if isa(element, Symbol)
            push!(new_space, element)
        else
            @assert isa(element[1], Symbol) "[MH] ($element[1]) should be a Symbol. For proposal, use the syntax MH(N, (:m, (x) -> Normal(x, 0.1)))"
            push!(new_space, element[1])
            proposals[element[1]] = element[2]
        end
    end
    set = Set(new_space)
    AMWG{eltype(set)}(n_iters, proposals, set)
end

function Sampler(alg::AMWG, model::Model, s::Selector)
    alg_str = "AMWG"

    # Sanity check for space
    if (s.tag == :default) && !isempty(alg.space)
        @assert issubset(Set(get_pvars(model)), alg.space) "[$alg_str] symbols specified to samplers ($alg.space) doesn't cover the model parameters ($(Set(get_pvars(model))))"
        if Set(get_pvars(model)) != alg.space
            warn("[$alg_str] extra parameters specified by samplers don't exist in model: $(setdiff(alg.space, Set(get_pvars(model))))")
        end
    end

    info = Dict{Symbol, Any}()
    info[:proposal_ratio] = 0.0
    info[:prior_prob] = 0.0
    info[:violating_support] = false

    return Sampler(alg, info, s)
end

function propose(model, spl::Sampler{<:AMWG}, vi::VarInfo)
    spl.info[:proposal_ratio] = 0.0
    spl.info[:prior_prob] = 0.0
    spl.info[:violating_support] = false
    return runmodel!(model, vi ,spl)
end

function step(model, spl::Sampler{<:AMWG}, vi::VarInfo, is_first::Val{true})
    return vi, true
end

function step(model, spl::Sampler{<:AMWG}, vi::VarInfo, is_first::Val{false})
    if spl.selector.tag != :default # Recompute joint in logp
        runmodel!(model, vi)
    end
    old_θ = copy(vi[spl])
    old_logp = getlogp(vi)

    Turing.DEBUG && @debug "Propose new parameters from proposals..."
    propose(model, spl, vi)

    Turing.DEBUG && @debug "computing accept rate α..."
    is_accept, _ = mh_accept(-old_logp, -getlogp(vi), spl.info[:proposal_ratio])

    Turing.DEBUG && @debug "decide wether to accept..."
    if is_accept && !spl.info[:violating_support]  # accepted
        is_accept = true
    else                      # rejected
        is_accept = false
        vi[spl] = old_θ         # reset Θ
        setlogp!(vi, old_logp)  # reset logp
    end

    return vi, is_accept
end

# From Turing ==================================================================


function sample(model::Model, alg::MH;
    save_state=false,         # flag for state saving
    resume_from=nothing,      # chain to continue
    reuse_spl_n=0,            # flag for spl re-using
    )

    spl = reuse_spl_n > 0 ?
    resume_from.info[:spl] :
    Sampler(alg, model)
    if resume_from != nothing
        spl.selector = resume_from.info[:spl].selector
    end
    alg_str = "MH"

    # Initialization
    time_total = 0.0
    n = reuse_spl_n > 0 ? reuse_spl_n : alg.n_iters
    samples = Array{Sample}(undef, n)
    weight = 1 / n
    for i = 1:n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    vi = if resume_from == nothing
        VarInfo(model)
    else
        resume_from.info[:vi]
    end

    if spl.selector.tag == :default
        runmodel!(model, vi, spl)
    end

    # MH steps
    accept_his = Bool[]
    PROGRESS[] && (spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0))
    for i = 1:n
        Turing.DEBUG && @debug "$alg_str stepping..."

        time_elapsed = @elapsed vi, is_accept = step(model, spl, vi, Val(i == 1))
        time_total += time_elapsed

        if is_accept # accepted => store the new predcits
            samples[i].value = Sample(vi, spl).value
        else         # rejected => store the previous predcits
            samples[i] = samples[i - 1]
        end

        samples[i].value[:elapsed] = time_elapsed
        push!(accept_his, is_accept)

        PROGRESS[] && (ProgressMeter.next!(spl.info[:progress]))
    end

    println("[$alg_str] Finished with")
    println("  Running time        = $time_total;")
    accept_rate = sum(accept_his) / n  # calculate the accept rate
    println("  Accept rate         = $accept_rate;")

    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.info[:samples]...)
    end
    c = Chain(0.0, samples)       # wrap the result by Chain
    if save_state               # save state
        c = save(c, spl, model, vi, samples)
    end

    c
end
