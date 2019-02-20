# Birth-death process related functions
# © Arthur Zwaenepoel - 2019

"""
    p_extinction(S::Tree, lambda::Float64, mu::Float64)
This is from ALS09. Computes for every node x in the species tree e_A(x) using a postorder traversal. This is not yet adapted for WGD nodes! (see RTA14 for that).
"""
function p_extinction(
    S::SpeciesTree, lambda::Float64, mu::Float64, q::Array{Float64}
)
    e_vtx = Dict{Int64,Float64}()
    e_arc = Dict{Int64,Float64}()

    function walk(x)
        if isleaf(S.tree, x)
            t = distance(S.tree, x, parentnode(S.tree, x))
            e_vtx[x] = 0.
            e_arc[x] = 1. - P_t(lambda, mu, t)
            return
        else
            # recurse over children
            ev = 1
            for n in childnodes(S.tree, x)
                walk(n)
                if haskey(S.wgd_index, x)
                    q_ = q[S.wgd_index[x]]
                    ev *= q_ * e_arc[n]^2 + (1-q_) * e_arc[n]
                else
                    ev *= e_arc[n]
                end
            end
            e_vtx[x] = ev
            if !isroot(S.tree, x)
                t = distance(S.tree, x, parentnode(S.tree, x))
                a = P_t(lambda, mu, t) * (1. - ev)
                b = 1. - u_t(lambda, mu, t) * ev
                e_arc[x] = 1. - (a / b)
            end
        end
    end
    walk(1)
    return e_vtx, e_arc
end


"""
    P_t(lambda::Float64, mu::Float64, t::Float64)
P(t) function from the Kendall process. The probability that a single lineage goes
extinct during a time interval t is 1-P(t).
"""
function P_t(lambda::Float64, mu::Float64, t::Float64)
    return (lambda - mu) / (lambda - mu * exp(-(lambda - mu) * t))
end


"""
    u_t(lambda::Float64, mu::Float64, t::Float64)
u_t function from the Kendall process. The probability that one lineage leaves a
descendant is P(t)*(1-u(t))*u(t)^(a-1).
"""
function u_t(lambda::Float64, mu::Float64, t::Float64)
    return lambda*(1. - exp(-(lambda-mu)*t))/(lambda - mu*exp(-(lambda-mu)*t))
end


"""
    p_transition_kendall(s::Int64, t::Float64, λ::Float64, μ::Float64)
From Rabier et al. (2014). Function only for the case where there is one gene
to start from, that is P(s|1). In that case the binomial coefficients all become
1.
"""
function p_transition_kendall(s::Int64, t::Float64, λ::Float64, μ::Float64)
    if -1e-5 < λ - μ < 1e-5  # λ == μ
        γ = ψ = λ*t/(1. + λ*t)
    else  # λ != μ
        γ = μ*(exp((λ - μ)*t) - 1.) / (λ*exp((λ - μ)*t) - μ)
        ψ = (λ/μ) * γ
    end
    if s == 1  # Single copy propagation
        return γ * ψ + (1. - γ - ψ)
    else  # duplication(s)
        return γ * ψ^(s) + ψ^(s-1) * (1. - γ - ψ)
    end
end


"""
    get_extinction_probabilities(S::SpeciesTree, slices::Slices, λ::Float64, μ::Float64)
Compute all extinction probabilities, assuming no WGDs.
"""
function get_extinction_probabilities(S::SpeciesTree, slices::Slices, λ::Float64, μ::Float64)
    #e_vtx, e_arc = p_extinction(S, λ, μ, [])
    ε = Dict{Int64,Array{Float64}}()
    for e in slices.branches
        #ε[e] = [e_vtx[e]]
        if isleaf(S.tree, e)
            ε[e] = [0.]
        else
            f, g = childnodes(S.tree, e)
            ε[e] = [ε[f][slices.slices[f]] * ε[g][slices.slices[g]]]
        end
        for i in 2:slices.slices[e]
            push!(ε[e], p_extinction_slice(
                λ, μ, slices.slice_lengths[e][i], ε[e][i-1]))
        end
    end
    return ε
end

function p_extinction_slice(λ::Float64, μ::Float64, t::Float64, ε::Float64)
    if isapprox(λ, μ, atol=1e-5)
        return 1. + (1. - ε)/(μ * (ε - 1.) * t - 1.)
    else
        return (μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ
    end
end

function p_propagation_slice(λ::Float64, μ::Float64, t::Float64, ε::Float64)
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

"""
    get_extinction_probabilities(S::SpeciesTree, slices::Slices, λ::Float64, μ::Float64,
        q::Array{Float64})
Compute all extinction probabilities with WGDs.
"""
function get_extinction_probabilities(S::SpeciesTree, slices::Slices, λ::Float64, μ::Float64,
        q::Array{Float64})
    ε = Dict{Int64,Array{Float64}}()
    for e in slices.branches  # this is in post-order
        if isleaf(S.tree, e)
            ε[e] = [0.]
        elseif haskey(S.wgd_index, e)
            qe = q[S.wgd_index[e]]
            f = childnodes(S.tree, e)[1]
            ε_wgd = ε[f][slices.slices[f]]
            ε[e] = [qe * ε_wgd^2 + (1-qe) * ε_wgd]
        else
            f, g = childnodes(S.tree, e)
            ε[e] = [ε[f][slices.slices[f]] * ε[g][slices.slices[g]]]
        end
        for i in 2:slices.slices[e]
            push!(ε[e], p_extinction_slice(
                λ, μ, slices.slice_lengths[e][i], ε[e][i-1]))
        end
    end
    return ε
end

"""
    get_propagation_probabilities(S::SpeciesTree, slices::Slices, λ::Float64, μ::Float64,
        ε::Dict{Int64,Array{Float64}})
Compute all single gene propagation probabilities.
"""
function get_propagation_probabilities(S::SpeciesTree, slices::Slices, λ::Float64, μ::Float64,
        ε::Dict{Int64,Array{Float64}})
    ϕ = Dict{Int64,Array{Float64}}()
    for e in slices.branches
        ϕ[e] = [1.]
        for i in 2:slices.slices[e]
            push!(ϕ[e], p_propagation_slice(
                λ, μ, slices.slice_lengths[e][i], ε[e][i-1]))
        end
    end
    return ϕ
end

"""
    get_extinction_probabilities(...)
Compute all extinction probabilities, with WGDs. With arbitrary
rates for each branch (given in the rate_index).
"""
function get_extinction_probabilities(S::SpeciesTree, slices::Slices, λ::Array{Float64},
        μ::Array{Float64}, q::Array{Float64}, rate_index::Dict{Int64,Int64})
    # note that we also compute the extinction probabilities in the branch
    # above the root, but in most implementations of the likelihood model we
    # don't use this.
    ε = Dict{Int64,Array{Float64}}()
    for e in slices.branches  # this is in post-order
        if isleaf(S.tree, e)
            ε[e] = [0.]
        elseif haskey(S.wgd_index, e)
            qe = q[S.wgd_index[e]]
            f = childnodes(S.tree, e)[1]
            ε_wgd = ε[f][slices.slices[f]]
            ε[e] = [qe * ε_wgd^2 + (1-qe) * ε_wgd]
        else
            f, g = childnodes(S.tree, e)
            ε[e] = [ε[f][slices.slices[f]] * ε[g][slices.slices[g]]]
        end
        if isroot(S.tree, e)
            return ε
        end
        for i in 2:slices.slices[e]
            λe = λ[rate_index[e]]
            μe = μ[rate_index[e]]
            push!(ε[e], p_extinction_slice(λe, μe, slices.slice_lengths[e][i], ε[e][i-1]))
        end
    end
end

"""
    get_propagation_probabilities(...)
Compute all single gene propagation probabilities. With arbitrary rates for
each branch (given in the rate_index).
"""
function get_propagation_probabilities(
    S::SpeciesTree, slices::Slices, λ::Array{Float64}, μ::Array{Float64},
    ε::Dict{Int64,Array{Float64}}, rate_index::Dict{Int64,Int64}
)
    ϕ = Dict{Int64,Array{Float64}}()
    for e in slices.branches[1:end-1]
        λe = λ[rate_index[e]]
        μe = μ[rate_index[e]]
        ϕ[e] = [1.]
        for i in 2:slices.slices[e]
            push!(ϕ[e], p_propagation_slice(
                λe, μe, slices.slice_lengths[e][i], ε[e][i-1]))
        end
    end
    return ϕ
end

# root survival probability recursion, just illustrative, isn't used
function root_ps(s, η, ε)
    if s == 1
        return η / (1 - (1-η) * ε)^2
    else
        return (1-η) * ps(s-1, η, ε) / (1 - (1-η) * ε)
    end
end
