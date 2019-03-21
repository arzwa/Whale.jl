#===============================================================================
ALE duplication-loss model likelihood calculation using dynamic programming.
===============================================================================#

# Main algorithm(s) ------------------------------------------------------------
"""
    whale_likelihood(S, ccd, slices, λ, μ, q, η)
Compute L(λ, μ, q| S, CCD, η). That is, the likelihood under the WHALE model.

This implementation makes use of te geometric prior on the number of lineages
at the root. The expected number of lineages at the root is 1/η. Note that by
the nature of the geometric distribution, setting η to 1 amounts to a
probability of 1 for 1 lineage at the root (so effectively assuming strictly 1
lineage at the root).

q should be of the same lenght as the WGD index in S (that is, there should be
an entry in q for every WGD). If there are no WGDs, q will be ignored, and the
same results will be acquired as with the ordinary DL model.
"""
function whale_likelihood(S::SpeciesTree, ccd::CCD, slices::Slices, λ::Float64, μ::Float64,
        q::Array{Float64}, η::Float64; one_in_both::Bool=true)
    results = initialize_dp_matrix(S.tree, ccd, slices)
    if ccd.Γ == -1 ; return results, 0. ; end  # dummy data
    valid = validate_params(λ, μ, q)
    if !(valid) ; return results, -Inf ; end

    ε = get_extinction_probabilities(S, slices, λ, μ, q)
    ϕ = get_propagation_probabilities(S, slices, λ, μ, ε)

    for e in slices.branches[1:end-1]  # skip the root branch (last one)
        wgd_node = haskey(S.wgd_index, e)
        sp_leaf = isleaf(S.tree, e)

        for γ in ccd.clades
            # if σ(γ) is not a subset of the species subtree rooted in e, skip
            # computation, as the probability is necessarily 0
            if !(ccd.species[γ] ⊆ S.clades[e]) ; continue ; end  # XXX
            leaf_γ = haskey(ccd.m3, γ)

            for i in 1:slices.slices[e]
                # beginning of branch (closest to present), speciation or leaf
                if i == 1
                    if leaf_γ && ccd.m3[γ] == e
                        results[e][γ, i] = 1.0

                    elseif !(sp_leaf || wgd_node)
                        f, g = childnodes(S.tree, e)
                        if !(leaf_γ)
                            results[e][γ, i] += Π_speciation(
                                γ, ccd.m2[γ], f, g, ccd.ccp, results)
                        end
                        results[e][γ, i] += Π_loss(γ, f, g, ε, results)

                    elseif wgd_node
                        qe = q[S.wgd_index[e]]
                        f = childnodes(S.tree, e)[1]
                        if !(leaf_γ)
                            results[e][γ, i] += Π_wgd_retention(
                                γ, ccd.m2[γ], f, qe, ccd.ccp, results)
                        end
                        results[e][γ, i] +=
                            Π_wgd_non_retention(γ, qe, f, results)
                        results[e][γ, i] +=
                            Π_wgd_loss(γ, qe, ε[f][end], f, results)
                    end

                # in the branch, propagation and duplication
                else
                    Δt = slices.slice_lengths[e][i]
                    results[e][γ, i] = ϕ[e][i] * results[e][γ, i-1]
                    if !(leaf_γ)
                        results[e][γ, i] += Π_duplication(
                            γ, ccd.m2[γ], e, i, Δt, λ, μ, ccd.ccp, results)
                    end
                end
            end
        end
    end

    # compute probabilities at the root, accounting for the prior
    compute_probabilities_at_root_prior!(S, ccd, ε, η, results)

    if one_in_both
        e, f = childnodes(S.tree, 1)
        l = probability_at_root_one_in_both(e, f, results, ε, η)
    else
        l = probability_at_root(results, ε[1][1], η)
    end
    return results, l
end


"""
    whale_likelihood_bw(S, ccd, slices, λ, μ, q, η, rate_index)
Compute L(λ, μ, q| S, CCD, η). That is, the likelihood under the WHALE model.

This implementation makes use of the geometric prior on the number of lineages
at the root. The expected number of lineages at the root is 1/η. Note that by
the nature of the geometric distribution, setting η to 1 amounts to a
probability of 1 for 1 lineage at the root (so effectively assuming strictly 1
lineage at the root).
"""
function whale_likelihood_bw(S::SpeciesTree, ccd::CCD, slices::Slices, λ::Array{Float64},
        μ::Array{Float64}, q::Array{Float64}, η::Float64, rate_index::Dict{Int64,Int64};
        one_in_both::Bool=true)
    results = initialize_dp_matrix(S.tree, ccd, slices)
    if ccd.Γ == -1 ; return results, 0. ; end  # HACK dummy data
    valid = validate_params(λ, μ, q)
    if !(valid) ; return results, -Inf ; end
    ε = get_extinction_probabilities(S, slices, λ, μ, q, rate_index)
    ϕ = get_propagation_probabilities(S, slices, λ, μ, ε, rate_index)
    l = whale_likelihood_bw!(results, S, ccd, slices, λ, μ, q, η, rate_index,
            ε, ϕ, one_in_both=one_in_both)
    return results, l
end

function whale_likelihood_bw!(results::Dict{Int64,Array{Float64,2}}, S::SpeciesTree, ccd::CCD,
        slices::Slices, λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64}, η::Float64,
        rate_index::Dict{Int64,Int64}, ε, ϕ; one_in_both::Bool=true)
    if ccd.Γ == -1 ; return 0. ; end  # HACK dummy data
    valid = validate_params(λ, μ, q)
    if !(valid) ; return results, -Inf ; end

    for e in slices.branches[1:end-1]  # skip the root branch (last one)
        wgd_node = haskey(S.wgd_index, e)
        sp_leaf = isleaf(S.tree, e)
        λe = λ[rate_index[e]]
        μe = μ[rate_index[e]]

        for γ in ccd.clades
            # if σ(γ) is not a subset of the species subtree rooted in e, skip
            # computation, as the probability is necessarily 0
            if !(ccd.species[γ] ⊆ S.clades[e]) ; continue ; end  # XXX
            leaf_γ = haskey(ccd.m3, γ)

            for i in 1:slices.slices[e]
                results[e][γ, i] = 0.
                # beginning of branch (closest to present), speciation or leaf
                if i == 1
                    if leaf_γ && ccd.m3[γ] == e
                        results[e][γ, i] = 1.0

                    elseif !(sp_leaf || wgd_node)
                        f, g = childnodes(S.tree, e)
                        if !(leaf_γ)
                            results[e][γ, i] += Π_speciation(γ, ccd.m2[γ], f, g, ccd.ccp, results)
                        end
                        results[e][γ, i] += Π_loss(γ, f, g, ε, results)

                    elseif wgd_node
                        qe = q[S.wgd_index[e]]
                        f = childnodes(S.tree, e)[1]
                        if !(leaf_γ)
                            results[e][γ, i] += Π_wgd_retention(
                                γ, ccd.m2[γ], f, qe, ccd.ccp, results)
                        end
                        results[e][γ, i] += Π_wgd_non_retention(γ, qe, f, results)
                        results[e][γ, i] += Π_wgd_loss(γ, qe, ε[f][end], f, results)
                    end

                # in the branch, propagation and duplication
                else
                    Δt = slices.slice_lengths[e][i]
                    results[e][γ, i] = ϕ[e][i] * results[e][γ, i-1]
                    if !(leaf_γ)
                        results[e][γ, i] += Π_duplication(
                            γ, ccd.m2[γ], e, i, Δt, λe, μe, ccd.ccp, results)
                    end
                end
            end
        end
    end

    # compute probabilities at the root, accounting for the prior
    compute_probabilities_at_root_prior!(S, ccd, ε, η, results)

    if one_in_both
        e, f = childnodes(S.tree, 1)
        l = probability_at_root_one_in_both(e, f, results, ε, η)
    else
        l = probability_at_root(results, ε[1][1], η)
    end
    return l
end


# Partial recompute ------------------------------------------------------------
# in the MCMC algorithm we often only change one pair of rates and want to
# recalculate the part of the matrix affected thereby, saving quite some
# computational time. Currently assuming the one in both filter is used!
# the methods assume that the computed reconciliation matrix from which to
# start is found in ccd.recmat and they store the new matrix in the tmpmat field
"""
    recompute_at_root!(S::SpeciesTree, ccd::CCD, slices::Slices,
        λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64}, η::Float64,
        ri::Dict{Int64,Int64})
Recompute the probabilities in the root, good when only η changed.
"""
function recompute_at_root!(S::SpeciesTree, ccd::CCD, slices::Slices,
        λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64}, η::Float64,
        ri::Dict{Int64,Int64})
    # Important, we may not modify the original one!
    if ccd.Γ == -1 ; return 0. ; end  # HACK dummy data
    m = deepcopy(ccd.recmat)
    # these are recomputed all the time, seems like a waste...
    ε = get_extinction_probabilities(S, slices, λ, μ, q, ri)
    compute_probabilities_at_root_prior!(S, ccd, ε, η, m)
    e, f = childnodes(S.tree, 1)
    l = probability_at_root_one_in_both(e, f, m, ε, η)
    ccd.tmpmat = m
    return l
end

# we need first to identify which parts to recompute, we can simply return a
# reduced slices object!
function get_slices_to_recompute(node::Int64, S::SpeciesTree, slices::Slices)
    branches = Int64[]; n = node
    while n != 1
        push!(branches, n)
        n = parentnode(S.tree, n)
    end
    push!(branches, 1)
    return Slices(slices.slices, slices.slice_lengths, branches)
end

"""
    partial_recompute!(node::Int64, S::SpeciesTree, ccd::CCD,
        slices::Slices, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, η::Float64, ri::Dict{Int64,Int64})
Do a partial recompute, i.e. recompute all branches that are affected by a change
in rates at the branch leading to `node`. These are all branches that are upstream
of the branch leading to `node`.
"""
function partial_recompute!(node::Int64, S::SpeciesTree, ccd::CCD,
        slices::Slices, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, η::Float64, ri::Dict{Int64,Int64})
    if ccd.Γ == -1 ; return 0. ; end  # HACK dummy data
    # Important keep the original unmodified!
    m = deepcopy(ccd.recmat)
    slices_ = get_slices_to_recompute(node, S, slices)
    ε = get_extinction_probabilities(S, slices, λ, μ, q, ri)
    ϕ = get_propagation_probabilities(S, slices, λ, μ, ε, ri)
    l = whale_likelihood_bw!(m, S, ccd, slices_, λ, μ, q, η, ri, ε, ϕ)
    ccd.tmpmat = m
    return l
end


# Helper functions -------------------------------------------------------------
"""
    initialize_dp_matrix(T::Tree, ccd::CCD, slices::Slices)
Data structure for the reconciliation matrix.
"""
function initialize_dp_matrix(T::Tree, ccd::CCD, slices::Slices)
    dp_matrix = Dict{Int64,Array{Float64,2}}()
    for n in keys(T.nodes)
        dp_matrix[n] = zeros(length(ccd.clades), slices.slices[n])
    end
    return dp_matrix
end

"""
    get_slices(T::Tree, Δt::Float64, min_n_slices::Int64; root::Float64=-1.,
        max_n_slices::Int64=1000)
Slicing function. Maybe put Δt and min_n_slices as keyword args.
"""
function get_slices(T::Tree, Δt::Float64, min_n_slices::Int64; root::Float64=-1.,
        max_n_slices::Int64=1000)
    slices = Dict{Int64,Int64}()
    slice_lengths = Dict{Int64,Array{Float64}}()
    branches = Int64[]
    function walk(node)
        if isleaf(T, node)
            l = distance(T, node, parentnode(T, node))
            n = min(max(ceil(Int64, l / Δt), min_n_slices), max_n_slices)
            slices[node] = n + 1
            slice_lengths[node] = get_slice_lengths(n , l)
            push!(branches, node)
        else
            for c in childnodes(T, node)
                walk(c)
            end
            # HACK/NOTE: This was a hack that is no longer necessary since we ---
            # introduced the prior on the root
            if !(isroot(T, node))
                l = distance(T, node, parentnode(T, node))
            else
                # l = slice_lengths[2][2] * min_n_slices # original idea
                if root < 0.
                    l = Δt*min_n_slices*5 # PsDL implementation
                else
                    l = root
                end
            end
            # --------------------------------------------------------------------
            n = max(ceil(Int64, l / Δt), min_n_slices)  # the number of slices we use
            slices[node] = n + 1
            slice_lengths[node] = get_slice_lengths(n , l)
            push!(branches, node)
        end
    end
    walk(1)
    slices[1] = 1; slice_lengths[1] = [0.]
    return Slices(slices, slice_lengths, branches)
end

# get slice lengths
function get_slice_lengths(n, l)
    slice_lengths = [[0] ; repeat([l/n], n)]
    @assert isapprox(sum(slice_lengths), l, atol= 0.00001)
    return slice_lengths
end


"""
    validate_params(λ, μ, q::Array; max_rate=10.)
    validate_params(λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64};
            max_rate::Float64=10.)
    validate_params(λ::Float64, μ::Float64, max_rate::Float64=10.)
Validate input parameters (if they are in allowed range)
"""
function validate_params(λ::Float64, μ::Float64, q::Array{Float64}; max_rate::Float64=10.)
    if !(0. < λ <= max_rate && 0. < μ <= max_rate)
        return false
    end
    if any(q .< 0.) || any(q .> 1.)
        return false
    end
    return true
end

function validate_params(λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64};
        max_rate::Float64=10.)
    if !(all(0. .< λ .<= max_rate) && all(0. .< μ .<= max_rate))
        return false
    end
    if any(q .< 0.) || any(q .> 1.)
        return false
    end
    return true
end

function validate_params(λ::Float64, μ::Float64, max_rate::Float64=10.)
    return 0. < λ <= max_rate && 0. < μ <= max_rate
end

# compute the joint probability [DEPRECATED]
function joint_probability(S::SpeciesTree, ccd::Array{CCD}, slices::Slices, λ::Float64, μ::Float64)
    joint_p = @distributed (+) for i = 1:length(ccd)
        ale_probability(S, ccd[i], slices, λ, μ)[2]
    end
    return joint_p
end

function joint_probability(S::SpeciesTree, ccd::Array{CCD}, slices::Slices, λ::Float64, μ::Float64,
        q::Array{Float64})
    joint_p = @distributed (+) for i = 1:length(ccd)
        ale_probability(S, ccd[i], slices, λ, μ, q)[2]
    end
    return joint_p
end

"""
    total_normalized_probability(results, ε; branches::Array{Int64}=[1])
Total probability, normalized as in ALE. This is the sum of the
probabilities of observing the ubiquitous clade Γ over all branches and time
slices.

The normalization factor amounts to conditioning on te fact that the family is
not extinct as in Rabier *et al.* (2014).

The branches kwarg is the set of branches over which the sum of Πe,t(Γ) (that
is the set over which e runs). By default this is set to [0], assuming
origination in the root branch.
"""
function total_normalized_probability(results, ε; branches::Array{Int64}=[1])
    total = 0
    nf = 0
    Γ = size(results[1], 1)
    for e in branches
        for t in 1:size(results[e], 2)
            total += results[e][Γ, t]
            nf +=  1 - ε[e][t]
            # I would think it should be like this?
            # total += results[e][Γ, t] / (1 - ε[e][t])
        end
    end
    if total > 0 && nf > 0
        return log(total / nf)
        # return log(total)
    else
        return -Inf
    end
end

"""
    probability_at_root_(results, ε)
Get the conditional probability of observing the ubiquitous clade Γ at the
beginning of the root branch.
"""
function probability_at_root_(results, ε)
    Γ = size(results[1], 1)
    t = size(results[1], 2)
    nf = 1 - ε[1][t]
    if results[1][Γ, t] > 0
        l = log(results[1][Γ,t] / nf)
    else
        l = -Inf
    end
end

"""
    compute_probabilities_at_root_prior!(S::SpeciesTree, ccd::CCD,
        ε::Dict{Int64,Array{Float64}}, η::Float64, results::Dict{Int64,Array{Float64,2}})
Compute L(λ,μ|S,CCD) with a prior on the number of genes at the root.
"""
function compute_probabilities_at_root_prior!(S::SpeciesTree, ccd::CCD,
        ε::Dict{Int64,Array{Float64}}, η::Float64, results::Dict{Int64,Array{Float64,2}})
    f, g = childnodes(S.tree, 1)
    ε0 = ε[1][1]
    η_ = 1.0/(1. - (1. - η) * ε0)^2
    for γ in ccd.clades
        results[1][γ, 1] = 0.
        p = 0.
        if !(haskey(ccd.m3, γ))  # not a leaf
            results[1][γ, 1] = Π_root(γ, ccd.m2[γ], η, ε0, ccd.ccp, results)
            p += Π_speciation(γ, ccd.m2[γ], f, g, ccd.ccp, results)
        end
        p += Π_loss(γ, f, g, ε, results)
        results[1][γ, 1] += η_ * p
    end
    results[1][ccd.Γ, 1] *= η
end

"""
    Π_root(γ::Int64, triples::Array{Tuple{Int64,Int64,Int64}}, η::Float64, ε0::Float64,
        ccp, results)
Recursion for the root clade in the implementation with a prior distribution on
the number of lineages at the root.
"""
function Π_root(γ::Int64, triples::Array{Tuple{Int64,Int64,Int64}}, η::Float64, ε0::Float64,
        ccp, results)
    p = 0.
    for (γ1, γ2, count) in triples
        p += ccp[(γ, γ1, γ2)] * results[1][γ1, 1] * results[1][γ2, 1]
    end
    p *= (1. - η) * (1. - (1. - η) * ε0)
    return p
end

"""
    Π_speciation(γ, triples, f, g, ε, ccp, results)
Recursion for speciation or speciation + loss events, that is, the recursion
operating between branches.
"""
function Π_speciation(γ::Int64, triples::Array{Tuple{Int64,Int64,Int64}}, f::Int64, g::Int64,
        ccp, results)
    p = 0.
    for (γ1, γ2, count) in triples
        p += ccp[(γ, γ1, γ2)] * results[f][γ1, end] * results[g][γ2, end]
        p += ccp[(γ, γ1, γ2)] * results[g][γ1, end] * results[f][γ2, end]
    end
    return p
end

"""
    Π_loss(γ, f, g, ε, results)
"""
function Π_loss(γ::Int64, f::Int64, g::Int64, ε::Dict{Int64,Array{Float64}}, results)
    return results[f][γ, end] * ε[g][end] + results[g][γ, end] * ε[f][end]
end

"""
    Π_duplication(γ, triples, e, i, Δt, λ, ccp, results)
"""
function Π_duplication(γ, triples, e, i, Δt, λ, μ, ccp, results)
    p = 0.
    for (γ1, γ2, count) in triples
        p += ccp[(γ, γ1, γ2)] * results[e][γ1, i-1] * results[e][γ2, i-1]
    end
    # p12 = p_transition_kendall(2, Δt, λ, μ)
    # return p * p12
    return p * λ * Δt #* 2 # NOTE: it would be more correct to use the BD
                            # transition probability to go from one lineage to
                            # two lineages. In practice it doesn't make a
                            # difference as long as the slice lengths (Δt) are
                            # short enough (∼ [λ/10,  λ/5]) and using teh
                            # approximation seems to counteract some bias.
end

"""
    Π_wgd_retention(γ, triples, f, q, ccp, results)
"""
function Π_wgd_retention(γ::Int64, triples::Array{Tuple{Int64,Int64,Int64}}, f::Int64,
        q::Float64, ccp, results)
    p = 0.
    for (γ1, γ2, count) in triples
        p += ccp[(γ, γ1, γ2)] * results[f][γ1, end] * results[f][γ2, end]
    end
    return p * q
end

"""
    Π_wgd_non_retention(γ, q, f, ccp, results)
"""
function Π_wgd_non_retention(γ::Int64, q::Float64, f::Int64, results)
    return (1-q) * results[f][γ, end]
end

"""
    Π_wgd_loss(γ, q, εf, f, ccp, results)
"""
function Π_wgd_loss(γ::Int64, q::Float64, εf::Float64, f::Int64, results)
    return 2 * q * εf * results[f][γ, end]
end


"""
    probability_at_root(results, ε::Float64, η::Float64)
Get the conditional probability of observing the ubiquitous clade Γ at the
root, conditioning (1) on the family being not extinct and using the geometric
prior distribution with parameter η.
"""
function probability_at_root(results, ε::Float64, η::Float64)
    Γ = size(results[1], 1)
    p_unobserved = update_extinction_prior(ε, η)
    if results[1][Γ, 1] > 0  && abs(p_unobserved) < 1
        l = log(results[1][Γ,1] / (1 - p_unobserved))
    else
        l = -Inf
    end
end

"""
    probability_at_root_one_in_both(e, f, results, ε::Dict{Int64,Array{Float64}}, η::Float64)
Get the conditional probability of observing the ubiquitous clade Γ at the
root conditioning (1) on the family being not extinct and (2) being observed in
both clades stemming from the root, using the geometric prior distribution with
parameter η.
"""
function probability_at_root_one_in_both(e, f, results, ε::Dict{Int64,Array{Float64}}, η::Float64)
    Γ = size(results[1], 1)
    p_extinct = update_extinction_prior(ε[1][1], η)
    p_extinct_left = update_extinction_prior(ε[e][end], η)
    p_extinct_right = update_extinction_prior(ε[f][end], η)
    nf = 1 - p_extinct_left - p_extinct_right + p_extinct

    if results[1][Γ, 1] > 0  && nf > 0
        l = log(results[1][Γ,1] / nf)
    else
        l = -Inf
    end
end

"""
    update_extinction_prior(ε, η)
Calculate the conditioning factor accounting for the prior distribution.
Function name signifies 'updating the extinction probabilities based on the
prior'.
"""
function update_extinction_prior(ε, η)
    return η * ε / (1 - (1 - η)*ε)
end
