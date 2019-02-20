
# Earlier implementation ===========================================================================
"""
    backtrack()

Stochastic backtracking along the reconciliation matrix to sample trees
proportional to their probability. This is for the algorithm with prior at the
root! This operates recursively.

Notes
=====
- the prior is already in the computed matrix!
- this effectively assumes that Γ is reconciled to the root, i.e. minimal
filtering is "one in both clades".
"""
function backtrack(matrix::Dict{Int64,Array{Float64,2}}, S::SpeciesTree, ccd::CCD,
        slices::Slices, λ::Float64, q::Array{Float64}, ε::Dict{Int64,Array{Float64}},
        ϕ::Dict{Int64,Array{Float64}}, η::Float64)
    p = matrix[1][end, 1]  # total unnormalized probability at the root
    Γ = size(matrix[1], 1)
    R = RecTree(
            Tree(), Dict{Int64,String}(), Dict{Int64,Int64}(),
            Dict{Int64,Int64}(), Dict{Int64,String}()
        )

    # choose a split of Γ to start from
    r = rand() * p
    e = 1 ; i = 1
    # this should be restricted to the root branch (e = 1) for most filtering
    # procedures! That is, we only allow Γ to be reconciled in branch 1.
    #for e_ in slices.branches
    e_ = 1
    for i_ in 1:slices.slices[e]
        r -= matrix[e][Γ, i]
        if r < 0
            e = e_ ; i = i_
            break
        end
    end
    #if r < 0; break ; end
    #end

    # how to backtrack the root?
    # backtrack_root!(...)  # this will start the full recursion instead of
    # backtrack_!(...)

    backtrack_root!(-1, R, matrix, e, Γ, i, S, ccd, slices, ε, ϕ, λ, η, q)
    correct_branch_lengths!(R, S)
    return R
end


"""
    _backtrack(e, γ, i)

Backtracking helper function.
"""
function _backtrack!(
        node::Int64, R::RecTree, matrix, e::Int64, γ::Int64, i::Int64,
        S::SpeciesTree, ccd::CCD, slices::Slices, ε, ϕ, λ, q
    )
    if i == 1
        #if length(R.tree.nodes) > 1 ; node_labeled_tree(R.tree) ; end
        return backtrack_inter!(
            node, R, matrix, e, γ, S, ccd, slices, ε, ϕ, λ, q)
    else
        return backtrack_intra!(
            node, R, matrix, e, γ, i, S, ccd, slices, ε, ϕ, λ, q)
    end
end


function backtrack_root!(node::Int64, R::RecTree, matrix, e::Int64, γ::Int64, i::Int64,
        S::SpeciesTree, ccd::CCD, slices::Slices, ε, ϕ, λ, η, q)
    # start by drawing random result
    p = matrix[1][γ, 1]
    r = rand() * p

    # root is always bifurcating
    f, g = childnodes(returnS.tree, e)

    η_ = (1. - (1. - η) * ε[1][1])
    if !haskey(ccd.m3, γ)
        for (γ1, γ2, count) in ccd.m2[γ]
            ccp = ccd.ccp[(γ, γ1, γ2)]

            # either stay in root
            r -= ccp * matrix[1][γ1, 1] * matrix[1][γ2, 1] * η_ * (1. - η)
            if (r < 0)
                n = add_duplication_node!(R, e, node, γ, ccd.branch_lengths[γ])
                backtrack_root!(n, R, matrix, e, γ1, 1, S, ccd, slices, ε, ϕ, λ, η, q)
                backtrack_root!(n, R, matrix, e, γ2, 1, S, ccd, slices, ε, ϕ, λ, η, q)
                return
            end

            # or speciate
            r -= ccp * matrix[f][γ1, end] * matrix[g][γ2, end] * (1/η_^2)
            if (r < 0)
                sp_node = add_speciation_node!(R, e, node, γ, ccd.branch_lengths[γ])
                _backtrack!(sp_node, R, matrix, f, γ1, slices.slices[f], S, ccd, slices, ε, ϕ, λ, q)
                _backtrack!(sp_node, R, matrix, g, γ2, slices.slices[g], S, ccd, slices, ε, ϕ, λ, q)
                return
            end

            r -= ccp * matrix[g][γ1, end] * matrix[f][γ2, end] * (1/η_^2)
            if (r < 0)
                sp_node = add_speciation_node!(R, e, node, γ, ccd.branch_lengths[γ])
                _backtrack!(sp_node, R, matrix, f, γ2, slices.slices[f], S, ccd, slices, ε, ϕ, λ, q)
                _backtrack!(sp_node, R, matrix, g, γ1, slices.slices[g], S, ccd, slices, ε, ϕ, λ, q)
                return
            end
        end
    end

    # or have a speciation + loss
    r -= matrix[f][γ, end] * ε[g][end] * (1/η_^2)
    if (r < 0)
        # distance to speciation node should be
        sp_node = add_speciation_node!(R, e, node, γ, ccd.branch_lengths[γ])
        add_loss_node!(R, g, sp_node)
        _backtrack!(sp_node, R, matrix, f, γ, slices.slices[f], S, ccd, slices, ε, ϕ, λ, q)
        return
    end

    r -= matrix[g][γ, end] * ε[f][end] * (1/η_^2)
    if (r < 0)
        sp_node = add_speciation_node!(R, e, node, γ, ccd.branch_lengths[γ])
        add_loss_node!(R, f, sp_node)
        _backtrack!(sp_node, R, matrix, g, γ, slices.slices[g], S, ccd, slices, ε, ϕ, λ, q)
        return
    end
end


"""
    backtrack_intra(e, γ)

Backtracking function for speciation/WGD nodes.
"""
function backtrack_intra!(
        node, R::RecTree, matrix, e::Int64, γ::Int64, i::Int64,
        S::SpeciesTree, ccd::CCD, slices::Slices, ε, ϕ, λ, q
    )
    leaf_σ = isleaf(S.tree, e)
    leaf_γ = haskey(ccd.m3, γ)

    # leaf
    if leaf_γ
        return _backtrack!(node, R, matrix, e, γ, i-1, S, ccd, slices, ε, ϕ, λ, q)
    end

    p = matrix[e][γ, i]
    r = rand() * p

    # propagation
    r -= ϕ[e][i] * matrix[e][γ, i-1]
    if r < 0
        return _backtrack!(node, R, matrix, e, γ, i-1, S, ccd, slices, ε, ϕ, λ, q)
    end

    # duplication
    Δt = slices.slice_lengths[e][i]
    for (γ1, γ2, count) in ccd.m2[γ]
        ccp = ccd.ccp[(γ, γ1, γ2)]
        # here the factor 2 or BDP issue comes up again
        r -= ccp * matrix[e][γ1, i-1] * matrix[e][γ2, i-1] * λ * Δt
        # is the Δt necessary?
        if r < 0
            dnode = add_duplication_node!(R, e, node, γ, ccd.branch_lengths[γ])
            _backtrack!(dnode, R, matrix, e, γ1, i-1, S, ccd, slices, ε, ϕ, λ, q)
            _backtrack!(dnode, R, matrix, e, γ2, i-1, S, ccd, slices, ε, ϕ, λ, q)
            return
        end
    end

    if r >= 0
        error("Backtracking failed (intra)! r = ", r)
    end
    return
end


"""
    backtrack_inter!(e, γ)

Backtracking function for speciation/WGD nodes. This should be split in
subroutines
"""
function backtrack_inter!(node, R::RecTree, matrix, e::Int64, γ::Int64,
        S::SpeciesTree, ccd::CCD, slices::Slices, ε, ϕ, λ, q)
    leaf_σ = isleaf(S.tree, e)
    leaf_γ = haskey(ccd.m3, γ)

    # leaf, end of recursion
    if leaf_σ && leaf_γ
        add_leaf_node!(R, e, node, γ, ccd.branch_lengths[γ], ccd.leaves[γ])
        return node
    end

    p = matrix[e][γ, 1]
    r = rand() * p
    children = childnodes(S.tree, e)

    # speciation/speciation + loss
    if length(children) == 2
        # This should be in a subroutine ---------------------------------------
        f, g = children
        # right loss
        r -= matrix[f][γ, end] * ε[g][end]
        if r < 0
            sp_node = add_speciation_node!(R, e, node, γ, ccd.branch_lengths[γ])
            add_loss_node!(R, g, sp_node)
            return _backtrack!(sp_node, R, matrix, f, γ, slices.slices[f], S,
                ccd, slices, ε, ϕ, λ, q)
        end

        # left loss
        r -= matrix[g][γ, end] * ε[f][end]
        if r < 0
            sp_node = add_speciation_node!(R, e, node, γ, ccd.branch_lengths[γ])
            add_loss_node!(R, f, sp_node)
            return _backtrack!(sp_node, R, matrix, g, γ, slices.slices[g], S,
                ccd, slices, ε, ϕ, λ, q)
        end

        # speciation
        for (γ1, γ2, count) in ccd.m2[γ]
            ccp = ccd.ccp[(γ, γ1, γ2)]
            r -= ccp * matrix[f][γ1, end] * matrix[g][γ2, end]
            if r < 0
                sp_node = add_speciation_node!(
                    R, e, node, γ, ccd.branch_lengths[γ])
                _backtrack!(
                    sp_node, R, matrix, f, γ1, slices.slices[f], S, ccd,
                    slices, ε, ϕ, λ, q)
                _backtrack!(
                    sp_node, R, matrix, g, γ2, slices.slices[g], S, ccd,
                    slices, ε, ϕ, λ, q)
                return
            end
            r -= ccp * matrix[g][γ1, end] * matrix[f][γ2, end]
            if r < 0
                sp_node = add_speciation_node!(
                    R, e, node, γ, ccd.branch_lengths[γ])
                _backtrack!(
                    sp_node, R, matrix, g, γ1, slices.slices[g], S, ccd,
                    slices, ε, ϕ, λ, q)
                _backtrack!(
                    sp_node, R, matrix, f, γ2, slices.slices[f], S, ccd,
                    slices, ε, ϕ, λ, q)
                return
            end
        end
        if r >=0
            println(join([node, e, γ], ", "))
            error("Backtracking failed (inter), r = ", r)
        end

    # wgd cases, currently q not correctly implemented
    else
        f = children[1]
        wgd_node = add_wgd_node!(R, e, node, γ, ccd.branch_lengths[γ])
        q_ = q[S.wgd_index[e]]

        # non-retention
        r -= Π_wgd_non_retention(γ, q_, f, matrix)
        if r < 0
            R.labels[wgd_node] = "wgd"
            add_loss_node!(R, f, wgd_node)
            return _backtrack!(
                wgd_node, R, matrix, f, γ, slices.slices[f], S, ccd,
                slices, ε, ϕ, λ, q)
        end

        # loss
        r -= Π_wgd_loss(γ, q_, ε[f][end], f, matrix)
        if r < 0
            R.labels[wgd_node] = "wgd"
            add_loss_node!(R, f, wgd_node)
            return _backtrack!(wgd_node, R, matrix, f, γ, slices.slices[f], S,
                ccd, slices, ε, ϕ, λ, q)
        end

        # retention
        for (γ1, γ2, count) in ccd.m2[γ]
            ccp = ccd.ccp[(γ, γ1, γ2)]
            r -= q_ * ccp * matrix[f][γ1, end] * matrix[f][γ2, end]
            if r < 0
                R.labels[wgd_node] = "wgd"
                _backtrack!(
                    wgd_node, R, matrix, f, γ1, slices.slices[f], S, ccd,
                    slices, ε, ϕ, λ, q)
                _backtrack!(
                    wgd_node, R, matrix, f, γ2, slices.slices[f], S, ccd,
                    slices, ε, ϕ, λ, q)
                return
            end
        end
    end
    return
end


function add_loss_node!(R, e, node)
    addnode!(R.tree)
    new_node = last_node(R.tree)
    addbranch!(R.tree, node, new_node, 0.01)
    R.σ[new_node] = e
    R.labels[new_node] = "loss"
    return new_node
end


function add_duplication_node!(R, e, node, γ, b)
    addnode!(R.tree)
    new_node = last_node(R.tree)
    if node != -1
        addbranch!(R.tree, node, new_node, b)
    end
    R.σ[new_node] = e
    R.γ[new_node] = γ
    R.labels[new_node] = "duplication"
    return new_node
end


function add_speciation_node!(R, e, node, γ, b)
    addnode!(R.tree)
    new_node = last_node(R.tree)
    if node != -1
        addbranch!(R.tree, node, new_node, b)
    end
    R.σ[new_node] = e
    R.γ[new_node] = γ
    R.labels[new_node] = "speciation"
    return new_node
end


function add_wgd_node!(R, e, node, γ, b)
    addnode!(R.tree)
    new_node = last_node(R.tree)
    addbranch!(R.tree, node, new_node, b)
    R.σ[new_node] = e
    R.γ[new_node] = γ
    R.labels[new_node] = "wgd"
    return new_node
end


function add_leaf_node!(R, e, node, γ, b, leaf_name)
    addnode!(R.tree)
    new_node = last_node(R.tree)
    addbranch!(R.tree, node, new_node, b)
    R.γ[new_node] = γ
    R.σ[new_node] = e
    R.labels[new_node] = "leaf"
    R.leaves[new_node] = leaf_name
    return new_node
end


"""
    correct_branch_lengths!(R::RecTree, S::SpeciesTree)

Branches for clades that span multiple species tree branches (i.e. have loss
nodes in the stem to their parent clade) are assigned too long branch lengths,
that is every branch in the reconciled tree for that clade has the branch
length given in the CCD, which is the distance to the parent clade. We correct
this by partitioning this branch length proportional to the relative lengths
of the corresponding species tree branches.
"""
function correct_branch_lengths!(R::RecTree, S::SpeciesTree)
    for node in keys(R.tree.nodes)
        if isroot(R.tree, node)
            continue
        elseif R.labels[node] != "loss"
            if !("loss" in [R.labels[c] for c in childnodes(R.tree, node)])
                do_correction!(R, S, node)
            end
        end
    end
end



"""
    do_correction!(R::RecTree, S::SpeciesTree, node::Int64)

This does the actual correcting described in `correct_branch_lengths!(R, S)`
"""
function do_correction!(R::RecTree, S::SpeciesTree, node::Int64)
    b = R.tree.branches[R.tree.nodes[node].in[1]].length
    n = R.tree.branches[R.tree.nodes[node].in[1]].source
    γ = R.γ[node]
    path = Int64[node]
    while !haskey(R.γ, n) || R.γ[n] == γ
        push!(path, n)
        n = R.tree.branches[R.tree.nodes[n].in[1]].source
    end
    push!(path, n)
    if length(path) <= 2
        return
    end

    l = distance(S.tree, R.σ[node], R.σ[path[end]])
    curr_n = path[1]
    for n in path
        l_ = distance(S.tree, R.σ[curr_n], R.σ[n])
        new_length = l_/l * b
        changelength!(R.tree, R.tree.nodes[curr_n].in[1], new_length)
        curr_n = n
    end
end
