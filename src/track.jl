#===============================================================================
Backtracking functions. The goal is to be able to sample reconciled trees given
an arbitrary input of duplication, loss and retention rates. The root poses a
challenge.

    10/01/2019: basics seem to work
===============================================================================#
# TODO: How to get reasonable/correct branch lengths, in particular for WGDs?

# Backtracking in parallel
function backtrackall(ccd::DArray, bt::BackTracker, N::Int64)
    rtrees = ppeval(backtrack, ccd, [bt], [N], dims=(2,2))
    return Dict(ccd[i].fname => Array(rtrees[:, i]) for i=1:length(ccd))
end

function backtrack(ccd, bt, N)
    rtrees = RecTree[]
    for i=1:N[1]
        push!(rtrees, backtrack(ccd[1], bt[1]))
    end
    @show size(rtrees)
    return rtrees
end

# Backtracking from posterior distribution
"""
    backtrackmcmcpost(sample::DataFrame, ccd, S, slices, N)
Backtrack using samples from the posterior
"""
function backtrackmcmcpost(sample::DataFrame, ccd, S, slices, N; q1::Bool=false)
    D = distribute(ccd)
    λc = [i for (i, var) in enumerate(names(sample))
        if startswith(string(var), "l")][1:end-1]
    μc = [i for (i, var) in enumerate(names(sample))
        if startswith(string(var), "m")][1:end]
    qc = [i for (i, var) in enumerate(names(sample))
        if startswith(string(var), "q")][1:end]
    allrtrees = Dict{Any,Array{RecTree}}(ccd[i].fname => RecTree[]
        for i in 1:length(ccd))
    p = Progress(N, 0.1, "| backtracking (N = $N)")
    for i in rand(1:size(sample)[1], N)
        # HACK: fix your package dependencies!
        λ = [] ; μ = [] ; q = []
        try
            λ = collect(sample[i, λc])
            μ = collect(sample[i, μc])
            q1 ? q = ones(length(qc)) .- 0.001 : q = collect(sample[i, qc])
        catch MethodError
            @warn "DataFrames not up to date, have to convert to DataFrameRow"
            λ = collect(DataFrameRow(sample[i, λc], 1))
            μ = collect(DataFrameRow(sample[i, μc], 1))
            q1 ? q = ones(length(qc)) .- 0.001 :
                q = collect(DataFrameRow(sample[i, qc], 1))
        end
        η = sample[i, :eta]
        ri = Dict(x => 1 for x in keys(S.tree.nodes))
        bt = BackTracker(S, slices, ri, λ, μ, q, η)
        Whale.evaluate_lhood!(D, S, slices, λ, μ, q, η, ri)
        Whale.set_recmat!(D)
        for c in D
            push!(allrtrees[c.fname], backtrack(c, bt))
        end
        next!(p)
    end
    return allrtrees
end

"""
    backtrack(...)
Sample a reconciled tree by stochastic backtracking along the dynamic
programming matrix. Works *recursively*.
"""
function backtrack(ccd::CCD, bt::BackTracker)
    R = RecTree()
    backtrack!(-1, ccd.Γ, 1, 1, R, ccd, bt)   # start of recursion
    correct_branch_lengths!(R, bt.S)
    return R
end

"""
    backtrack!(node, γ, e, i, R, ccd, bt::BackTracker)
"""
function backtrack!(node, γ, e, i, R, ccd, bt::BackTracker)
    if e == 1
        return backtrack_root!(node, γ, R, ccd, bt)
    elseif i == 1
        return backtrack_inter!(node, γ, e, R, ccd, bt)
    else
        return backtrack_intra!(node, γ, e, i, R, ccd, bt)
    end
end

# Backtracking within a branch =====================================================================
function backtrack_intra!(node, γ::Int64, e::Int64, i::Int64, R::RecTree, ccd::CCD, bt::BackTracker)
    if haskey(ccd.m3, γ)  # leaf clade, cannot duplicate
        return backtrack!(node, γ, e, i-1, R, ccd, bt)
    end
    p = ccd.recmat[e][γ, i]
    r = rand() * p
    λ = bt.λ[bt.ri[e]]
    Δt = bt.slices.slice_lengths[e][i]

    # propogation
    r -= bt.ϕ[e][i] * ccd.recmat[e][γ, i-1]
    if r < 0.
        return backtrack!(node, γ, e, i-1, R, ccd, bt)
    end

    # duplication
    r, γ1, γ2 = duplication(r, ccd, γ, e, i, λ, Δt)
    if r < 0.
        dnode = add_duplication_node!(R, e, node, γ, ccd.blens[γ])
        backtrack!(dnode, γ1, e, i-1, R, ccd, bt)
        backtrack!(dnode, γ2, e, i-1, R, ccd, bt)
        return
    else
        error("Backtracking failed (intra): r = $r; γ = $γ; e = $e")
    end
end

# duplication events
function duplication(r, ccd, γ, e, i, λ, Δt)
    for (γ1, γ2, count) in ccd.m2[γ]
        ccp = ccd.ccp[(γ, γ1, γ2)]
        # here the factor 2 or BDP issue comes up again
        r -= ccp * ccd.recmat[e][γ1, i-1] * ccd.recmat[e][γ2, i-1] * λ * Δt
        if r < 0.
            return r, γ1, γ2
        end
    end
    return r, -1, -1
end

# Backtracking at speciation and WGD nodes =========================================================
# backtrack at nodes in S
function backtrack_inter!(node::Int64, γ::Int64, e::Int64, R::RecTree, ccd::CCD, bt::BackTracker)
    if isleaf(bt.S.tree, e) && haskey(ccd.m3, γ)
        # leaf node in both species and gene tree
        add_leaf_node!(R, e, node, γ, ccd.blens[γ], ccd.leaves[γ])
        return node  # ends recursion
    elseif length(childnodes(bt.S.tree, e)) == 2
        # speciation node
        return bt_sp!(node, γ, e, R, ccd, bt)
    elseif length(childnodes(bt.S.tree, e)) == 1
        # wgd node
        return bt_wgd!(node, γ, e, R, ccd, bt)
    else
        # happens when we're in a leaf branch, at time 1, but not with a leaf clade!
        error("Backtracking failed (inter, γ ≠ leaf, e = leaf): γ = $γ, e = $e")
        return  # stops recursion, REALLY shouldn't occur
    end
end

# backtrack at bifurcating node in S (speciation)
function bt_sp!(node::Int64, γ::Int64, e::Int64, R::RecTree, ccd::CCD, bt::BackTracker)
    p = ccd.recmat[e][γ, 1]
    r = rand() * p
    f, g = childnodes(bt.S.tree, e)

    # speciation + loss event
    r, bs, bl = non_bifurcation(r, ccd, γ, bt.ε, f, g)
    if r < 0.
        spnode = add_speciation_node!(R, e, node, γ, ccd.blens[γ])
        add_loss_node!(R, bl, spnode)
        return backtrack!(spnode, γ, bs, bt.slices.slices[bs], R, ccd, bt)
    end

    # speciation event
    if !(haskey(ccd.m2, γ))
        # if γ is a leaf, it should have gone through an SL event
        error("Backtracking failed (inter, γ = leaf): r = $r; γ = $γ; e = $e")
    end
    r, γ1, γ2, b1, b2 = bifurcation(r, ccd, γ, f, g)
    if r < 0.
        spnode = add_speciation_node!(R, e, node, γ, ccd.blens[γ])
        backtrack!(spnode, γ1, b1, bt.slices.slices[b1], R, ccd, bt)
        backtrack!(spnode, γ2, b2, bt.slices.slices[b2], R, ccd, bt)
        return
    else
        error("Backtracking failed (inter): r = $r; γ = $γ; e = $e")
    end
end

# backtrack at non-bifurcating node in S (wgd)
function bt_wgd!(node::Int64, γ::Int64, e::Int64, R::RecTree, ccd::CCD, bt::BackTracker)
    p = ccd.recmat[e][γ, 1]
    r = rand() * p
    f = childnodes(bt.S.tree, e)[1]
    q = bt.q[bt.S.wgd_index[e]]

    # add wgd node
    wgd_node = add_wgd_node!(R, e, node, γ, ccd.blens[γ])

    # non-retention/loss, currently no loss node added
    r -= Π_wgd_non_retention(γ, q, f, ccd.recmat) + Π_wgd_loss(γ, q, bt.ε[f][end], f, ccd.recmat)
    if r < 0.
        #add_loss_node!(R, f, wgd_node)
        return backtrack!(wgd_node, γ, f, bt.slices.slices[f], R, ccd, bt)
    end

    # retention
    r, γ1, γ2 = wgd_retention(r, ccd, γ, f, q)
    if r < 0.
        backtrack!(wgd_node, γ1, f, bt.slices.slices[f], R, ccd, bt)
        backtrack!(wgd_node, γ2, f, bt.slices.slices[f], R, ccd, bt)
        return
    else
        error("Backtracking failed (WGD): r = $r; γ = $γ; e = $e")
    end
end

# wgd retention
function wgd_retention(r, ccd, γ, f, q)
    for (γ1, γ2, count) in ccd.m2[γ]
        ccp = ccd.ccp[(γ, γ1, γ2)]
        r -= q * ccp * ccd.recmat[f][γ1, end] * ccd.recmat[f][γ2, end]
        if r < 0.
            return r, γ1, γ2
        end
    end
end

# speciation + loss events
function non_bifurcation(r, ccd, γ, ε, f, g)
    r -= ccd.recmat[f][γ, end] * ε[g][end]
    if r < 0.
        return r, f, g
    end
    r -= ccd.recmat[g][γ, end] * ε[f][end]
    if r < 0.
        return r, g, f
    end
    return r, f, g
end

# speciation events
function bifurcation(r, ccd, γ, f, g)
    for (γ1, γ2, count) in ccd.m2[γ]
        ccp = ccd.ccp[(γ, γ1, γ2)]
        r -= ccp * ccd.recmat[f][γ1, end] * ccd.recmat[g][γ2, end]
        if r < 0.
            return r, γ1, γ2, f, g
        end
        r -= ccp * ccd.recmat[g][γ1, end] * ccd.recmat[f][γ2, end]
        if r < 0.
            return r, γ1, γ2, g, f
        end
    end
    return r, -1, -1, f, g
end

# Root backtracking stuff ==========================================================================
"""
    backtrack_root!(γ, R::RecTree, ccd, slices, S, state, ε, ϕ)

Backtracking for root events.

Note that the conditioning steps do not change the reconciliation matrix, does
this matter? Or does it simply mean we should not bother with it here?

The root probability entry for a clade γ is composed of
    Π_root + η_ * Π_spec + η_ * Π_spec + loss
"""
function backtrack_root!(node, γ, R, ccd, bt::BackTracker)
    node == -1 ? p = ccd.recmat[1][γ] / bt.η : p = ccd.recmat[1][γ] # XXX I think?
    r = rand() * p
    ε0 = bt.ε[1][1]
    η_ = 1.0/(1. - (1. - bt.η) * ε0)^2
    f, g = childnodes(bt.S.tree, 1)  # root is always bifurcating in the species tree

    # bifurcating events
    if !haskey(ccd.m3, γ)
        r, γ1, γ2, b1, b2 = root_bifurcation(r, ccd, γ, bt.η, η_, f, g)
        if r < 0. && b1 == b2 == 1
            n = add_duplication_node!(R, 1, node, γ, ccd.blens[γ])
            backtrack_root!(n, γ1, R, ccd, bt)
            backtrack_root!(n, γ2, R, ccd, bt)
            return
        elseif r < 0. && b1 != b2
            spnode = add_speciation_node!(R, 1, node, γ, ccd.blens[γ])
            backtrack!(spnode, γ1, b1, bt.slices.slices[b1], R, ccd, bt)
            backtrack!(spnode, γ2, b2, bt.slices.slices[b2], R, ccd, bt)
            return
        end
    end

    # non-bifurcating events (speciation + loss)
    r, bs, bl = root_nonbifurcation(r, ccd, γ, η_, bt.ε, f, g)
    # bs = branch where species is represented, bl = branch with loss
    if r < 0.
        spnode = add_speciation_node!(R, 1, node, γ, ccd.blens[γ])
        add_loss_node!(R, bl, spnode)
        return backtrack!(spnode, γ, bs, bt.slices.slices[bs], R, ccd, bt)
    else
        error("Backtracking failed (root): r = $r; γ = $γ")
    end
end

# a bifurcation at (speciation) or before (duplication) the root
function root_bifurcation(r, ccd, γ, η, η_, f, g)
    for (γ1, γ2, count) in ccd.m2[γ]
        ccp = ccd.ccp[(γ, γ1, γ2)]

        # either stay in root and duplicate
        r -= ccp * ccd.recmat[1][γ1, 1] * ccd.recmat[1][γ2, 1] * √(1. / η_) * (1. - η)
        if r < 0
            return r, γ1, γ2, 1, 1
        end

        # or speciate
        r -= ccp * ccd.recmat[f][γ1, end] * ccd.recmat[g][γ2, end] * η_
        if r < 0.
            return r, γ1, γ2, f, g
        end

        r -= ccp * ccd.recmat[g][γ1, end] * ccd.recmat[f][γ2, end] * η_
        if r < 0.
            return r, γ1, γ2, g, f
        end
    end
    return r, -1, -1, g, f
end

# speciation + loss at the root (this should only be possible when there
# are multiple lineages at the root)
function root_nonbifurcation(r, ccd, γ, η_, ε, f, g)
    r -= ccd.recmat[f][γ, end] * ε[g][end] * η_
    if r < 0.
        return r, f, g
    end
    r -= ccd.recmat[g][γ, end] * ε[f][end] * η_
    if r < 0.
        return r, g, f
    end
    return r, f, g
end

# Rectree functions
function add_loss_node!(R, e, node)
    new_node = addnode!(R.tree)
    addbranch!(R.tree, node, new_node, 0.01)
    R.σ[new_node] = e
    R.labels[new_node] = "loss"
    return new_node
end

function add_duplication_node!(R, e, node, γ, b)
    new_node = addnode!(R.tree)
    if node != -1
        addbranch!(R.tree, node, new_node, b)
    end
    R.σ[new_node] = e
    R.γ[new_node] = γ
    R.labels[new_node] = "duplication"
    return new_node
end

function add_speciation_node!(R, e, node, γ, b)
    new_node = addnode!(R.tree)
    if node != -1
        addbranch!(R.tree, node, new_node, b)
    end
    R.σ[new_node] = e
    R.γ[new_node] = γ
    R.labels[new_node] = "speciation"
    return new_node
end

function add_wgd_node!(R, e, node, γ, b)
    new_node = addnode!(R.tree)
    addbranch!(R.tree, node, new_node, b)
    R.σ[new_node] = e
    R.γ[new_node] = γ
    R.labels[new_node] = "wgd"
    return new_node
end

function add_leaf_node!(R, e, node, γ, b, leaf_name)
    new_node = addnode!(R.tree)
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

# FIXME: I think it (the branch length) may be incorrect when for instance
# a wgd node is in between two duplication nodes
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
        new_length = (l_/l) * b
        changelength!(R.tree, R.tree.nodes[curr_n].in[1], new_length)
        curr_n = n
    end
end


#= Backtracking at speciation and WGD nodes
# backtrack at nodes in S
function backtrack_inter!(node::Int64, γ::Int64, e::Int64, R::RecTree, ccd::CCD, bt::BackTracker)
    if isleaf(bt.S.tree, e) && haskey(ccd.m3, γ)
        # leaf node in both species and gene tree
        add_leaf_node!(R, e, node, γ, ccd.blens[γ], ccd.leaves[γ])
        return node  # ends recursion
    elseif isleaf(bt.S.tree, e) && isambiguousleafof(γ, ccd, e, S) # XXX subgenome ambiguity
        # leaf node in both species and gene tree
        print("yes")
        add_leaf_node!(R, e, node, γ, ccd.blens[γ], ccd.leaves[γ])
        return node  # ends recursion
    elseif length(childnodes(bt.S.tree, e)) == 2
        # speciation node
        return bt_sp!(node, γ, e, R, ccd, bt)
    elseif length(childnodes(bt.S.tree, e)) == 1
        # wgd node
        return bt_wgd!(node, γ, e, R, ccd, bt)
    else
        # happens when we're in a leaf branch, at time 1, but not with a leaf clade!
        error("Backtracking failed (inter, γ ≠ leaf, e = leaf): γ = $γ, e = $e")
        return  # stops recursion, REALLY shouldn't occur
    end
end =#
