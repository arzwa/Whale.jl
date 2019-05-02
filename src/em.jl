# Expectation - Maximization algorithm for (WH)ALE (only DL model probably)
# Arthur Zwaenepoel (2019)
# NOTE: Use the BirthDeathProcesses.jl library as much as possible!
# XXX: This won't work with the GBM prior I guess?
#=
1. Get initial parameters θ(m) = θ(1)
2. Generate reconciliations (observed data Y) from these parameter values
3. Get sufficient statistics E[U|Y,θ(m)], E[D|Y,θ(m)] and E[T|Y,θ(m)]
4. Maximization step, find new θ
5. repeat until convergence
=#
# NOTE: If we want to use this for real, we will need to implement the E step
# in parallel, because currently it's probably not competitive with the MCMC
# NOTE: MAP doesn't seem to work all that wel, diffuse prior does not give ≈ ML

function evaluate_lhood!(em::WhaleEM)
    λ = [bdp.λ for bdp in em.θ]
    μ = [bdp.μ for bdp in em.θ]
    evaluate_lhood!(em.D, em.S, em.L, λ, μ, Float64[], em.η, em.r)
    set_recmat!(em.D)
end

function backtrack!(em::WhaleEM)
    bt = BackTracker(em)
    for c in em.D
        em.T[c.fname] = [backtrack(c, bt) for i=1:em.N]
    end
end

function emiter!(em::WhaleMlEM)
    evaluate_lhood!(em)         # compute lhood
    backtrack!(em)              # backtrack
    y = get_transitions(em.S, em.T)   # get transitions
    whale_emstep!(em, y)  # get expected values
end

function emiter!(em::WhaleMapEM; nt::Int64=20, nk::Int64=5)
    evaluate_lhood!(em)         # compute lhood
    backtrack!(em)              # backtrack
    y = get_transitions(em.S, em.T)   # get transitions
    whale_emstep!(em, y, nt=nt, nk=nk)  # get expected values
end

function whale_emstep!(em::WhaleMlEM, ys; nt::Int64=20, nk::Int64=5)
    # assume branch-wise rates
    root = findroots(em.S.tree)[1]
    for n in postorder(em.S.tree)
        n == root ? continue : nothing
        emstep!(em.θ[em.r[n]], collect(keys(ys[n])), collect(values(ys[n])),
            parentdist(em.S.tree, n), em.ε[n], nt=nt, nk=nk)
    end
    for bdp in em.θ
        bdp.λ = max(bdp.λ, 0.00001)
        bdp.μ = max(bdp.μ, 0.00001)
    end
    update_ε!(em)
    # the extinction probabilities should not be updated during the postorder
    # since we want the E step to be conditional on the parameter values
    # obtained in the previous iteration, right? Or would it just speed up
    # convergence if we did adapt them during the postorder?
end

function whale_emstep!(em::WhaleMapEM, ys; nt::Int64=20, nk::Int64=5)
    # assume branch-wise rates
    root = findroots(em.S.tree)[1]
    for n in postorder(em.S.tree)
        n == root ? continue : nothing
        emmapstep!(em.θ[em.r[n]], em.πλ[1], em.πλ[2], em.πμ[1], em.πμ[2],
            collect(keys(ys[n])), collect(values(ys[n])),
            parentdist(em.S.tree, n), em.ε[n], nt=nt, nk=nk)
    end
    for bdp in em.θ
        bdp.λ = max(bdp.λ, 0.00001)
        bdp.μ = max(bdp.μ, 0.00001)
    end
    update_ε!(em)
end

<<<<<<< HEAD
function get_transitions(S::SpeciesTree, trees::Dict{String,Array{RecTree,1}})
    transitions = Dict(n => zeros(Int64, 0, 2) for (n, node) in S.tree.nodes)
    for (k, v) in trees
        ts = get_transitions(S, v)
        for (n, node) in S.tree.nodes
            transitions[n] = [
                transitions[n] ; vcat([collect(t[n]) for t in ts]...)]
        end
=======
update_ε!(em::WhaleEM) = em.ε = get_extinction_probabilities(em.S, em.θ, em.r)

function get_transitions(S::SpeciesTree, trees::Dict{String,Array{RecTree,1}})
    transitions = Dict(n=>
        Dict{Tuple{Int64,Int64},Int64}() for (n,node) in S.tree.nodes)
    for (k, v) in trees
        get_transitions!(transitions, S, v)
>>>>>>> 07a419c67fa7d95b63344536feadba30284c2c6c
    end
    return transitions
end

function get_transitions(S::SpeciesTree, trees::Array{RecTree})
    transitions = Dict(n=>
        Dict{Tuple{Int64,Int64},Int64}() for (n,node) in S.tree.nodes)
    get_transitions!(transitions, S, trees)
    return transitions
end

function get_transitions!(d::Dict{}, S::SpeciesTree, trees::Array{RecTree})
    # for every branch of the species tree, compute the number of lineages
    # entering and leaving the branch
<<<<<<< HEAD
    transitions = Dict{Int64,Array{Int64,2}}[]
    root = findroots(S.tree)[1]
    for t in trees
        d = Dict{Int64,Array{Int64,2}}()
        function walk(n)
            if isroot(S.tree, n)
                d[n] = [-1 length([k for (k, v) in t.σ if v == n])]
=======
    # d = Dict(n=>Dict{Tuple{Int64,Int64},Int64}() for (n,node) in S.tree.nodes)
    root = findroots(S.tree)[1]
    for t in trees
        endcounts = Dict()
        function walk(n)
            if isroot(S.tree, n)
                trans = (-1, length([k for (k, v) in t.σ if v == n]))
                endcounts[n] = trans[2]
                haskey(d[n], trans) ? d[n][trans] += 1 : d[n][trans] = 1
>>>>>>> 07a419c67fa7d95b63344536feadba30284c2c6c
            else
                pnode = parentnode(S.tree, n)
                startcount = endcounts[pnode]
                endcount = length([k for (k, v) in t.σ
                    if (v == n && t.labels[k] != "duplication")])
<<<<<<< HEAD
                d[n] = [startcount endcount]
=======
                endcounts[n] = endcount
                trans = (startcount, endcount)
                haskey(d[n], trans) ? d[n][trans] += 1 : d[n][trans] = 1
>>>>>>> 07a419c67fa7d95b63344536feadba30284c2c6c
                isleaf(S.tree, n) ? (return) : nothing
            end
            [walk(c) for c in childnodes(S.tree, n)]
        end
        walk(root)
    end
    # return d
end

function mstep(em::WhaleEM)
end
