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

function evaluate_lhood!(em::WhaleEM)
    evaluate_lhood!(em.D, em.S, em.L, em.λ, em.μ, em.q, em.η, em.r)
    set_recmat!(em.D)
end

function bactrack!(em::WhaleEM)
    bt = BackTracker(em)
    for c in em.D
        em[c.fname] = [backtrack(c, bt) for i=1:em.N]
    end
end

function emiter!(em::WhaleEM)
    # E-step ↓
    # compute lhood
    evaluate_lhood!(em)
    # backtrack
    backtrack!(em)
    # get transitions
    y = get_transitions(em.T)
    # get expected values
    𝔼u, 𝔼d, 𝔼t = 𝔼step(em, y)

    # M-step ↓
    λp, μp = mstep(em, 𝔼u, 𝔼d, 𝔼t)
    emlog(λp, μp)
    em.λ = λp
    em.μ = μp
end

function 𝔼step(em::WhaleEM, y::Array)
    # use the BirthDeathProcesses.jl code
    # postorder over species tree
    # summarize for each branch the data in y over all families
    # get expected values under discretely observed BDP, put them in a dict
    # species tree branch => (𝔼u, 𝔼d, 𝔼t)
end

function get_transitions(S::SpeciesTree, trees::Dict{String,Array{RecTree}})
    transitions = Dict{Int64,Tuple{Int64,Int64}}[]
    for (k, v) in trees
        push!(transitions, get_transitions(S, v))
    end
end

function get_transitions(S::SpeciesTree, trees::Array{RecTree})
    # for every branch of the species tree, compute the number of lineages
    # entering and leaving the branch
    transitions = Dict{Int64,Tuple{Int64,Int64}}[]
    root = findroots(S.tree)[1]
    for t in trees
        d = Dict{Int64,Tuple{Int64,Int64}}()
        function walk(n)
            if isroot(S.tree, n)
                d[n] = (-1, length([k for (k, v) in t.σ if v == n]))
            else
                pnode = parentnode(S.tree, n)
                startcount = d[pnode][2]
                endcount = length([k for (k, v) in t.σ
                    if (v == n && t.labels[k] != "duplication")])
                d[n] = (startcount, endcount)
                isleaf(S.tree, n) ? (return) : nothing
            end
            [walk(c) for c in childnodes(S.tree, n)]
        end
        walk(root)
        push!(transitions, d)
    end
    return transitions
end

function mstep(em::WhaleEM)
end
