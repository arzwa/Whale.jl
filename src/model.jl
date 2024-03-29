# NOTE: best to access the model through functions, so that we can adapt the API
# without too much work
"""
    Slices
"""
struct Slices{T,I}
    slices ::Matrix{T} # slice lengths and extinction and propagation Ps
    name   ::String
    clade  ::Set{I}
    leafℙ  ::Float64   # ℙ of a compatible leaf to be seen at this node (MUL)
    wgdid  ::I    # NOTE: it's up to the node to identify its WGD if it is one
end

const ModelNode{T,I} = Node{I,Slices{T,I}}

function Slices(node::Node{I,D}, Δt, minn, maxn, wgdid=I(0), ℙ=1.0) where {I,D}
    t = distance(node)
    n = isnan(t) ? 0 : min(maxn, max(minn, ceil(Int, t/Δt)))
    slices = ones(n+1, 4)
    slices[:,1] = vcat(0.0, repeat([n== 0 ? 0. : t/n], n))
    Slices(slices, String(name(node)), Set{I}(), ℙ, wgdid)
end

function copynode(n::ModelNode{T,I}, p, V::Type) where {T,I}
    @unpack data = n
    slices = ones(V, size(data.slices))
    slices[:,1] .= data.slices[:,1]
    s = Slices(slices, name(n), data.clade, data.leafℙ, data.wgdid)
    isroot(n) ? Node(id(n), s) : Node(id(n), s, p)
end

function setclade!(n::Node, mulid=Dict())
    if isleaf(n)
        c = haskey(mulid, name(n)) ? Set([mulid[name(n)]]) : Set([id(n)])
        union!(n.data.clade, c)
    else
        for c in children(n)
            setclade!(c, mulid)
            union!(n.data.clade, c.data.clade)
        end
    end
end

NewickTree.name(s::Slices) = s.name
NewickTree.distance(s::Slices) = sum(s.slices[:,1])
Base.show(io::IO, s::Slices) = write(io::IO,
    "Slices($(name(s)), $(s.slices[end,1]), $(size(s.slices)[1]))")

Base.getindex(m::ModelNode, i, j) = m.data.slices[i, j]
Base.setindex!(m::ModelNode, x, i, j) = m.data.slices[i, j] = x
Base.axes(m::ModelNode, d) = Base.axes(m.data.slices, d)
Base.lastindex(m::ModelNode, d) = Base.axes(m, d)[end]
Base.length(m::ModelNode) = size(m.data.slices)[1]
iswgd(n::Node) = startswith(name(n), "wgd")
wgdid(n::ModelNode) = n.data.wgdid
lastslice(m::ModelNode) = lastindex(m, 1)
ismulnode(n::Node) = 0. < n.data.leafℙ < 1.0

abstract type SamplingCondition end

"""
    WhaleModel(ratesmodel, tree, Δt, [minn=5, maxn=50])

The main Whale model object. This is defined by a `ratesmodel` specifying the
prameterization of the phylogenetic model and a tree specifying the structure
of the model. `Δt` is the size of the slices used in the discretization of the
branch lengths. `minn` is the minimum number of slices for each branch (which
would then correspond to the maximum number of duplication/loss events possible
along a branch. `maxn` is then the maximum number of slices on a branch.
"""
struct WhaleModel{T,M,I,C<:SamplingCondition} <:DiscreteMultivariateDistribution
    rates::M
    order::Vector{ModelNode{T,I}}
    index::Vector{I}
    condition::C
end

Base.length(m::WhaleModel) = length(m.order)
Base.getindex(m::WhaleModel, i) = m.order[m.index[i]]
Base.lastindex(m::WhaleModel) = lastindex(m.order)
root(m::WhaleModel) = m.order[end]
NewickTree.getroot(m::WhaleModel) = root(m)

# XXX this is insanely ugly. The main hassle is that we want node IDs in
# order such that the leaves come first, than the internal nodes, and
# finally the WGDs (but the iteration order is still a postorder).
# NOTE: MUL trees are a bit of a hack, they are defined by providing a
# tree with duplicate leaves. Each MUL tree leaf gets a unique ID for
# indexing, but we use a single representative of these IDs to be used
# as clade IDs. This is because the `iscompatible` function takes a look
# at `node.data.clade`. Using this approach, we have minimal struggle
# with gene/species names and we can easily specify a MUL tree from a
# newick string, but it doesn't feel very consistent...
# NOTE: with MUL trees, it is implictly assumed that there is perfect 
# retention.
function WhaleModel(rates::Params{T}, Ψ::Node{I}, Δt;
        minn=5, maxn=10000, condition=RootCondition()) where {T,I}
    nonwgd = 0  # count non-wgd nodes
    wgdid = I(0)
    order = ModelNode{T,I}[]
    mulgroups = filter(x->x.second > 1, countmap(name.(getleaves(Ψ))))
    function walk(x, y)
        if iswgd(x)
            wgdid += I(1)
            i = wgdid
        else
            nonwgd += 1
            i = I(0)
        end
        ℙ = haskey(mulgroups, name(x)) ?
            1.0/mulgroups[name(x)] : isleaf(x) ? 1.0 : 0.0
        y′ = isroot(x) ?
            Node(id(x), Slices(x, Δt, minn, maxn, i, ℙ)) :
            Node(id(x), Slices(x, Δt, minn, maxn, i, ℙ), y)
        for c in children(x)
            walk(c, y′)
        end
        push!(order, y′)
        return y′
    end
    n = walk(Ψ, nothing)
    i = nonwgd+1
    j = 1
    order = union(getleaves(order[end]), order)
    index = zeros(I, length(order))
    mulid = Dict{String,I}()
    for (k,n) in enumerate(order)
        if iswgd(n)
            n.id = I(i)
            i += 1
        else
            n.id = I(j)
            if haskey(mulgroups, name(n)) 
                mulid[name(n)] = n.id
            end
            j += 1
        end
        index[id(n)] = k
    end
    # setclade!.(order, Ref(mulid))
    setclade!(order[end], mulid)
    model = WhaleModel(rates, order, index, condition)
    setmodel!(model)  # assume the model should be initialized
    return model
end

(m::WhaleModel)(θ) = m(m.rates(θ))
function (m::WhaleModel)(rates::Params{T}) where T
    r = root(m)
    I = typeof(id(r))
    o = similar(m.order, ModelNode{T,I})
    for i in reverse(1:length(o))
        n = m.order[i]
        o[i] = copynode(n, isroot(n) ?
            nothing : o[m.index[id(parent(n))]], T)
    end
    model = WhaleModel(rates, o, m.index, m.condition)
    setmodel!(model)
    return model
end

function setmodel!(model)
    @unpack order, rates = model
    for n in order setnode!(n, rates) end
end

function setnode!(n::ModelNode{T}, rates) where T
    iswgd(n) && return setwgdnode!(n, rates)
    θn = getθ(rates, n)
    n[1,2] = isleaf(n) ? θn.p : prod([c[end,2] for c in children(n)])
    n[1,3] = one(T)
    setslices!(n.data.slices, θn.λ, θn.μ)
end

function setwgdnode!(n::ModelNode{T}, rates) where T
    ϵ = n[1][end,2]
    θn = getθ(rates, n)
    n[1,2] = θn.q * ϵ^2 + (1. - θn.q) * ϵ
    setslices!(n.data.slices, θn.λ, θn.μ)
end

function setslices!(A::Matrix, λ, μ)
    for i=2:size(A)[1]
        α = getα(λ, μ, A[i,1])
        β = (λ/μ)*α
        ϵ = A[i-1,2]
        A[i,2] = _ϵ(α, β, ϵ)
        A[i,3] = _ϕ(α, β, ϵ)
        A[i,4] = _ψ(α, β, ϵ)
    end
end

# for testing
function getslice(λ, μ, t, ϵ)
    α = getα(λ, μ, t)
    β = (λ/μ)*α
    (ϵ=_ϵ(α, β, ϵ), ϕ=_ϕ(α, β, ϵ), ψ=_ψ(α, β, ϵ))
end

function nonwgdchild(n)
    while iswgd(n) n = first(children(n)) end
    return n
end

function nonwgdparent(n)
    while iswgd(n) n = parent(n) end
    return n
end

getwgds(m::WhaleModel) = [n for n in m.order if iswgd(n)]

# We want the show method to display all relevant information so that we can
# always check easily when paranoid. Better be a bit too verbose here!
function Base.show(io::IO, m::WhaleModel)
    write(io, "WhaleModel\n$("—"^10)\n⋅Parameterization:\n$(m.rates)\n")
    write(io, "⋅Condition:\n$(typeof(m.condition))\n\n")
    write(io, "⋅Model structure:\n$(length(m.order)) nodes (")
    write(io, "$(length(getleaves(getroot(m)))) leaves, ")
    write(io, "$(length(getwgds(m))) WGD nodes)\n")
    write(io, "node_id,wgd_id,distance,Δt,n,subtree\n")
    for n in m.order
        line = [Int(id(n)), Int(wgdid(n)),
            round(distance(n), digits=4),
            round(n[end,1], digits=4), length(n)-1,
            "\"$(nwstr(n, dist=false))\""]
        write(io, join(line, ","), "\n")
    end
end

# I need this quite often; add a WGD on each internal branch
function addbranchwgds!(tree; tips=false)
    nwgd = 0
    for n in postwalk(tree)
        (isroot(n) || (!tips && isleaf(n))) && continue
        nwgd += 1
        insertnode!(n, name="wgd_$nwgd")
    end
    nwgd
end

"""
    setsamplingp!(model, dict)

Set the sampling probabilities for the leaves in the model given the leaf
names. This is a convenience function enabling one to use the leaf labels
instead of node IDs.
"""
function setsamplingp!(model, dict::Dict)
    @unpack rates = model
    leaves = getleaves(getroot(model))
    length(rates.p) == 0 && push!(rates.p, zeros(length(leaves))...)
    for l in leaves
        rates.p[id(l)] = haskey(dict, name(l)) ? dict[name(l)] : 0.
    end
end

struct ModelArray{M} <: DiscreteMultivariateDistribution
    models::Vector{M}
end

Base.length(m::ModelArray) = length(m.models)

