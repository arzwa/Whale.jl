# Yet another reimplementation, now with NewickTree and RatesModel interfaces,
# this has no drawbacks as far as I can see compared to the previous
# implementation.

# NOTE: best to access the model through functions, so that we can adapt the API
# without too much work
"""
    Slices
"""
struct Slices{T,I}
    slices ::Matrix{T}    # slice lengths and extinction and propagation Ps
    name   ::String
    clade  ::Set{I}
    wgdid  ::I    # NOTE: it's up to the node to identify its WGD if it is one
end

const ModelNode{T,I} = Node{I,Slices{T,I}}

function Slices(node::Node{I,D}, Δt, minn, wgdid=I(0)) where {I,D}
    t = distance(node)
    n = isnan(t) ? 0 : max(minn, ceil(Int, t/Δt))
    slices = ones(n+1, 4)
    slices[:,1] = vcat(0.0, repeat([n== 0 ? 0. : t/n], n))
    Slices(slices, name(node), Set{I}(), wgdid)
end

function copynode(n::ModelNode{T,I}, p, V::Type) where {T,I}
    @unpack data = n
    slices = ones(V, size(data.slices))
    slices[:,1] .= data.slices[:,1]
    s = Slices(slices, name(n), data.clade, data.wgdid)
    isroot(n) ? Node(id(n), s) : Node(id(n), s, p)
end

getclade(n::Node) = isleaf(n) ? Set([id(n)]) :
    Set(union([getclade(c) for c in children(n)]...))
setclade!(n::Node) = union!(n.data.clade, getclade(n))

NewickTree.name(s::Slices) = s.name
NewickTree.distance(s::Slices) = sum(s.slices[:,1])
Base.show(io::IO, s::Slices) = write(io::IO,
    "Slices($(name(s)), $(s.slices[end,1]), $(size(s.slices)[1]))")

Base.getindex(m::ModelNode, i, j) = m.data.slices[i, j]
Base.setindex!(m::ModelNode, x, i, j) = m.data.slices[i, j] = x
Base.axes(m::ModelNode, d) = Base.axes(m.data.slices, d)
Base.lastindex(m::ModelNode, d) = Base.axes(m, d)[end]
Base.length(m::ModelNode) = size(m.data.slices)[1]
FakeFamily.iswgd(n::Node) = startswith(name(n), "wgd")
FakeFamily.wgdid(n::ModelNode) = n.data.wgdid
lastslice(m::ModelNode) = lastindex(m, 1)

"""
    WhaleModel(ratesmodel, tree, Δt, [minn=5])

The main Whale model object. This is defined by a `ratesmodel` specifying the
prameterization of the phylogenetic model and a tree specifying the structure
of the model. `Δt` is the size of the slices used in the discretization of the
branch lengths. `minn` is the minimum number of slices for each branch (which
would then correspond to the maximum number of duplication/loss events possible
along a branch.
"""
struct WhaleModel{T,M,I} <:DiscreteMultivariateDistribution
    rates::M
    order::Vector{ModelNode{T,I}}
end

Base.show(io::IO, m::WhaleModel) = write(io::IO, "WhaleModel(\n$(m.rates))")
Base.length(m::WhaleModel) = length(m.order)
Base.getindex(m::WhaleModel, i) = m.order[i]
Base.lastindex(m::WhaleModel) = lastindex(m.order)
root(m::WhaleModel) = m[end]

# XXX this is insanely ugly. The ain hassle is that we want node IDs in
# order such that the leaves come first, than the internal nodes, and
# finally the WGDs (but the iteration order is still a postorder).
function WhaleModel(rates::RatesModel{T}, Ψ::Node{I}, Δt, minn=5) where {T,I}
    nonwgd = 0  # count non-wgd nodes
    wgdid = I(0)
    order = ModelNode{T,I}[]
    function walk(x, y)
        if iswgd(x)
            wgdid += I(1)
            i = wgdid
        else
            nonwgd += 1
            i = I(0)
        end
        y′ = isroot(x) ?
            Node(id(x), Slices(x, Δt, minn, i)) :
            Node(id(x), Slices(x, Δt, minn, i), y)
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
    for n in order
        if iswgd(n)
            n.id = I(i)
            i += 1
        else
            n.id = I(j)
            j += 1
        end
    end
    setclade!.(order)
    model = WhaleModel(rates, order)
    setmodel!(model)  # assume the model should be initialized
    return model
end

(m::WhaleModel)(θ) = m(m.rates(θ))
function (m::WhaleModel)(rates::RatesModel{T}) where T
    o = ModelNode{T,typeof(id(m[1]))}[]
    function walk(x, y)
        y′ = copynode(x, y, T)
        for c in children(x) walk(c, y′) end
        push!(o, y′)
    end
    walk(root(m), nothing)
    model = WhaleModel(rates, o)
    setmodel!(model)
    return model
end

function setmodel!(model)
    @unpack order, rates = model
    for n in order setnode!(n, rates) end
end

function setnode!(n::ModelNode{T}, rates::M) where {T,M}
    iswgd(n) && return setwgdnode!(n, rates)
    θn = getθ(rates, n)
    n[1,2] = isleaf(n) ? θn.p : prod([c[end,2] for c in children(n)])
    n[1,3] = one(T)
    setslices!(n.data.slices, θn.λ, θn.μ)
end

function setwgdnode!(n::ModelNode{T}, rates::M) where {T,M}
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

# define all in terms of α and β!
const ΛMATOL = 1e-6
getα(λ, μ, t) = isapprox(λ, μ, atol=ΛMATOL) ?
    λ*t/(one(t) + λ*t) : μ*(exp(t*(λ-μ)) - one(t))/(λ*exp(t*(λ-μ)) - μ)
_ϵ(α, β, ϵ) = (α + (one(α)-α-β)*ϵ)/(one(α)-β*ϵ)
_ϕ(α, β, ϵ) = (one(α)-α)*(one(α)-β)/(one(α)-β*ϵ)^2
_ψ(α, β, ϵ) = (one(α)-α)*(one(α)-β)*β/(one(α)-β*ϵ)^3

# for testing
function getslice(λ, μ, t, ϵ)
    α = getα(λ, μ, t)
    β = (λ/μ)*α
    (ϵ=_ϵ(α, β, ϵ), ϕ=_ϕ(α, β, ϵ), ψ=_ψ(α, β, ϵ))
end

function FakeFamily.nonwgdchild(n::ModelNode)
    while iswgd(n) n = first(children(n)) end
    return n
end

getwgds(m::WhaleModel) = [n for n in m.order if iswgd(n)]
