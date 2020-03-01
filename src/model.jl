# event in a species tree
# The approach with the nodes not directly being linked is getting a mess, should
# test whether direct approach is equally efficient
# NOTE: here λ and μ are stored on a log-scale
abstract type Event{T} end

mutable struct Root{T<:Real} <: Event{T}
    η::T
    λ::T
    μ::T
end

mutable struct Speciation{T<:Real} <: Event{T}
    λ::T
    μ::T
    t::Float64
end

mutable struct WGD{T<:Real} <: Event{T}
    q::T
    t::Float64
end

"""
    WhaleNode{T,E<:Event{T},I}

The node struct for the WhaleModel, not part of the API.

!!! note
    Here I'm still unsure whether it would be better to store links to
    children directly, insetad of storing links to their indices. For
    many downstream stuff it is more convenient that the node holds
    direct references to its neighbors, but there are more type related
    issues in that case.
"""
mutable struct WhaleNode{T<:Real,E<:Event{T},I<:Integer}
    id      ::I
    parent  ::I            # parent node ID
    children::Set{I}       # child node IDs
    slices  ::Matrix{T}    # slice lengths and extinction and propagation Ps
    event   ::E
    clade   ::Set{I}
end

"""
    WhaleModel{T,I}

The model structure, i.e. a tree consisting of parameterized
event nodes (e.g. Speciation, WGD, Root).
"""
struct WhaleModel{T<:Real,I<:Integer}
    nodes ::Dict{I,WhaleNode{T,<:Event{T},I}}
    leaves::Dict{I,String}
    order ::Vector{I}
end

Base.length(wn::WhaleNode) = size(wn.slices)[1]
Base.length(wm::WhaleModel) = length(wm.nodes)
Base.show(io::IO, w::WhaleModel{T}) where {T} = write(io, "WhaleModel{$T}(N=$(length(w)))")
Base.show(io::IO, n::WhaleNode{T,E}) where {T,E} = write(io, "WhaleNode{$T,$E}($(n.id); $(Int.(n.clade)))")
Base.push!(n::WhaleNode, i) = push!(n.children, i)
Base.getindex(wm::WhaleModel, i) = wm.nodes[i]

NewickTree.id(n::WhaleNode) = n.id
NewickTree.isroot(n::WhaleNode) = typeof(n.event)<:Root
NewickTree.isleaf(n::WhaleNode) = length(n.children) == 0
NewickTree.distance(n::WhaleNode) = isroot(n) ? 0. : n.event.t
NewickTree.children(n::WhaleNode) = n.children
parentnode(n::WhaleNode) = n.parent

iswgd(n::WhaleNode) = typeof(n.event)<:WGD
nnonwgd(wm::WhaleModel) = 2*length(wm.leaves) - 1
nwgd(wm::WhaleModel) = length(wm) - nnonwgd(wm)
lastslice(wn::WhaleNode) = lastindex(wn.slices, 1)
lastslice(wm::WhaleModel, i::Integer) = lastindex(wm[i].slices, 1)

function nonwgdchild(n::WhaleNode, wm::WhaleModel)
    while iswgd(n)
        n = wm[first(n.children)]
    end
    n
end

WhaleNode{I}(event::Root) where I<:Integer =
    WhaleNode(I(1), I(0), Set{I}(), ones(1,3), event, Set{I}())

function WhaleNode(id::I, event::Union{Speciation,WGD}, parent::I, n) where I
    M = [vcat(0.0, repeat([event.t/n], n)) ones(n+1) ones(n+1)]
    WhaleNode(id, parent, Set{I}(), M, event, Set{I}())
end

function WhaleModel(nw; Δt=0.05, minn=5, λ=0., μ=0., η=0.9, I=UInt16)
    @unpack nodes, leaves, values = readnw(nw)
    d = Dict{I,WhaleNode{Float64,<:Event{Float64},I}}(
            I(1)=>WhaleNode{I}(Root(η, λ, μ)))
    n = Dict{I,String}()
    order = I[]
    function walk(x, y)
        id = I(x.id)
        push!(order, id)
        if isroot(x)
            x_ = d[id]
        else
            nslices = max(minn, ceil(Int, x.x / Δt))
            d[id] = x_ = WhaleNode(id, Speciation(λ, μ, x.x), y.id, nslices)
            push!(y, id)  # add child to parent
        end
        isleaf(x) ? n[id] = leaves[id] : [walk(c, x_) for c in x.children]
    end
    walk(nodes[1], nothing)
    wm = WhaleModel(d, n, reverse(order))
    setclades!(wm)
    set!(wm)
    return wm
end

setclades!(wm::WhaleModel) = setclades!(wm[1], wm)
setclades!(n::WhaleNode, wm::WhaleModel) =
    union!(n.clade, isleaf(n) ?
        n.id : union([setclades!(wm[x], wm) for x in n.children]...))

set!(w::WhaleModel) = setwalk!(w[1], w)
set!(n::WhaleNode, wm::WhaleModel) = setnode!(n, n.event, wm)

function setwalk!(n, wm)
    if !isleaf(n)
        for c in children(n)
            setwalk!(wm[c], wm)
        end
    end
    set!(n, wm)
end

function setabove!(n::WhaleNode, wm::WhaleModel)
    while parentnode(n) != 0
        set!(n, wm)
        n = wm[n.parent]
    end
end

function setnode!(n::WhaleNode, e::Union{Speciation,Root}, wm::WhaleModel)
    n.slices[1,2] = isleaf(n) ?
        0.0 : prod([wm[c].slices[end,2] for c in children(n)])
    n.slices[1,3] = 1.0
    setslices!(n.slices, e.λ, e.μ)
end

function setnode!(n::WhaleNode, e::WGD, wm::WhaleModel)
    ϵ = wm[first(n.children)].slices[end,2]
    nextsp = nonwgdchild(n, wm)
    n.slices[1,2] = e.q * ϵ^2 + (1. - e.q) * ϵ
    setslices!(n.slices, nextsp.event.λ, nextsp.event.μ)
end

function setslices!(A::Matrix, λ, μ)
    for i=2:size(A)[1]
        A[i,2] = ϵ_slice(exp(λ), exp(μ), A[i,1], A[i-1,2])
        A[i,3] = ϕ_slice(exp(λ), exp(μ), A[i,1], A[i-1,2])
    end
end

# NOTE: here λ and μ should be on rate scale
function ϵ_slice(λ, μ, t, ε)
    if isapprox(λ, μ, atol=1e-5)
        return 1. + (1. - ε)/(μ * (ε - 1.) * t - 1.)
    else
        return (μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ
    end
end

# NOTE: here λ and μ should be on rate scale
function ϕ_slice(λ, μ, t, ε)
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

# WGDs
function addwgd!(wm::WhaleModel{T,I}, node, t::T, q::T=rand(T)) where {T,I}
    @assert !isroot(node) "Cannot add WGD above root node"
    ti = node.event.t
    @assert ti - t > 0. "Invalid WGD time $ti - $t < 0."
    parent = wm[node.parent]
    islices = cumsum(node.slices, dims=1)[:,1]
    k = findfirst(x->abs(x - t) <= islices[2]/2, islices)
    twgd = islices[k]
    j = I(length(wm)+1)
    wgdslices = vcat(node.slices[1,:]', node.slices[k:end,:])
    node.slices = node.slices[1:k-1,:]
    wgd = WhaleNode(j, parent.id, Set{I}(node.id), wgdslices, WGD(q, ti-t), node.clade)
    wm.nodes[j] = wgd
    delete!(parent.children, node.id)
    push!(parent.children, j)
    node.parent = j
    node.event.t = t
    idx = findfirst(x->x==I(node.id), wm.order)
    insert!(wm.order, idx+1, j)
    setabove!(wm[j], wm)
end

function rmwgd!(wm::WhaleModel{T,I}, i) where {T,I}
    wgdnode = wm[i]
    @assert iswgd(wgdnode) "Not a WGD node $i"
    ti = wgdnode.event.t
    parent = wm[wgdnode.parent]
    child = wm[first(wgdnode.children)]
    child.parent = parent.id
    delete!(parent.children, i)
    push!(parent, child.id)
    child.slices = vcat(child.slices, wgdnode.slices[2:end,:])
    idx = findfirst(x->x==I(i), wm.order)
    deleteat!(wm.order, idx)
    delete!(wm.nodes, i)
    setabove!(wm[child.id], wm)
end

getq(m::WhaleModel{T}) where T = T[m[i].event.q for i in nnonwgd(m)+1:length(m)]
getwgd(m::WhaleModel) = nnonwgd(m)+1:length(m)
leaves(n::WhaleNode, wm::WhaleModel) = [wm.leaves[i] for i in n.clade]
lcanode(wm::WhaleModel, lca::Array{String}) =
    wm.order[findfirst((n)->lca ⊆ leaves(wm[n], wm) && !iswgd(wm[n]), wm.order)]

abstract type RatesModel{T} end

"""
    ConstantRates

Rates model with one λ and one μ for the entire tree.
"""
@with_kw struct ConstantRates{T} <: RatesModel{T}
    λ::T = 0.
    μ::T = 0.
    q::Vector{T} = Float64[]
    η::T = 0.8
end

ConstantRates(θ::NamedTuple) = ConstantRates(θ...)

function ConstantRates(wm::WhaleModel)
    @unpack λ, μ, η = wm[1].event
    ConstantRates(λ=λ, μ=μ, q=getq(wm), η=η)
end

"""
    BranchRates

Rates model with a λ and μ rate parameter for each branch in the tree,
including the root (which may represent a mean rate for instance in
a hierarchical prior setting).
"""
@with_kw struct BranchRates{T} <: RatesModel{T}
    r::Matrix{T}
    q::Vector{T} = Float64[]
    η::T = 0.9
end

function BranchRates(θ::NamedTuple)
    @unpack r, q, η = θ  # θ may contain more prameters, so extract
    BranchRates(r=r, q=q, η=η)
end

function BranchRates(wm::WhaleModel)
    r = zeros(2, nnonwgd(wm))
    q = getq(wm)
    for i=1:size(r)[2]
        @inbounds r[:,i] = [wm[i].event.λ, wm[i].event.μ]
    end
    BranchRates(r, q, wm[1].event.η)
end

function (wm::WhaleModel{T,I})(θ::R) where {T,V,I,R<:RatesModel{V}}
    d = Dict{I,WhaleNode{V,<:Event{V},I}}()
    recursivecopy!(d, θ, wm[1], nothing, wm)
    wm_ = WhaleModel(d, wm.leaves, wm.order)
    set!(wm_)
    wm_
end

function recursivecopy!(d, θ, x, y, wm)
    d[x.id] = copynode!(x, y, θ, wm)  # modifies y (parent)
    if !isleaf(x)
        for c in x.children
            recursivecopy!(d, θ, wm[c], d[x.id], wm)
        end
    end
end

copynode!(x::WhaleNode{T,Root{T}}, y, θ::ConstantRates, _) where T =
    copynode!(x, Root(θ.η, θ.λ, θ.μ))

copynode!(x::WhaleNode{T,Speciation{T}}, y, θ::ConstantRates, _) where T =
    copynode!(x, Speciation(θ.λ, θ.μ, x.event.t))

copynode!(x::WhaleNode{T,WGD{T}}, y, θ::ConstantRates, wm::WhaleModel) where T =
    copynode!(x, WGD(θ.q[x.id-nnonwgd(wm)], x.event.t))

copynode!(x::WhaleNode{T,Root{T}}, y, θ::BranchRates, _) where T =
    copynode!(x, Root(θ.η, θ.r[1,1], θ.r[2,1]))

copynode!(x::WhaleNode{T,Speciation{T}}, y, θ::BranchRates, _) where T =
    copynode!(x, Speciation(θ.r[1,x.id], θ.r[2,x.id], x.event.t))

copynode!(x::WhaleNode{T,WGD{T}}, y, θ::BranchRates, wm::WhaleModel) where T =
    copynode!(x, WGD(θ.q[x.id-nnonwgd(wm)], x.event.t))

copynode!(x::WhaleNode, ev::Event{T}) where T =
    WhaleNode(x.id, x.parent, copy(x.children), T.(x.slices), ev, copy(x.clade))

function branchlengths(wm::WhaleModel)
    b = zeros(nnonwgd(wm))
    for (i,n) in wm.nodes
        t = isroot(n) ? 0. : n.event.t
        while !isroot(n) && iswgd(wm[n.parent])
            n = n.parent
            t += n.event.t
        end
        b[i] = t
    end
    return b
end
