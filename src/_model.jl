# How to get type stability in the WhaleModel?
# maybe hve a look at Distributions.jl?

const extree = "((MPOL:4.752,PPAT:4.752):0.292,(SMOE:4.457,(((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);"

# event in a species tree
abstract type Event{T} end

mutable struct Root{T<:Real} <: Event{T}
    η::T
    λ::T
    μ::T
end

mutable struct Speciation{T<:Real} <: Event{T}
    λ::T
    μ::T
    t::T
end

mutable struct WGD{T<:Real} <: Event{T}
    q::T
    t::T
end

# problem: slice lengths could be different type from ϵ and ϕ (e.g. ForwardDiff)
# solution could be a Slices struct, having a vector and a matrix as fields
struct WhaleNode{T<:Real,E<:Event{T},I<:Integer}
    id      ::I
    parent  ::I            # parent node ID
    children::Set{I}       # child node IDs
    slices  ::Matrix{T}    # slice lengths and extinction and propagation Ps
    event   ::E
    clade   ::Set{I}
end

struct WhaleModel{T<:Real,I<:Integer}
    nodes ::Dict{I,WhaleNode{T,<:Event{T},I}}
    leaves::Dict{I,String}
    order ::Vector{I}
end

Base.length(wn::WhaleNode) = size(wn.slices)[1]
Base.length(wm::WhaleModel) = length(wm.nodes)
Base.show(io::IO, w::WhaleModel{T}) where {T} = write(io, "WhaleModel{$T}(N=$(length(w)))")
Base.show(io::IO, n::WhaleNode{T,E}) where {T,E} = write(io, "WhaleNode{$T,$E}")
Base.push!(n::WhaleNode, i) = push!(n.children, i)
Base.getindex(wm::WhaleModel, i) = wm.nodes[i]

NewickTree.isroot(n::WhaleNode) = typeof(n.event)<:Root
NewickTree.isleaf(n::WhaleNode) = length(n.children) == 0
children(n::WhaleNode) = n.children
parentnode(n::WhaleNode) = n.parent
iswgd(n::WhaleNode) = typeof(n.event)<:WGD

function nonwgdchild(n::WhaleNode)
    while iswgd(first(n.children))
        n = first(n.children)
    end
    n
end

WhaleNode{I}(event::Root) where I<:Integer =
    WhaleNode(I(1), I(0), Set{I}(), ones(1,3), event, Set{I}())

function WhaleNode(id::I, event::Union{Speciation,WGD},
        parent::I, Δt::T) where {T<:Real,I<:Integer}
    n = ceil(Int, event.t / Δt)  # number of slices
    M = [vcat(0.0, repeat([event.t/n], n)) ones(n+1) ones(n+1)]
    WhaleNode(id, parent, Set{I}(), M, event, Set{I}())
end

function WhaleModel(nw; Δt=0.05, λ=0.3, μ=0.2, η=0.9, I=UInt16)
    t, l, _ = readnw(nw)
    d = Dict{I,WhaleNode{Float64,<:Event{Float64},I}}(
            I(1)=>WhaleNode{I}(Root(η, λ, μ)))
    n = Dict{I,String}()
    order = I[]
    function walk(x, y)
        id = I(x.i)
        push!(order, id)
        if isroot(x)
            x_ = d[id]
        else
            d[id] = x_ = WhaleNode(id, Speciation(λ, μ, x.x), y.id, Δt)
            push!(y, id)  # add child to parent
        end
        isleaf(x) ? n[id] = l[id] : [walk(c, x_) for c in x.c]
    end
    walk(t, nothing)
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
    ϵ = first(n.children).slices[end,2]
    nextsp = wm[nonwgdchild(n)]
    n.slices[1,2] = e.q * ϵ^2 + (1. - e.q) * ϵ
    setslices!(n.slices, nextsp.λ, nextsp.μ)
end

function setslices!(A::Matrix, λ, μ)
    for i=2:size(A)[1]
        A[i,2] = ϵ_slice(λ, μ, A[i,1], A[i-1,2])
        A[i,3] = ϕ_slice(λ, μ, A[i,1], A[i-1,2])
    end
end

function ϵ_slice(λ, μ, t, ε)
    if isapprox(λ, μ, atol=1e-5)
        return 1. + (1. - ε)/(μ * (ε - 1.) * t - 1.)
    else
        return (μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ
    end
end

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

# model acrobatics
abstract type RatesModel end
struct ConstantRates <: RatesModel end

function (wm::WhaleModel{T,I})(θ::Vector{V}, r::R) where {T,V,I,R<:RatesModel}
    d = Dict{I,WhaleNode{V,<:Event{V},I}}()
    recursivecopy!(d, θ, r, wm[1], nothing)
    m = WhaleModel(d, wm.leaves, wm.order)
    set!(m)
    return m
end

function recursivecopy!(d, θ, r, x, y)
    d[x.id] = copynode!(x, y, θ, r)  # modifies y (parent)
    if !isleaf(x)
        for c in x.children
            recursivecopy!(d, θ, r, wm[c], d[x.id])
        end
    end
end

copynode!(x::WhaleNode{T,Root{T},I}, y, θ, ::ConstantRates) where {T,I} =
    WhaleNode(x.id, I(0), copy(x.children), copy(x.slices),
        Root(θ[end], θ[1], θ[2]), copy(x.clade))

copynode!(x::WhaleNode{T,Speciation{T},I}, y, θ, ::ConstantRates) where {T,I} =
    WhaleNode(x.id, x.parent, copy(x.children), copy(x.slices),
        Speciation(θ[1], θ[2], x.event.t), copy(x.clade))
