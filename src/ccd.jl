# Synteny amounts in Whale with a binary random varible associated with any
# internal node indicating whether or not the node is synteny-preserving.
#
# In the simplest approach to include synteny in Whale, we associate with each
# triple a binary observation indicating whether a syntenic relation is
# observed between the left and right subclade. In such a model everything is
# straightforwardly observed, and we don't have to deal with the evolution of
# synteny relationships along the species tree. In such an approach it would be
# best to only use synteny information as evidence against small-scale
# duplication, and don't use an absence of syntenic relation as evidence
# against a speciation or WGD node. This is necessary because synteny breaks
# down over time, and we would have no model for that in this case (since that
# would require full model that enables to compute the probability of loss
# of synteny down the tree, which is exactly the thing we want to avoid in
# this simple approach).
"""
    Triple
"""
struct Triple{T<:Integer}
    γ1::T
    γ2::T
    p ::Float64
end

Triple{T}(x::Tuple) where T = Triple(T(x[1]), T(x[2]), x[3])
Triple{T}(x::Tuple, count) where T = Triple(T(x[1]), T(x[2]), x[3]/count)

"""
    Clade
"""
struct Clade{T<:Integer}
    id     ::T
    count  ::Int
    splits ::Vector{Triple{T}}
    leaves ::Set{T}
    species::Set{T}
    blens  ::Float64
end

Base.length(c::Clade) = length(c.leaves)
Base.isless(c1::Clade, c2::Clade) = length(c1) < length(c2)
NewickTree.isleaf(c::Clade) = length(c.leaves) == 1

function Base.show(io::IO, c::Clade{T}) where T
    @unpack id, count = c
    write(io, "Clade{$T}(γ$id, $count, $(length(c)))")
end

"""
    CCD
"""
mutable struct CCD{T<:Integer,V<:Real}
    total  ::Int               # number of trees on which the CCD is based
    clades ::Vector{Clade{T}}  # ordered by size (small to large)!
    leaves ::Vector{String}
    ℓ      ::Vector{Matrix{V}}
    fname  ::String
    # NOTE leaves have the first clade IDs {1, ...} so we can use a vector
end

Base.length(ccd::CCD) = length(ccd.clades)
Base.lastindex(ccd::CCD) = length(ccd)
Base.getindex(ccd::CCD, i::Integer) = ccd.clades[i]
Base.show(io::IO, ccd::CCD{T,V}) where {T,V} =
    write(io, "CCD{$T,$V}(Γ=$(length(ccd)), 𝓛=$(length(ccd.leaves)))")

NewickTree.getleaves(ccd::CCD, γ) = [ccd.leaves[l] for l in ccd[γ].leaves]

CCD(s::String, wm::WhaleModel, spmap) = CCD(parse_aleobserve(s), wm, spmap)

function CCD(ale::NamedTuple, wm::WhaleModel{T,M,I}, spmap) where {T,M,I}
    @unpack Bip_counts, Dip_counts, set_id, Bip_bls, leaf_id, observations = ale
    clades = Clade{I}[]
    # get a new order, from small to large, keep leaf order intact. we don't use
    # the IDs from the ALE file, but asign new ones based on this order.
    order = sort([(length(v), k) for (k,v) in set_id])  # sort clades by size
    idmap = Dict(k => i for (i, (_, k)) in enumerate(order))  # γ => new ID map
    for (i, (_, k)) in enumerate(order)
        v = Bip_counts[k]
        t = [(idmap[t[1]], idmap[t[2]], t[3]) for t in Dip_counts[k]]
        p = Triple{I}.(t, v)
        s = getspecies(leaf_id, set_id[k], spmap)
        c = Clade(I(idmap[k]), v, p, Set(I.(set_id[k])), s, Float64(Bip_bls[k]))
        push!(clades, c)
    end
    leaves = last.(sort(collect(leaf_id)))
    ℓ = Vector{Matrix{T}}(undef, length(wm))
    for n in wm.order ℓ[id(n)] = zeros(T, length(clades), length(n)) end
    CCD(observations, clades, leaves, ℓ, ale.fname)
end

"""
    read_ale(path, wm::WhaleModel)
"""
function read_ale(s::String, wm::WhaleModel, darray=false)
    @assert ispath(s) "Not a file nor directory `$s`"
    spmap = Dict(name(l)=>id(l) for l in getleaves(root(wm)))
    ccd = if isfile(s) && endswith(s, ".ale")
        [CCD(s, wm, spmap)]
    else
        fs = isfile(s) ? readlines(s) : readdir(s, join=true)
        fs = filter(x->!startswith(x, "#"), fs) 
        tmap(f->CCD(f, wm, spmap), fs)
    end
    darray ? distribute(ccd) : ccd
end

getspecies(l, ids, spmap) = Set([spmap[split(l[id], "_")[1]] for id in ids])

"""
    CCDArray{I,T}
"""
const CCDArray{I,T} = DArray{CCD{I,T},1,Array{CCD{I,T},1}} where {T,I}

# ALEobserve parser
"""
    parse_aleobserve(fname)

Parses a `.ale` file (from ALE v0.4) in a reasonable format.
"""
function parse_aleobserve(fname)
    alestring = join(readlines(fname), "\n")
    sections = split(alestring, "#")[2:end-1]
    @assert length(sections) == 8 "Not a valid .ale file $fname"
    d = Dict{Symbol,Any}()
    for section in sections
        section = replace(section, ":\t"=>"")
        x = split(section, "\n")
        header = Symbol(replace(x[1], '-'=>'_'))
        d[header] = parse_body(x[2:end], header)
    end
    d[:leaf_id] = invert(d[:leaf_id])
    addleafclades!(d)
    addubiquitous!(d)
    (; fname=basename(fname), d...)
end

invert(d::Dict) = Dict(v=>k for(k,v) in d)

# yuck
function parse_body(xs::Array, header::Symbol)
    xs = [x for x in xs if x != ""]
    return if length(xs) == 1
        _tryparse(xs[1])
    elseif header == :Dip_counts
        d = Dict()
        for x in xs
            y = split(x)
            y1 = _tryparse(y[1])
            y2 = Tuple(_tryparse(y[2:end]))
            haskey(d, y1) ? push!(d[y1], y2) : d[y1] = [y2]
        end
        d
    elseif header == :set_id
        Dict(_tryparse(split(x)[1])=>_tryparse.(split(x)[2:end]) for x in xs)
    else
        Dict(_tryparse(split(x)[1])=>_tryparse(split(x)[2:end]) for x in xs)
    end
end

_tryparse(x::Array) = length(x) == 1 ? _tryparse(x[1]) : _tryparse.(x)

function _tryparse(x)
    y = tryparse(Int, x)
    y = isnothing(y) ? tryparse(Float64, x) : y
    isnothing(y) ? x : y
end

# NOTE this does all the confusing stuff; it replaces 'leaf ids' with their
# corresponding 'set ids', because there is no reason these should be different
# it changes the 'set ids' of non leaf clades accordingly and it adds the leaf
# clades to the Dip and Bip counts
function addleafclades!(d::Dict)
    leafclades = Dict{Int,String}()
    themap = Dict()  # 'leaf id' to 'set id'
    for (k, v) in sort!(OrderedDict(d[:set_id]))
        if length(v) == 1
            d[:Bip_counts][k] = d[:observations]
            d[:Dip_counts][k] = Tuple{Int,Int,Int}[]
            leafclades[k] = d[:leaf_id][v[1]]
            themap[v[1]] = k
        else
            d[:set_id][k] = [themap[i] for i in v]
        end
    end
    for (k, v) in themap
        d[:set_id][v] = [v]
    end
    d[:leaf_id] = leafclades
end

# adds the ubiquitous clade to the parsed ale file
function addubiquitous!(d::Dict)
    n = length(d[:leaf_id])
    l = d[:set_id]
    c = collect(keys(l))
    Γ = length(c)+1
    d[:Dip_counts][Γ] = Tuple{Int,Int,Int}[]
    # each observed clade is a potential subclade of the root, with a sister
    # clade that is also observed and uniquely determined by the set of leaves
    # and the first subclade. If there are 2n non-root clades in the sample, the
    # root (ubiquitous) clade has n possible splits, each with conditional
    # clade probability equal to the observed frequency of either subclade
    # (which are identical, cfr. assertion below). Note that the total count of
    # observations for the root clade differs from the number of samples because
    # each unrooted tree is associated with multiple (a priori equally likely)
    # rootings (`N` below, although this is a known function of the number of
    # the number of leaves and number of trees in the sample).
    N = 0
    for i=1:length(c), j=i+1:length(c)
        if length(l[i] ∩ l[j]) == 0 && length(l[i] ∪ l[j]) == n
            @assert d[:Bip_counts][i] == d[:Bip_counts][j]
            N += d[:Bip_counts][i]
            push!(d[:Dip_counts][Γ], (i, j, d[:Bip_counts][j]))
        end
    end
    triple = last(d[:Dip_counts][Γ])
    d[:Bip_counts][Γ] = N
    d[:set_id][Γ] = l[triple[1]] ∪ l[triple[2]]
    d[:Bip_bls][Γ] = 0.
end

# Something like this should be possible
# function getccd(trees::Vector{String})
#     ccd = getccd(tree)
#     for i in 2:length(trees)
#         tree = readnw(trees[i])
#         update!(ccd, tree)
#     end
#     ccd
# end
#
# function getccd(tree)
#     function walk(n)
#
#     end
# end
