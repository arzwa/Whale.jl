# CCD etc.
const DPMat{T} = Dict{Int64,Array{T,2}} where T<:Real
const TripleDict = Dict{Int64,Array{Tuple{Int64,Int64,Int64}}}

Base.getindex(d::DPMat{T}, e::Int64, γ::Int64) where T<:Real = d[e][γ, :]
Base.getindex(d::DPMat{T}, e::Int64, γ::Int64, i::Int64) where T<:Real =
    d[e][γ, i]
Base.setindex!(d::DPMat{T}, x::T, e::Int64, γ::Int64, i::Int64) where T<:Real =
    d[e][γ, i] = x

"""
    CCD{<:Real,RecTree}

Conditional clade distribution with many helper fields. See [`read_ale`](@ref)
for details on IO.

```julia-repl
julia> x
CCD{Float64,PhyloTrees.RecTree}(13 taxa, 83 clades, 5001 samples)

julia> x.ccp
Dict{Tuple,Float64} with 203 entries:
  (83, 65, 66) => 0.014797
  (46, 6, 60)  => 0.0185963
  (25, 16, 23) => 1.0
  (43, 7, 38)  => 0.995601
  ⋮            => ⋮

julia> Whale.get_triples(x, 68)
3-element Array{Tuple{Int64,Int64,Int64},1}:
 (24, 75, 1)
 (2, 74, 22)
 (6, 36, 103)
```
"""
mutable struct CCD{T<:Real,RecTree}
    Γ::Int64                                        # ubiquitous clade
    total::Int64                                    # total # of samples
    m1::Dict{Int64,Int64}                           # counts for every clade
    m2::TripleDict                                  # counts for every triple
    m3::Dict{Int64,Int64}                           # leaf to species node map
    ccp::Dict{Tuple,Float64}                        # conditional clade p's
    leaves::Dict{Int64,String}                      # leaf names
    blens::Dict{Int64,Float64}                      # branch lengths for γ's'
    clades::Array{Int64,1}                          # clades ordered by size
    species::Dict{Int64,Set{Int64}}                 # clade to species nodes
    tmpmat::DPMat{T}                                # tmp reconciliation matrix
    recmat::DPMat{T}                                # the reconciliation matrix
    rectrs::Array{RecTree}                          # backtracked rectrees
    fname::String

    function CCD{T}(N, m1, m2, m3, l, blens,
            clades, species, Γ, ccp, fname) where T<:Real
        m  = DPMat{T}()
        m_ = DPMat{T}()
        r  = RecTree[]
        new{T,RecTree}(Γ, N, m1, m2, m3, ccp, l, blens, clades,
            species, m, m_, r, fname)
    end
end

# display method
function Base.display(io::IO, ccd::CCD)
    print("$(typeof(ccd))($(length(ccd.leaves)) taxa, $(length(ccd.clades))")
    print(" clades, $(ccd.total) samples)")
end

function Base.show(io::IO, ccd::CCD)
    write(io,"$(typeof(ccd))($(length(ccd.leaves)) taxa, $(length(ccd.clades))")
    write(io, " clades, $(ccd.total) samples)")
end

"""
    read_ale(fname::String, s::SlicedTree)

Read in a bunch of conditional clade distributions (CCD) from ALEobserve
(`.ale`) files. Either provide

- a file with a filename on each line
- a directory with `.ale` files
- a single `.ale` file
- an empty file, for running MCMC under the prior alone

```julia-repl
julia> st = Whale.example_tree()
SlicedTree(9, 17, 7)

julia> ccd = read_ale("example/example-ale/", st)
[ Info:  .. read 12 ALE files
12-element DistributedArrays.DArray{CCD,1,Array{CCD,1}}:
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 83 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 55 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 89 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 131 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 107 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 59 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 53 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 83 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 59 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 95 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 67 clades, 5001 samples)
 CCD{Float64,PhyloTrees.RecTree}(13 taxa, 65 clades, 5001 samples)
```
"""
function read_ale(fname::String, s::SlicedTree; d=true)
    if isfile(fname) && endswith(fname, ".ale")
        D = [read_ale_observe(fname, s)]
        @show D
    elseif isfile(fname)
        if filesize(fname) == 0
            @warn "$fname is an empty file, will create a dummy CCD"
            D = [get_dummy_ccd()]
        else
            lines = open(fname, "r") do f
                readlines(f)
            end
            lines = [x for x in lines if !startswith(x, "#")]
            D = read_ale(lines, s)
        end
    elseif isdir(fname)
        fnames = [joinpath(fname, x) for x in readdir(fname)]
        D = read_ale(fnames, s)
    else
        @error "Could not read ale files"
        return
    end
    return d ? distribute(D) : D
end

# Read a bunch of ALE files
function read_ale(fnames::Array{String,1}, s::Arboreal)
    ccds = CCD[]
    @showprogress 1 "Reading ALE files " for f in fnames
        try
            push!(ccds, read_ale_observe(f, s))
        catch x
            @warn "Failed reading ALE file $f ($x)"
        end
    end
    n = length(ccds)
    @info " .. read $n ALE files"
    return ccds
end

# Note that the branch lengths field is the total sum of branchlengths for that
# clade in the sample!
function read_ale_observe(fname::String, S::Arboreal)
    s = open(fname) do file
        join(readlines(file), "\n", )
    end
    s = split(s, "#")
    if length(s) != 10
        throw(ArgumentError("Not a a valid `.ale` file $fname"))
    end

    # total count
    total = parse(Int64, String(split(s[3], "\n")[2]))

    # bipartition counts
    ss = split(s[4], "\n")
    ss = ss[2:end-1]
    m1 = Dict(parse(Int64, String(split(x, "\t")[1])) =>
        parse(Int64, String(split(x, "\t")[2])) for x in ss)

    # branch lengths
    ss = split(s[5], "\n")
    ss = ss[2:end-1]
    blens = Dict{Int64,Float64}()
    for γ_branch_len_sum in ss
        γbl = split(γ_branch_len_sum, "\t")
        γ = parse(Int64, String(γbl[1]))
        count = haskey(m1, γ) ?  m1[γ] : total
        bl = parse(Float64, String(γbl[2])) / count
        blens[γ] = bl
    end

    # triple counts
    # not sure if dictionary of variably sized arrays is the best solution
    ss = split(s[6], "\n")
    ss = ss[2:end-1]
    m2 = TripleDict()
    for triple in ss
        t = [parse(Int64, String(x)) for x in split(triple, "\t")]
        if !(haskey(m2, t[1]))
            m2[t[1]] = Tuple[]
        end
        push!(m2[t[1]],(t[2], t[3], t[4]))
    end

    # leaf mapping, NOTE the ID assigned to the leaf NAMES, is not the same as
    # the ID assigned to the leaf CLADES, which leads to horrible confusion!
    ss = split(s[8], "\n")
    ss = ss[2:length(ss)-1]
    m3_ =  Dict(parse(Int64, String(split(x, "\t")[2])) =>
        String(split(x, "\t")[1]) for x in ss)
    # m3_ now contains the IDs for gene tree leaf NAMES

    sp = Dict(v=>k for (k,v) in S.leaves)
    g2s = gene_to_species(collect(values(m3_)))
    leaf_to_spnode = Dict{Int64,Int64}()
    for (k, v) in m3_
        leaf_to_spnode[k] = sp[g2s[v]]
    end
    # leaf_to_spnode now contains leaf NAME IDs to NODES in the species tree

    # set_ids & m3
    # Set IDs contain for every clade ID the leaf NAME IDs it contains
    # So, often a single gene clade may have an ID, say 5, which refers to
    # gene 4, which is very confusing.
    m3 = Dict{Int64,Int64}()
    ss = split(s[9], "\n")
    ss = ss[2:length(ss)-1]
    set_ids = Dict{Int64,Set{Int64}}()
    species = Dict{Int64,Set{Int64}}()
    leaves = Dict{Int64,String}()
    clades_ = Tuple{Int64,Int64}[]
    for set_id in ss
        set_line = split(set_id, "\t")
        t = [parse(Int64, String(x)) for x in set_line[3:end]]
        s = [leaf_to_spnode[x] for x in t]
        clade_id = parse(Int64, set_line[1])
        if length(t) == 1 # leaf
            leaves[clade_id] = m3_[t[1]]
        end
        set_ids[clade_id] = Set(t)
        species[clade_id] = Set(s)
        push!(clades_, (length(t), clade_id))

        if length(s) == 1
            m3[clade_id] = s[1]
            m1[clade_id] = total
        end
    end
    sort!(clades_)
    clades = [x[2] for x in clades_]

    Γ = ubiquitous_clade!(m1, m2, clades, total, set_ids,
            Set(keys(leaf_to_spnode)))
    blens[Γ] = minimum(values(blens))
    species[Γ] = Set(values(m3))
    ccp = compute_ccps(m1, m2)
    ccd = CCD{Float64}(total, m1, m2, m3, leaves,
        blens, clades, species, Γ, ccp, fname)
    return ccd
end

# dummy CCD (for running MCMC under the prior alone)
function get_dummy_ccd()
    m1 = Dict{Int64,Int64}()
    m2 = TripleDict()
    m3 = Dict{Int64,Int64}()
    ccp = Dict{Tuple,Float64}()
    blens = Dict{Int64,Float64}()
    leaves = Dict{Int64,String}()
    clades = Array{Int64,1}()
    species = Dict{Int64,Set{Int64}}()
    return CCD{Float64}(-1, m1, m2, m3, leaves, blens, clades,
        species, -1, ccp, "dummy")
end

# get the ubiquitous clade, private to read_ale
function ubiquitous_clade!(m1, m2, clades, total, set_ids, alleaves::Set{Int64})
    # In the unrooted case, any pair of non-overlapping clades that cover all
    # leaves is a gamma' gamma'' pair for the ubiquitous clade
    Γ = maximum(collect(keys(m1))) + 1
    push!(clades, Γ)
    m1[Γ] = total
    Γset = Set{Tuple}()
    for (clade, leaves) in set_ids
        for (sister, s_leaves) in set_ids
            if length(intersect(s_leaves, leaves)) == 0 &&
                    union(s_leaves, leaves) == alleaves
                γ1, γ2 = sort([clade, sister])
                push!(Γset, (γ1, γ2, m1[clade]))
            end
        end
    end
    m2[Γ] = [x for x in Γset]
    return Γ
end

# compute conditional clade probabilities
function compute_ccps(m1::Dict{Int64,Int64}, m2::TripleDict)
    ccps = Dict{Tuple,Float64}()
    for (γ, triples) in m2
        for (γ1, γ2, count) in triples
            ccps[(γ, γ1, γ2)] = count / m1[γ]
        end
    end
    return ccps
end

gene_to_species(genes::Array{String}) = Dict(
    x => String(split(x, ['|', '_'])[1]) for x in genes)

getp(x::CCD, e::Int64, γ::Int64, i::Int64) = x.tmpmat[e][γ, i]
setp!(x::CCD, e::Int64, γ::Int64, i::Int64, p::Float64) = x.tmpmat[e][γ, i] = p
addp!(x::CCD, e::Int64, γ::Int64, i::Int64, p::Float64) = x.tmpmat[e][γ, i] += p

get_triples(x::CCD, γ::Int64) = x.m2[γ]
get_species(x::CCD, γ::Int64) = x.species[γ]
PhyloTrees.isleaf(x::CCD, γ::Int64) = haskey(x.m3, γ)
