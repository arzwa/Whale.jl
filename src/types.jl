# Types for Whale.
# © Arthur Zwaenepoel - 2019
"""
    SpeciesTree(tree, node2sp::Dict)
Species tree struct, holds the species tree related information (location of
WGD nodes etc.)
"""
struct SpeciesTree <: Arboreal
    tree::Tree                       # if it has a tree and a leaves field
    leaves::Dict{Int64,String}       # it's a proper Arboreal object
    clades::Dict{Int64,Set{Int64}}   # clades under each node
    wgd_index::Dict{Int64,Int64}     # an index relating node to WGD id
    ambiguous::Dict{Int64,String}    # 'ambiguous' species (allo-assignment)

    function SpeciesTree(tree::Tree, node2sp::Dict{Int64,String})
        ambiguous = Dict{Int64,Set{String}}()
        wgd_index = Dict{Int64,Int64}()
        clades = _get_clades(tree)
        new(tree, node2sp, clades, wgd_index, ambiguous)
    end
end

# private to species tree construction
function _get_clades(tree)
    clades = Dict{Int64,Set{Int64}}()
    function walk(node)
        if isleaf(tree, node)
            clades[node] = Set([node])
        else
            clades[node] = Set([])
            for c in childnodes(tree, node)
                walk(c)
                union!(clades[node], clades[c])
            end
        end
    end
    walk(findroots(tree)[1])
    return clades
end

# NOTE: Maybe it would be better design to have a SlicedSpeciesTree type
"""
    Slices(slices, slice_lengths, branches)
Species tree slices structure.
"""
struct Slices
    slices::Dict{Int64,Int64}
    slice_lengths::Dict{Int64,Array{Float64}}
    branches::Array{Int64}  # a postorder of species tree branches

    function Slices(slices, slice_lengths, branches)
        new(slices, slice_lengths, branches)
    end
end

# NOTE: should add a field for backtracked rectrees, would be more convenient
"""
    CCD(...)
CCD composite type, holds an approximation of the posterior distribution over
trees. This version is adapted for the parallel MCMC algorithm.
"""
mutable struct CCD
    Γ::Int64                                        # ubiquitous clade
    total::Int64                                    # total # of samples
    m1::Dict{Int64,Int64}                           # counts for every clade
    m2::Dict{Int64,Array{Tuple{Int64,Int64,Int64}}} # counts for every triple
    m3::Dict{Int64,Int64}                           # leaf to species node map
    ccp::Dict{Tuple,Float64}                        # conditional clade p's
    leaves::Dict{Int64,String}                      # leaf names
    blens::Dict{Int64,Float64}                      # branch lengths for γ's'
    clades::Array{Int64,1}                          # clades ordered by size
    species::Dict{Int64,Set{Int64}}                 # clade to species nodes
    tmpmat::Dict{Int64,Array{Float64,2}}            # tmp reconciliation matrix
    recmat::Dict{Int64,Array{Float64,2}}            # the reconciliation matrix
    rectrs::Array{RecTree}                          # backtracked rectrees
    fname::String

    # the idea of the `tmpmat` and `recmat` fields is that the latter contains
    # the reconciliation matrix computed using the parameters of the last
    # completed iteration of the MCMC sample, whereas the former can hold the
    # copied and partially recomputed matrix under some new parameter values.
    # When all partial recomputations are performed we evaluate the likelihood,
    # and at that moment we must be able to, upon acceptance, store the newly
    # computed matrices (such that they can be used for partial recomputation
    # in the next iteration) **but**, upon rejection, we must be able to revert
    # to the matrices before partial recomputation. Hence the idea of working
    # on a deepcopy in `tmpmat` and setting `recmat` to `tmpmat` upon
    # acceptance. The nice thing about encoding this in the CCD type is that it
    # is straightforward (I think) to do this efficiently in a parallel setting.

    function CCD(N, m1, m2, m3, l, blens, clades, species, Γ, ccp, fname)
        m  = Dict{Int64,Array{Float64,2}}()
        m_ = Dict{Int64,Array{Float64,2}}()
        r  = RecTree[]
        new(Γ, N, m1, m2, m3, ccp, l, blens, clades, species, m, m_, r, fname)
    end
end

# display method
function Base.display(io::IO, ccd::CCD)
    @printf "CCD of %d (%d) taxa " length(ccd.leaves) length(ccd.clades)
    @printf "(clades) based on %d samples " ccd.total
end

function Base.show(io::IO, ccd::CCD)
    write(io, "CCD of $(length(ccd.leaves)) ($(length(ccd.clades))) taxa ")
    write(io, "(clades) based on $(ccd.total) samples")
end

abstract type WhaleEM end

"""
    WhaleEM(S::SpeciesTree, L::Slices, C::Array{CCD}; r, η, q)
ML inference using Expectation-Maximization for the DL(+WGD?) model.
Assuming branch-wise rates for now.
"""
mutable struct WhaleMlEM <: WhaleEM
    S::SpeciesTree
    L::Slices
    D::DArray{CCD,1,Array{CCD,1}}
    T::Dict{String,Array{RecTree,1}}  # backtracked trees
    r::Dict{Int64,Int64}
    θ::Array{LinearBDP}
    ε::Dict{Int64,Float64}
    η::Float64
    N::Int64

    function WhaleMlEM(S, L, C, r; η::Float64=0.80, N::Int64=100)
        @assert length(S.wgd_index) == 0 "EM not implemented for WGDs"
        D = distribute(C)
        T = Dict(c.fname => RecTree[] for c in C)
        nr = length(S.tree.nodes)
        θ = LinearBDP.(rand(nr), rand(nr))
        ε = get_extinction_probabilities(S, θ, r)
        new(S, L, D, T, r, θ, ε, η, N)
    end
end

"""
    WhaleMapEM(S::SpeciesTree, L::Slices, C::Array{CCD}; r, η, q)
ML inference using Expectation-Maximization for the DL(+WGD?) model.
Assuming branch-wise rates for now.
"""
mutable struct WhaleMapEM <: WhaleEM
    S::SpeciesTree
    L::Slices
    D::DArray{CCD,1,Array{CCD,1}}
    T::Dict{String,Array{RecTree,1}}  # backtracked trees
    r::Dict{Int64,Int64}
    θ::Array{LinearBDP}
    ε::Dict{Int64,Float64}
    η::Float64
    N::Int64
    πλ::Tuple{Number,Number}  # k (shape) and θ (scale) for Γ prior on λ
    πμ::Tuple{Number,Number}  # k (shape) and θ (scale) for Γ prior on μ

    function WhaleMapEM(S, L, C, r, πλ, πμ; η::Float64=0.80, N::Int64=100)
        @assert length(S.wgd_index) == 0 "EM not implemented for WGDs"
        D = distribute(C)
        T = Dict(c.fname => RecTree[] for c in C)
        nr = length(S.tree.nodes)
        θ = LinearBDP.(rand(nr), rand(nr))
        ε = get_extinction_probabilities(S, θ, r)
        new(S, L, D, T, r, θ, ε, η, N, πλ, πμ)
    end
end

"""
BackTracker(S, slices, ri, λ, μ, q, η)
A struct just for efficient passing around of data across recursions.
"""
struct BackTracker
    S::SpeciesTree
    slices::Slices
    ri::Dict{Int64,Int64}
    ε::Dict{Int64,Array{Float64}}
    ϕ::Dict{Int64,Array{Float64}}
    λ::Array{Float64}
    μ::Array{Float64}
    q::Array{Float64}
    η::Float64

    function BackTracker(em::WhaleEM)
        λ = [bdp.λ for bdp in em.θ]
        μ = [bdp.μ for bdp in em.θ]
        q = Float64[]
        return BackTracker(em.S, em.L, em.r, λ, μ, q, em.η)
    end

    BackTracker(S, slices, ri, ε, ϕ, λ, μ, q, η) = new(
        S, slices, ri, ε, ϕ, λ, μ, q, η)

    function BackTracker(S, slices, ri, λ, μ, q, η)
        ε = get_extinction_probabilities(S, slices, λ, μ, q, ri)
        ϕ = get_propagation_probabilities(S, slices, λ, μ, ε, ri)
        new(S, slices, ri, ε, ϕ, λ, μ, q, η)
    end
end
