# Parallel functions
# I think it should be possible to implement the parallel part quite independent
# from the main likelihood functions etc. So I'll try to keep it well separated
# FIXME: all computations currently assume one-in-both filtering

# DistributedArrays approach
# Preliminary tests showed some nice speed-ups

# likelihood computation functions for `ppeval` on distributed arrays
# Note that all arguments to these functions should be arrays, I'm not sure why?
# the extra arguments are broadcasted to all workers as far as I understand,
# so it seems similar as declaring them with the @everywhere macro. There seems
# to be no speed difference when declaring it with @everywhere *and* passing it
# as an argument. The differences when defining all neede args everywhere and
# treating them as global vars compared to broadcasting them upon the ppeval
# call doesn't make too much a difference (1000 simulated families):
#   `global` vars: 511.176 ms (1168 allocations: 98.16 KiB)
#   `broadcasted`: 531.485 ms (1742 allocations: 155.22 KiB)
# The second (`broadcasted`) is implemented below

# Note that type annotations do not give speed ups here
function lhood_tmpmat!(x, S, slices, λ, μ, q, η, ri)
    # x is of some array type in the case D is 1 dimensional
    m, l = whale_likelihood_bw(S[1], x[1], slices[1], λ, μ, q, η, ri[1])
    x[1].tmpmat = m
    return l
end

function lhood_recmat!(x, S, slices, λ, μ, q, η, ri)
    # x is of some array type in the case D is 1 dimensional
    m, l = whale_likelihood_bw(S[1], x[1], slices[1], λ, μ, q, η, ri[1])
    x[1].recmat = m
    return l
end

function partial_lhood!(x, node, S, slices, λ, μ, q, η, ri)
    # note that the partial_recompute! function already stores the recomputed
    # matrix in the `tmpmat` field.
    return partial_recompute!(node, S[1], x[1], slices[1], λ, μ, q, η, ri[1])
end

function root_lhood!(x, S, slices, λ, μ, q, η, ri)
    # note that the recompute_at_root! function already stores the recomputed
    # matrix in the `tmpmat` field.
    return recompute_at_root!(S[1], x[1], slices[1], λ, μ, q, η, ri[1])
end

function set_recmat!(x)
    x[1].recmat = x[1].tmpmat
    return 0.
end

"""
    evaluate_lhood!(D::DArray{CCD,1,Array{CCD,1}}, S::SpeciesTree,
            slices::Slices, λ::Array{Float64}, μ::Array{Float64}, q::Float64,
            η::Float64, ri::Dict{Int64,Int64})
Evaluate the likelihood on a DArray of CCDs and store the computed
reconciliation matrices in the `recmat` field. This function could be used both
in ML optimization and the initial computation in the MCMC algorithm.
"""
function evaluate_lhood!(D, S, slices, λ, μ, q, η, ri)
    return sum(ppeval(lhood_tmpmat!, D, [S], [slices], λ, μ, q, η, [ri]))
end

"""
    evaluate_partial!(D::DArray{CCD,1,Array{CCD,1}}, node::Int64,
        S::SpeciesTree, slices::Slices, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, η::Float64, ri::Dict{Int64,Int64})
Perform partial recomputation of the reconciliation matrices on a DArray of
CCDs and store the computed reconciliation matrices in the `tmpmat` field.
Mainly for use in the MCMC algorithm.
"""
function evaluate_partial!(D::DArray{CCD,1,Array{CCD,1}}, node::Int64,
        S::SpeciesTree, slices::Slices, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, η::Float64, ri::Dict{Int64,Int64})
    return sum(ppeval(partial_lhood!, D, node, [S], [slices], λ, μ, q, η, [ri]))
end

"""
    evaluate_root!(D::DArray{CCD,1,Array{CCD,1}}, S::SpeciesTree,
        slices::Slices, λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64},
        η::Float64, ri::Dict{Int64,Int64})
Perform partial recomputation of the reconciliation matrices on a DArray of
CCDs and store the computed reconciliation matrices in the `tmpmat` field.
Mainly for use in the MCMC algorithm.
"""
function evaluate_root!(D::DArray{CCD,1,Array{CCD,1}}, S::SpeciesTree,
        slices::Slices, λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64},
        η::Float64, ri::Dict{Int64,Int64})
    return sum(ppeval(root_lhood!, D, [S], [slices], λ, μ, q, η, [ri]))
end

"""
    set_recmat!(D::DArray{CCD,1,Array{CCD,1}})
Set all `recmat` fields to the current `tmpmat`.
"""
set_recmat!(D::DArray{CCD,1,Array{CCD,1}}) = ppeval(set_recmat!, D)
