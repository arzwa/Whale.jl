module Whale
    using Distributed
    using Parameters
    using NewickTree
    using DistributedArrays
    using Distributions
    using ForwardDiff
    using Random
    using LinearAlgebra

    using LogDensityProblems
    using TransformVariables
    using DynamicHMC
    import TransformVariables: TransformTuple, transform_and_logjac
    import LogDensityProblems: logdensity_and_gradient, transform_and_logjac

    import Distributions: logpdf

    include("model.jl")
    include("ccd.jl")
    include("core.jl")
    # include("grad.jl")
    include("dhmc.jl")

    export WhaleModel, CCD, read_ale
end
