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
    import LogDensityProblems: logdensity_and_gradient

    import Distributions: logpdf

    include("model.jl")
    include("ccd.jl")
    include("core.jl")
    include("dhmc.jl")

    export WhaleModel, CCD, read_ale, logpdf, logpdf!, addwgd!
    export WhaleProblem, CRPrior, IRPrior
end
