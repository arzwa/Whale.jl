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
    import TransformVariables: TransformTuple, transform_and_logjac, transform
    import LogDensityProblems: logdensity_and_gradient

    import Distributions: logpdf, logpdf!

    include("model.jl")
    include("ccd.jl")
    include("core.jl")
    include("dhmc.jl")
    include("prior.jl")
    include("track.jl")

    # an example tree
    const extree = "((MPOL:4.752,PPAT:4.752):0.292,(SMOE:4.457,(((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);"

    export WhaleModel, CCD, CCDArray, read_ale, logpdf, logpdf!, addwgd!
    export WhaleProblem, CRPrior, IRPrior, ConstantRates, BranchRates
end
