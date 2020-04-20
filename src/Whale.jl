module Whale
    using Distributed
    using Parameters
    using NewickTree
    using DistributedArrays
    using Distributions
    using ForwardDiff
    using DiffResults
    using Random
    using LinearAlgebra
    using StatsBase
    using LogDensityProblems
    using AbstractTrees
    using TransformVariables
    using DataFrames
    import TransformVariables: TransformTuple, transform_and_logjac, transform
    import LogDensityProblems: logdensity_and_gradient
    import NewickTree: id, name, distance
    import Distributions: logpdf, logpdf!

    include("rmodels.jl")
    include("model.jl")
    include("ccd.jl")
    include("core.jl")
    include("prior.jl")
    include("track.jl")
    include("rectree.jl")
    include("dhmc.jl")
    include("post.jl")

    # an example tree
    const extree1 = readnw("((MPOL:4.752,PPAT:4.752):0.292,(SMOE:4.457,(((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")
    const extree2 = readnw("((MPOL:4.752,(PPAT:2.752)wgd_1:2.0):0.292,(SMOE:4.457,((((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:1.293):1.0)wgd_2:1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")

    export WhaleModel, CCD, CCDArray, read_ale, logpdf, logpdf!
    export WhaleProblem, CRPrior, IRPrior, IWIRPrior, DLWGD, ConstantDLWGD
    export backtrack, transform, sumtrees
end
