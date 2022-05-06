module Whale
    using Distributed
    using Parameters
    using DataStructures
    using NewickTree
    using Distributions
    using Random
    using StatsBase
    using DataFrames
    using Base.Threads
    using ThreadTools
    using RecipesBase
    import NewickTree: id, name, distance
    import Distributions: logpdf, logpdf!

    include("rmodels.jl")
    include("bdputil.jl")
    include("model.jl")
    include("condition.jl")
    include("ccd.jl")
    include("core.jl")
    include("track.jl")
    include("rectree.jl")
    include("post.jl")
    include("recplot.jl")
    include("simulation.jl")
    include("lca.jl")

    # should include some example data...
    # an example tree
    const extree = readnw("((MPOL:4.752,PPAT:4.752):0.292,(SMOE:4.457,((("*
                          "OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0"*
                          ".738,ATRI:2.293):1.225,(GBIL:3.178,PABI:3.178)"*
                          ":0.34):0.939):0.587);")
    export WhaleModel, CCD, read_ale, logpdf, logpdf!, DLWGD, ConstantDLWGD
    export TreeTracker, track, transform, sumtrees, RatesModel, transform
end
