module Whale
    using Distributed
    using PhyloTrees
    using ConsensusTrees
    using Distributions
    using DistributedArrays
    using Optim
    using ForwardDiff
    using Random
    using MCMCChains
    using DataFrames
    using CSV
    import ProgressMeter: @showprogress
    import Distributions: @check_args, logpdf

    # I guess order matters for type dependencies
    include("slicedtree.jl")
    include("ccd.jl")
    include("core.jl")
    include("mle.jl")
    include("gbm.jl")
    include("mcmc.jl")
    include("backtrack.jl")
    include("consensus.jl")

    export
        SlicedTree, WhaleModel, read_ale, logpdf, nwgd, nrates, nslices, ntaxa,
        CCD, gradient, mle, set_constantrates!, set_equalrootrates!, describe,
        WhaleChain, GBMModel, IRModel, mcmc!, backtrack!, consensus, wgds,
        contreetable, write_consensusrectrees
end
