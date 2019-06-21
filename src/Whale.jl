module Whale
    using Distributed
    using PhyloTrees
    using Distributions
    using DistributedArrays
    using Optim
    import ProgressMeter: @showprogress
    import Distributions: @check_args, logpdf

    # I guess order matters for type dependencies
    include("slicedtree.jl")
    include("ccd.jl")
    include("core3.jl")
    #include("$base/gbm.jl")
    #include("$base/mle.jl")
    #include("$base/mcmc.jl")

    export
        SlicedTree, WhaleModel, read_ale, logpdf, nwgd, nrates, nslices, ntaxa
end
