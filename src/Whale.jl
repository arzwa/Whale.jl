__precompile__()

module Whale

    using Distributed
    using DistributedArrays
    using PhyloTrees
    using Optim
    using Distributions
    using Printf
    using Logging
    using ProgressMeter
    using Statistics
    using StatsBase
    using DataFrames  # diagnostics
    using CSV         # diagnostics
    using UUIDs
    using ColorSchemes
    using KernelDensity
    import Luxor

    export
        # types
        SpeciesTree,
        RecTree,
        CCD,
        Slices,
        BackTracker,

        # I/O
        read_sp_tree,
        read_ale_observe,
        read_ale_from_dir,
        read_ale_from_list,
        read_nw,
        write_nw,
        get_ccd,

        # misc
        lca_node,
        get_slices,
        lca_rec!,
        add_wgd_node!,
        reverse_labels,
        insert_node!,
        gene_to_species,
        read_whale_conf!,
        read_whaleconf,
        mark_wgds!,
        get_slices_conf,
        configure_mcmc,
        get_rateindex,

        # sim
        dlsim,

        # main algorithms
        get_extinction_probabilities,
        get_propagation_probabilities,
        simulate_dl,
        simulate_dl_trees,
        whale_likelihood,
        whale_likelihood_bw,
        nm_aledl,
        nm_whale,
        nm_whale_bw,
        nm_whale_parallel,
        joint_likelihood_parallel,
        update_q,
        nm_whale_bw_parallel,
        mhmcmc,
        backtrack,
        partial_recompute!,
        recompute_at_root!,
        nmwhale,
        map_nmwhale,
        backtrackmcmcpost,
        backtrackmcmcmap,

        # visualization
        # minimal_tree,
        # node_labeled_tree,
        # draw_sp_tree,
        # draw_rectree,
        # draw_colortree,

        # MCMC
        PriorSettings,
        ProposalSettings,
        ChainSettings,
        IidRates,
        GeometricBrownianMotion,
        Chain,
        UvAdaptiveProposals,
        UvProposal,
        draw_from_prior,
        mcmc!,
        amcmc!,
        diagnostics,
        bayesfactor,
        decide,
        computebfs,
        drawtree

    include("types.jl")
    include("rtree.jl")
    include("aledl.jl")
    include("optim.jl")
    include("gbmpr.jl")
    include("iidpr.jl")
    include("dmcmc.jl")
    include("diagn.jl")
    include("track.jl")
    include("pllel.jl")
    include("utils.jl")
    include("dlsim.jl")
    include("bfact.jl")
    include("ccp.jl")
    include("bdp.jl")
    include("viz.jl")

    @info "This is Whale v0.1 - © Arthur Zwaenepoel 2018-2019"
    @info "https://doi.org/10.1101/556076 "
end
