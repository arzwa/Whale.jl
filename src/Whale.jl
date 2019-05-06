module Whale
    using PhyloTrees
    using ConsensusTrees
    using BirthDeathProcesses
    using Distributed
    using DistributedArrays
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
    using KernelDensity

    export
        # types
        SpeciesTree, CCD, Slices, BackTracker, WhaleEM, WhaleMapEM, WhaleMlEM,
        # I/O
        read_sp_tree, read_ale_observe, read_ale_from_dir, read_ale_from_list,
        get_ccd,
        # misc
        lca_node, get_slices, lca_rec!, add_wgd_node!, reverse_labels,
        insert_node!, gene_to_species, read_whale_conf!, read_whaleconf,
        mark_wgds!, get_slices_conf, configure_mcmc, get_rateindex,
        add_ambiguous!,
        # sim
        dlsim,
        # main algorithms
        get_extinction_probabilities, get_propagation_probabilities,
        simulate_dl, simulate_dl_trees, whale_likelihood, whale_likelihood_bw,
        nm_aledl, nm_whale, nm_whale_bw, nm_whale_parallel,
        joint_likelihood_parallel, update_q, nm_whale_bw_parallel, mhmcmc,
        backtrack, partial_recompute!, recompute_at_root!, nmwhale, map_nmwhale,
        backtrackmcmcpost!,
        # MCMC
        PriorSettings, ProposalSettings, ChainSettings, IidRates,
        GeometricBrownianMotion, Chain, UvAdaptiveProposals, UvProposal,
        draw_from_prior, mcmc!, amcmc!, diagnostics, bayesfactor, decide,
        computebfs

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
    include("cnsns.jl")
    include("ccp.jl")
    include("bdp.jl")
    include("em.jl")

    @info "This is Whale v0.2 - © Arthur Zwaenepoel 2018-2019"
    @info "https://doi.org/10.1101/556076 "
end
