#===============================================================================
Executable for Whale; uses configuration files for parameter settings.
===============================================================================#
if length(ARGS) != 3
    @error "Usage: <nmwhale.jl> <sptree> <ale> <config>"
    exit(1)
end

using Distributed
using DistributedArrays
using ArgParse
using Logging
using CSV
using PhyloTrees
using ProgressMeter
@everywhere using Whale

# main routine
function main(ARGS)
    @info "Please consider the environment before running this software on large data sets"
    S = read_sp_tree(ARGS[1])
    conf = read_whaleconf(ARGS[3])
    add_ambiguous!(S, conf)
    q, ids = mark_wgds!(S, conf["wgd"])
    slices = get_slices_conf(S, conf["slices"])
    ccd = get_ccd(ARGS[2], S)
    if haskey(conf, "ml")
        out = do_ml(S, ccd, slices, q, conf["ml"]["e"][1], conf)
    elseif haskey(conf, "mcmc")
        do_mcmc(S, ccd, slices, conf, ids)
    else
        @error "No [mcmc] or [ml] section, no idea what to do"
        exit(1)
    end
end

# ML program
function do_ml(S::SpeciesTree, ccd::Array{CCD}, slices::Slices,
        q::Array{Float64}, η::Float64, config)
    @info "Maximum-likelihood estimation"
    haskey(config, "rates") ? ri = get_rateindex(S, config["rates"]) :
        ri = Dict(x => 1 for x in 1:length(S.tree.nodes))
    init = haskey(config["ml"], "init") ?
        [x for x in config["ml"]["init"]] : Float64[]
    mxit = haskey(config["ml"], "maxiter") ?
        Integer(config["ml"]["maxiter"][1]) : 5000
    @show mxit
    @info ri
    @info q
    @info init
    out, D = nmwhale(S, ccd, slices, η, q, ri, init=init, max_iter=mxit)
    @show out
    @show out.minimizer
    @show out.minimum
    haskey(config, "track") ?
        do_track_ml(out, D, S, slices, ri, η, config) : nothing
end

#XXX should go partly to src/track.jl
function do_track_ml(out, ccd, S, slices, rate_index, η, config)
    nr = length(Set(values(rate_index)))
    λh = out.minimizer[1:nr]
    μh = out.minimizer[nr+1:2nr]
    qh = out.minimizer[2nr+1:end]
    bt = BackTracker(S, slices, rate_index, λh, μh, qh, η)
    Whale.backtrack!(ccd, bt, round(Int64, config["track"]["N"][1]))
    prefix = config["track"]["outfile"][1]
    df2 = Whale.alelike_summary(ccd, S)
    CSV.write(prefix * ".alesum.csv", df2)
    df1 = Whale.summarize_wgds(ccd, S)
    CSV.write(prefix * ".wgdsum.csv", df1)
    try mkdir("$prefix.conrec/"); catch ; end
    Whale.write_consensus_reconciliations(ccd, S, "$prefix.conrec/")
    if haskey(config["track"], "trees")
        Whale.write_rectrees(ccd, S, prefix * ".rectrees.xml")
        mkdir("$prefix.nw/")
        @showprogress 1 "... and again " for c in ccd
            open("$prefix.nw/$(basename(k)).nws", "w") do f
                for t in c.rectrs ; write(f, Whale.prune_loss_nodes(t)) ; end
            end
        end
    end
    if haskey(config, "ambiguous")
        annotation = Whale.sumambiguous(ccd, S)
        Whale.write_ambiguous_annotation("$prefix.ambsum.csv", annotation)
    end
end

# MCMC and MAP programs
function do_mcmc(S, ccd, slices, conf, ids; burnin=1000)
    @info "MCMC computation"
    prior, prop, chain = configure_mcmc(S, slices, conf["mcmc"])
    if haskey(conf["mcmc"], "map")
        init = haskey(conf["mcmc"], "init") ?
            [x for x in conf["mcmc"]["init"]] : Float64[]
        mxit = haskey(conf["mcmc"], "maxiter") ?
            Integer(conf["mcmc"]["maxiter"][1]) : 5000
        out = map_nmwhale(ccd, chain, prior; init=init, max_iter=mxit)
        @show out
        @show out.minimizer
        @show out.minimum
    else
        @info "Writing samples to $(conf["mcmc"]["outfile"][1])"
        burnin = conf["mcmc"]["ngen"][1] < burnin ? 1 : burnin
        amcmc!(ccd, chain, prop, prior, conf["mcmc"]["ngen"][1],
            conf["mcmc"]["freq"][1], fname=conf["mcmc"]["outfile"][1])
        df = CSV.read(conf["mcmc"]["outfile"][1])[burnin:end, :]
        diagnostics(df)
        @info "Approximate Savage-Dickey density ratio (Bayes factors)"
        computebfs(df, ids)
    end
end

# execute
main(ARGS)
