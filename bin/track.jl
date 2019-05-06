#===============================================================================
Executable for Whale - backtracking
(c) Arthur Zwaenepoel - 2019

There are two main ways for backtracking N trees per family, one is to really
sample from the posterior, but this would require recalculating the likelihood
N times! The other is to recalculate the likelihood for the MAP estimates, which
requires a single likelihood evaluation for every family.
===============================================================================#
if !(6 <= length(ARGS) <= 7)
    @error "Usage: <track.jl> <sptree> <ale> <sample> <burnin> <N> <config> <trees?>"
    exit(1)
end

using Distributed
using Logging
using CSV
using DataFrames
sample = CSV.read(ARGS[3])[parse(Int64, ARGS[4])+1:end, :]
using DistributedArrays
using ProgressMeter
@everywhere using Whale

function main(ARGS, sample)
    S = read_sp_tree(ARGS[1])
    N = parse(Int64, ARGS[5])
    ccd = get_ccd(ARGS[2], S)
    conf = read_whaleconf(ARGS[6])
    trees = length(ARGS) == 7 ? parse(Bool, ARGS[7]) : false
    q, ids = mark_wgds!(S, conf["wgd"])
    slices = get_slices_conf(S, conf["slices"])
    D = distribute(ccd)
    if haskey(conf, "ml")
        error("ML: Not implemented")
    elseif haskey(conf, "mcmc")
        Whale.backtrackmcmcpost!(D, sample, S, slices, N; q1=false)
        prefix = conf["mcmc"]["outfile"][1]
        @info "Getting ALE-like summary ($prefix.alesum.csv)"
        df2 = Whale.alelike_summary(D, S)
        CSV.write(prefix * ".alesum.csv", df2)
        @info "Summarizing WGDs ($prefix.wgdsum.csv)"
        df1 = Whale.summarize_wgds(D, S)
        CSV.write(prefix * ".wgdsum.csv", df1)
        @info "Writing consensus reconciliations ($prefix.conrec/)"
        try mkdir("$prefix.conrec/"); catch ; end
        Whale.write_consensus_reconciliations(D, S, "$prefix.conrec/")
        if trees
            @info "Writing trees ($prefix.rectrees.xml)"
            Whale.write_rectrees(D, S, prefix * ".rectrees.xml")
            @info "Writing trees ($prefix.nw/)"
            mkdir("$prefix.nw/")
            @showprogress 1 "... and again " for c in D
                open("$prefix.nw/$(basename(k)).nws", "w") do f
                    for t in c.rectrs ; write(f, Whale.prune_loss_nodes(t)) ;
                    end
                end
            end
        end
    else
        @error "No [mcmc] or [ml] section, no idea what to do"
        exit(1)
    end
end

main(ARGS, sample)
