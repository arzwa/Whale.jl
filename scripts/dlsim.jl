#===================================================================================================
Executable for performing simulations
Arthur Zwaenepoel - 2018
===================================================================================================#
# This is motivated by the fact that all simulation studies are somehow similar, (1) simulate some
# trees; (2) perform ML estimation; (3) optionally perform ML estimation with q = 0; (4) store
# results. This is a program to do this stuff based on a generic config file, ensuring
# reproducibility.
# NOTE: Currently only supports one WGD at a time.
# NOTE: Currently supports only analyses with constant rates model

# Argument check
if length(ARGS) < 3
    # note that don't analyze makes no sense when stuff is removed, so options are unambiguous.
    @error "Usage: <script.jl> <conf> <sptree> <outdir> [profiles]"
    exit(1)
end

# Boilerplate module loading
using Distributed
using Random
using Logging
using Distributions
using UUIDs
@everywhere using Whale
@info "Modules loaded"

# main program
function main(ARGS)
    outdir = ARGS[3]
    try ; mkdir(outdir) ; catch ; @warn "Directory `$outdir` already exists"; end
    outfile = rstrip(ARGS[3], ['/']) * ".csv" ; @info "Output file: $outfile"
    write_header(outfile)

    sptree = read_sp_tree(ARGS[2])
    conf = read_whaleconf(ARGS[1])
    wgds, ids = mark_wgds!(sptree, conf["wgd"])
    slices = get_slices_conf(sptree, conf["slices"])
    if haskey(conf, "mcmc")
        prior, prop, chain = configure_mcmc(S, slices, conf["mcmc"])
    else
        λ, μ, q, η, constant_rates, λisμ, uncorr, familywise = get_rates(conf["sim"], sptree)
    end
    η_ml = conf["ml"]["e"][1]
    n = Int(conf["sim"]["n"][1])
    N = Int(conf["sim"]["N"][1])
    q0 = parse(Bool, conf["ml"]["q0"][1])
    ale = parse(Bool, conf["sim"]["ale"][1])
    ml = parse(Bool, conf["sim"]["ml"][1])
    remove = parse(Bool, conf["sim"]["remove"][1])
    profile = parse(Bool, conf["sim"]["profile"][1])

    # constant rates, with WGD
    if length(wgds) > 0 && constant_rates
        for λ_ in λ ; for μ_ in μ ; for q_ in q ; for η_ in η ; for i = 1:n
            d = do_sim(sptree, λ_, μ_, [q_], η_, N, outdir, outfile, oib=true,
                    ale=ale, profile=profile)
            if ml ; analyze(d, outfile, sptree, slices, η_ml, q0=q0, oib=true) ; end
            if remove ; rm_sim(d); end
        end ; end ; end ; end ; end
    end

    # constant rates, no WGD
    if length(wgds) == 0 && constant_rates
        for λ_ in λ ; for μ_ in μ ; for η_ in η ; for i = 1:n
            d = do_sim(sptree, λ_, μ_, Float64[], η_, N, outdir, outfile; oib=true,
                    ale=ale, profile=profile)
            if ml ; analyze(d, outfile, sptree, slices, η_ml, q0=q0, oib=true) ; end
            if remove ; rm_sim(d); end
        end ; end ; end ; end
    end

    # variable rates, with WGD
    if length(wgds) > 0 && !(constant_rates) && !(familywise)
        for i = 1:n ; for η_ in η ; for q_ in q ;
            uncorr ? λ_ = rand(Gamma(λ[1], λ[2]), λ[3]) : @error "Not implemented yet"
            uncorr ? μ_ = rand(Gamma(μ[1], μ[2]), μ[3]) : @error "Not implemented yet"
            λisμ ? λ_ = μ_ : nothing
            d = do_sim(sptree, λ_, μ_, [q_], η_, N, outdir, outfile; oib=true,
                    ale=ale, profile=profile)
            if ml ; analyze(d, outfile, sptree, slices, η_ml, q0=q0, oib=true) ; end
            if remove ; rm_sim(d); end
        end ; end ; end
    end

    # variable rates across families, constant across species
    if familywise
        @info "Simulating with family-wise constant rates model"
        for i = 1:n ; for η_ in η ; for q_ in q ;
            λ_ = rand(Gamma(λ[1], λ[2]), N)
            μ_ = rand(Gamma(μ[1], μ[2]), N)
            λisμ ? λ_ = μ_ : nothing
            d = do_sim_fw(sptree, λ_, μ_, [q_], η_, N, outdir, outfile; oib=true,
                    ale=ale, profile=profile)
            if ml ; analyze(d, outfile, sptree, slices, η_ml, q0=q0, oib=true) ; end
            if remove ; rm_sim(d); end
        end ; end ; end
    end
end

function get_rates(conf::Dict, sptree::SpeciesTree)
    λisμ = haskey(conf, "lism") ? parse(Bool, conf["lism"][1]) : false
    familywise = haskey(conf, "familywise") ? parse(Bool, conf["familywise"][1]) : false
    rates = Dict()
    constant_rates = true
    uncorr = true
    for k in ["e", "l", "m", "q"]
        s = conf[k]
        if s[1] == "l"
            rates[k] = collect(s[2]:s[4]:s[3])
        elseif s[1] == "g"
            rates[k] = exp10.(log10(s[2]):s[4]:log10(s[3]))
        elseif s[1] == "c"
            # draw correlated rates
            constant_rates = false
            uncorr = false
        elseif s[1] == "u"
            rates[k] = (s[2], s[3], length(sptree.tree.nodes)-length(sptree.wgd_index))
            constant_rates = false
        else
            rates[k] = [s[1]]
        end
    end
    return rates["l"], rates["m"], rates["q"], rates["e"], constant_rates, λisμ, uncorr, familywise
end

function analyze(d, outfile, sptree, slices, η; q0=false, oib=true)
    ccd = read_ale_from_dir(d, sptree)
    out1 = nmwhale(sptree, ccd, slices, η, [-1.], oib=true)
    if q0
        out0 = nmwhale(sptree, ccd, slices, η, [0.], oib=true)
        write_out(outfile, out1, out0, η, length(ccd))
    else
        write_out(outfile, out1, η, length(ccd))
    end
end

function write_header(fname)
    open(fname, "w") do f
        write(f, ",l_true,m_true,q_true,e_true,e_used,N,l,m,q,L1,L0")
    end
end

function write_out(fname, out1, η_ml, N)
    open(fname, "a+") do f
        write(f, ",", string(η_ml), ",", string(N))
        write(f, ",", join(out1.minimizer, ","))
        write(f, ",-", string(out1.minimum))
    end
end

function write_out(fname, out1, out0, η_ml, N)
    open(fname, "a+") do f
        write(f, ",", string(η_ml), ",", string(N))
        write(f, ",", join(out1.minimizer, ","))
        write(f, ",-", string(out1.minimum))
        write(f, ",-", string(out0.minimum))
    end
end

function write_sim(fname, λ::Float64, μ::Float64, q::Float64, η::Float64, id::String)
    open(fname, "a+") do f
        write(f, "\n", id, ",", join([λ, μ, q, η], ","))
    end
end

function write_sim(fname, λ::Array{Float64}, μ::Array{Float64}, q::Float64, η::Float64, id::String)
    open(fname, "a+") do f
        write(f, "\n", id, ",", join([join(λ, ";"), join(μ, ";"), q, η], ","))
    end
end

function do_sim(S, λ, μ, q, η, N, outdir, outfile; oib=true, ale=true, profile=false)
    d = joinpath(outdir, string(uuid1())) ; mkdir(d)
    dlsim(S, λ, μ, q, η, N, d; oib=true)
    if profile
        Whale.write_profile(Whale.get_profile(d, S), d * ".profile.csv")
    end
    ale ? do_aleobserve(d) : nothing
    write_sim(outfile, λ, μ, q[1], η, d)
    return d
end

function do_sim_fw(S, λ, μ, q, η, N, outdir, outfile; oib=true, ale=true, profile=false)
    d = joinpath(outdir, string(uuid1())) ; mkdir(d)
    for i in 1:N
        dlsim(S, λ[i], μ[i], q, η, 1, d; oib=true)
    end
    if profile
        Whale.write_profile(Whale.get_profile(d, S), d * ".profile.csv")
    end
    ale ? do_aleobserve(d) : nothing
    write_sim(outfile, mean(λ), mean(μ), q[1], η, d)
    return d
end

function rm_sim(outdir)
    for f in readdir(outdir)
        nw = joinpath(outdir, f)
        rm(nw)
    end
    rm(outdir)
end

function do_aleobserve(outdir)
    for f in readdir(outdir)
        nw = joinpath(outdir, f)
        run(`ALEobserve $nw`)
        rm(nw)
    end
end

# run
main(ARGS)
