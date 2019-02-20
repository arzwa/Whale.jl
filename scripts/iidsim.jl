using Distributed
using Distributions
using DistributedArrays
using ArgParse
using Logging
using CSV
using UUIDs
@everywhere using Whale

# functions ----------------------------------------------------------------------------------------
function write_header(fname)
    open(fname, "w") do f
        write(f, ",l_true,m_true,q_true,e_true,e_used,N,nu,l,m,q,L1,L0")
    end
end

function write_sim(fname, params::Dict, id::String, η, N, ν)
    open(fname, "a+") do f
        write(f, "\n", id, ",",
            join([join(params["λ"], ";"), join(params["μ"], ";"), join(params["q"], ";"),
            params["η"][1]], ","), ",", string(η), ",", string(N), ",", string(ν), ",")
    end
end

function write_results(fname, minimizer, L1, L0)
    open(fname, "a+") do f
        write(f, join([minimizer[1], minimizer[2], minimizer[3], L1, L0], ","))
    end
end

function getrates(λ1, μ1, λrange, μrange, ν, n)
    if ν == 0.
       return Dict("λ"=>zeros(n) .+ λ1, "μ"=>zeros(n) .+ μ1)
    end
    λ = rand(LogNormal(log(λ1), ν), n)
    μ = rand(LogNormal(log(μ1), ν), n)
    λ[λ .< λrange[1]] .= λrange[1]
    λ[λ .> λrange[2]] .= λrange[2]
    μ[μ .< μrange[1]] .= μrange[1]
    μ[μ .> μrange[2]] .= μrange[2]
    return Dict("λ"=>λ, "μ"=>μ)
end

function plateau(rates, r0, r1)
    rates[rates .< r0] .= r0
    rates[rates .> r1] .= r1
    return rates
end

function rm_sim(outdir)
    for f in readdir(outdir)
        nw = joinpath(outdir, f)
        rm(nw)
    end
    rm(outdir)
end

# configuration ------------------------------------------------------------------------------------
               # S = read_sp_tree("10taxa.nw")
                 S = read_sp_tree("example/morris-9taxa.nw")
           simmap1 = "10taxa.seed.simmap"
           simmap0 = "10taxa.simmap"
      wgdgc_script = "wgdgc.R"
            nrates = length(S.tree.nodes)
              conf = read_whaleconf("example/baysim.conf")
            q, ids = mark_wgds!(S, conf["wgd"])
                qs = [0.0, 0.05, 0.1, 0.2]
            slices = get_slices_conf(S, conf["slices"])
           # fname = "./sim.$(ARGS[1]).csv"
           #   dir = "./$(ARGS[1])"
             fname = "/home/arzwa/tmp/baysim1000.csv"
               dir = "/home/arzwa/tmp/baysim1000"
                Ns = [1000]
                λ1 = 0.1
                μ1 = 0.15
            λrange = (0.05, 0.25)
            μrange = (0.10, 0.50)
               #νs = [0.50]
                νs = [0.1, 0.25, 0.5]
                 n = 1
                 η = 0.66
               sim = 0

# run simulations ----------------------------------------------------------------------------------
write_header(fname)
try mkdir(dir) ; catch ; @warn "Directory `$dir` already exists"; end

if sim == 0
    if haskey(conf, "mcmc")
        prior, prop, chain = configure_mcmc(S, slices, conf["mcmc"])
    else
        error("no mcmc section inf config")
    end
    stats = []
    for N in Ns
        for i in 1:n
            for ν in νs
                d = joinpath(dir, string(uuid1())) ; mkdir(d)
                @info ν
                pars = draw_from_prior(S, prior, ν)
                pars["λ"] = plateau(pars["λ"], λrange[1], λrange[2])
                pars["μ"] = plateau(pars["μ"], μrange[1], μrange[2])
                dlsim(S, pars, N, d, oib=true, max=40)
                prof = Whale.get_profile(d, S)
                sizes = [sum(values(x)) for x in prof]
                push!(stats, [mean(sizes), minimum(sizes), maximum(sizes)])
                Whale.convert_nw_ale(d)
                write_sim(fname, pars, d, η, N, ν)
            end
        end
    end
    print(stats)
end


# entirely from prior
if sim == 1
    length(q) > 1 ? error("Only implemented for a single WGD") : @info "Starting simulations"
    for i in 1:n
        for N in Ns
            d = joinpath(dir, string(uuid1())) ; mkdir(d)
            pars = draw_from_prior(S, prior, ν)
            dlsim(S, pars, N, d, oib=true)
            Whale.convert_nw_ale(d)
            write_sim(fname, pars, d, η, N)
            ccd = read_ale_from_dir(d, S)
            out1, D = nmwhale(S, ccd, slices, η, [-1.], oib=true)
            out0, D = nmwhale(S, ccd, slices, η,  [0.], oib=true)
            write_results(fname, out1.minimizer, -out1.minimum, -out0.minimum)
        end
    end
end

# with fixed qs and eta
if sim == 2
    for N in Ns
        for i in 1:n
            for ν in νs
                for q in qs
                    d = joinpath(dir, string(uuid1())) ; mkdir(d)
                    @info ν
                    pars = getrates(λ1, μ1, λrange, μrange, ν, nrates)
                    pars["q"] = [q]
                    pars["η"] = [η]
                    dlsim(S, pars, N, d, oib=true)
                    Whale.convert_nw_ale(d)
                    write_sim(fname, pars, d, η, N, ν)
                    ccd = read_ale_from_dir(d, S)
                    out1, D = nmwhale(S, ccd, slices, η, [-1.], oib=true)
                    out0, D = nmwhale(S, ccd, slices, η,  [0.], oib=true)
                    write_results(fname, out1.minimizer, -out1.minimum, -out0.minimum)
                    rm_sim(d)
                end
            end
        end
    end
end

# with fixed qs and eta; wgdgc analysis
if sim == 3
    for N in Ns
        for i in 1:n
            for ν in νs
                for q in qs
                    d = joinpath(dir, string(uuid1())) ; mkdir(d)
                    @info ν
                    pars = getrates(λ1, μ1, λrange, μrange, ν, nrates)
                    pars["q"] = [q]
                    pars["η"] = [η]
                    dlsim(S, pars, N, d, oib=true)
                    Whale.write_profile(Whale.get_profile(d, S), d * ".profile.csv")
                    write_sim(fname, pars, d, η, N, ν)
                    λ, μ, L1, q = run_wgdgc(d * ".profile.csv", simmap1)
                    l, m, L0, _ = run_wgdgc(d * ".profile.csv", simmap0, q0=true)
                    write_results(fname, [λ, μ, q], L1, L0)
                    rm_sim(d)
                end
            end
        end
    end
end

# NOTE, works only for the 10 taxon test tree!
function run_wgdgc(profile, simmap; out="out.RData", q0=false)
    s = read(`Rscript $wgdgc_script $profile $simmap $out`, String)
    d = [x for x in s if startswith(x, "[1]") || startswith(x, "21-20")]
    λ = parse(Float64, split(d[1])[2])
    μ = parse(Float64, split(d[2])[2])
    L = parse(Float64, split(d[3])[2])
    q = q0 ? 0 : parse(Float64, split(d[4])[5])
    return λ, μ, L, q
end
