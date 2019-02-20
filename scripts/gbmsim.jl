using Distributed
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

function rm_sim(outdir)
    for f in readdir(outdir)
        nw = joinpath(outdir, f)
        rm(nw)
    end
    rm(outdir)
end

# configuration ------------------------------------------------------------------------------------
                 S = read_sp_tree("/home/arzwa/whale/whale/example/morris-9taxa.nw")
              conf = read_whaleconf("/home/arzwa/whale/whale/example/gbmsim.conf")
            q, ids = mark_wgds!(S, conf["wgd"])
                qs = exp10.(log10(0.01):0.25:log10(0.1))
            slices = get_slices_conf(S, conf["slices"])
prior, prop, chain = configure_mcmc(S, slices, conf["mcmc"])
           # fname = "./sim.$(ARGS[1]).csv"
            #  dir = "./$(ARGS[1])"
             fname = "/home/arzwa/tmp/gbmsimtest.csv"
               dir = "/home/arzwa/tmp/gbmsimtest"
                Ns = [500, 1000]
                νs = [0.1, 0.2, 0.4]
                 n = 10
                 η = 0.66
               sim = 2

# run simulations ----------------------------------------------------------------------------------
length(q) > 1 ? error("Only implemented for a single WGD") : @info "Starting simulations"
write_header(fname)
try mkdir(dir) ; catch ; @warn "Directory `$dir` already exists"; end
# entirely from prior
if sim == 1
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
                    pars = draw_from_prior(S, prior, ν)
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
