using UUIDs

# functions ----------------------------------------------------------------------------------------
function write_header(fname)
    open(fname, "w") do f
        write(f, ",l_true,m_true,q_true,e_true,e_used,N,l,m,q,L1,L0")
    end
end

function write_sim(fname, params::Dict, id::String)
    open(fname, "a+") do f
        write(f, "\n", id, ",",
            join([join(params["λ"], ";"), join(params["μ"], ";"), join(params["q"], ";"),
            params["η"][1]], ","))
    end
end

# configuration ------------------------------------------------------------------------------------
                 S = read_sp_tree("example/morris-9taxa.nw")
              conf = read_whaleconf("example/baysim.conf")
            q, ids = mark_wgds!(S, conf["wgd"])
            slices = get_slices_conf(S, conf["slices"])
prior, prop, chain = configure_mcmc(S, slices, conf["mcmc"])
             fname = "/home/arzwa/tmp/baysim1000.csv"
               dir = "/home/arzwa/tmp/baysim1000/"
                 N = 1000
                 n = 1

# run simulations ----------------------------------------------------------------------------------
write_header(fname)
for i in 1:n
    d = joinpath(dir, string(uuid1())) ; mkdir(d)
    pars = dlsim(S, prior, N, d)
    Whale.convert_nw_ale(d)
    write_sim(fname, pars, d)
end
