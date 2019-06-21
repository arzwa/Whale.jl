"""
    read_whaleconf(cfile::String)
Read the Whale configuration to a dictionary.
"""
function read_whaleconf(cfile::String)
    @info "Reading whale configuration"
    # defaults can be added here, make everything a tuple, that's nice
    conf = default_conf()
    if !(isfile(cfile))
        @warn "Not a file ``$cfile`; will use the default configuration"
        return conf
    end
    s = open(cfile, "r") do f
        curr_section = ""
        for line in eachline(f)
            line = strip(split(line, "#")[1])
            line = replacements(line)
            if isempty(line) || startswith(line,'#') || startswith(line,';')
                continue
            end
            if startswith(line, "[")
                curr_section = split(split(line, "[")[2], "]")[1]
                if !haskey(conf, curr_section)
                    conf[curr_section]= Dict()
                end
                continue
            end
            if curr_section == ""
                continue
            end
            if !occursin("=", line)
                continue
            end
            par_val = [strip(x) for x in split(line, "=")]
            par = par_val[1]
            val = [trytoparse(Float64, x) for x in split(par_val[2])]
            conf[curr_section][par] = Tuple(val)
        end
    end
    return conf
end

"""
    mark_wgds!(S::SpeciesTree, wgd_conf::Dict{String,Tuple})
Insert WGD nodes in the species tree.
"""
function mark_wgds!(S::SpeciesTree, wgd_conf::Dict{String,Tuple})
    q = Float64[]
    i = 1
    wgd_ids = Dict{String,Int64}()
    wgd_order = sort_wgds_by_age(wgd_conf)  # NOTE, WGDs *have* to be sorted by age (recent to old)!
    for tup in wgd_order
        k = tup[2]
        v = wgd_conf[k]
        @info "id of WGD event $k = $i"
        taxa = split(v[1], ",")
        node  = lca_node(taxa, S)
        pnode = non_wgd_parent(S, node)
        lnode = lca_node([taxa[1]], S)
        inode = (node in childnodes(S.tree, pnode)) ? node : farthest_wgd(node, S)
        t2 = distance(S.tree, pnode, lnode)
        τ = t2 - v[2]
        τ <= 0 ? error("Invalid WGD age $τ") : @info "τ = $τ"
        wgd_node, τ = add_wgd_node!(S, inode, τ=τ)
        push!(q, v[3])
        wgd_ids[k] = i
        i += 1
    end
    return q, wgd_ids
end

# get the deepest wgd node from here on
function farthest_wgd(node, S::SpeciesTree)
    n = parentnode(S.tree, node)
    while haskey(S.wgd_index, n)
        n_ = parentnode(S.tree, n)
        if !(haskey(S.wgd_index, n_))
            return n
        else
            n = n_
        end
    end
    return n
end

# sort WGDs
sort_wgds_by_age(wgd_conf) = sort([(v[2], k) for (k, v) in wgd_conf])

"""
    get_slices_conf(S::SpeciesTree, slice_conf::Dict{String,Tuple})
Get slices based on the configuration.
"""
function get_slices_conf(S::SpeciesTree, slice_conf::Dict{String,Tuple})
    return get_slices(S.tree, slice_conf["length"][1], round(Int64,
        slice_conf["min"][1]))
end

"""
    configure_mcmc(S::SpeciesTree, slices::Slices, mcmc_conf)
Configure MCMC objects based on config file (in dict).
"""
function configure_mcmc(S::SpeciesTree, slices::Slices, mcmc_conf)
    # prior on rates
    mcmc_conf["rates"][1] == "iid" ? prior = get_iidprior(mcmc_conf) :
    mcmc_conf["rates"][1] == "gbm" ? prior = get_gbmprior(mcmc_conf) :
        throw(ArgumentError("unsupported prior $(mcmc_conf["rates"])"))
    # transition kernels
    mcmc_conf["kernel"][1] == "arwalk"  ? prop = get_arwkernel(S) :
        throw(ArgumentError("unsupported kernel $(mcmc_conf["kernel"])"))
    # chain object
    chain = get_chain(S, slices, prior, mcmc_conf)
    mcmc_conf["rates"][1] == "iid" ? push!(chain.fixed_rates, 1) : nothing
    # resume a previous chain
    haskey(mcmc_conf, "resume") ? load_state!(chain, mcmc_conf["resume"][1]) : nothing
    return prior, prop, chain
end

"""
    get_rateindex(S::SpeciesTree, conf::Dict{String,Tuple})
Get rate index based on configuration file (in dict).
"""
function get_rateindex(S::SpeciesTree, conf::Dict{String,Tuple})
    if haskey(conf, "all")
        return get_rateindex(S)
    end
    classes = Int64[]
    rate_index = Dict{Int64,Int64}()
    for (k, v) in conf
        node = lca_node(split(k, ","), S)
        class = Int(v[1])
        clade = trytoparse(Bool, v[2])
        push!(classes, class)
        clade ? set_rates_clade!(node, class, rate_index, S) :
            rate_index[node] = class
    end
    length(classes) > 0 ? class = maximum(classes)+1 : class = 1
    set_rates_clade!(findroots(S.tree)[1], class, rate_index, S)
    for (n, i) in S.wgd_index
        rate_index[n] = rate_index[non_wgd_child(S, n)]
    end
    delete!(rate_index, 1)  # HACK, root has no rate.
    return rate_index
end

# handling ambiguous clades in the case of multiple subgenomes
function add_ambiguous!(S::SpeciesTree, conf)
    !haskey(conf, "ambiguous") ? (return) : nothing
    for (k,v) in conf["ambiguous"]
        @info "Adding ambiguous species ID $k → $v"
        spid = maximum(keys(S.leaves)) + 1
        S.ambiguous[spid] = k
        for sp in v
            subgenome = get_branchno(S, sp)  # leaf node/branch ID for the subgenome `sp`
            ambiguous_to_clades!(S, spid, subgenome)
        end
    end
end

function get_branchno(S::SpeciesTree, sp::String)
    return [k for (k,v) in S.leaves if v == sp][1]
end

function ambiguous_to_clades!(S::SpeciesTree, amb, subgenome)
    for (k,v) in S.clades
        (subgenome in v) ? union!(S.clades[k], amb) : nothing
    end
end

# replace greek characters
function replacements(line::AbstractString)
    line = replace(line, "λ" => "l")
    line = replace(line, "μ" => "m")
    line = replace(line, "η" => "e")
    line = replace(line, "θ" => "r")
    line = replace(line, "ν" => "v")
    return line
end

# try to parse a string to some type, else return just the string
trytoparse(T, x) = tryparse(T, x) == nothing ? string(x) : tryparse(T, x)

# default configuration
default_conf() = Dict{String,Dict{String,Tuple}}(
    "slices" => Dict("length"=>(0.05,), "min"=>(5,)),
    "wgd"    => Dict{String,Tuple}())

# Get Kernel types
# get_uniformkernel(x) = UniformProposals(x["k_e"][1], x["k_r"][1], x["k_q"][1])
# get_rwkernel(x) = RandomWalkProposals(x["k_e"][1], x["k_r"][1], x["k_q"][1])
# get_arwkernel(x, nθ, nq) = AdaptiveRandomWalk(nθ, nq, x["k_e"][1], x["k_q"][1],
#    x["k_r"][1], ν=x["k_v"][1])
get_arwkernel(S) = UvAdaptiveProposals(S)

# Get prior types
get_iidprior(x) = IidRates(x["p_l"], x["p_m"], x["p_q"], x["p_e"], x["p_v"])
get_gbmprior(x) = GeometricBrownianMotion(x["p_v"], x["p_l"], x["p_m"], x["p_q"], x["p_e"])

# Get the chain object
function get_chain(S, slices, prior::GeometricBrownianMotion, mcmc_conf)
    ri = get_rateindex(S)
    init = draw_from_prior(S, prior, length(Set(values(ri))))
    length(mcmc_conf["p_e"]) == 1 ? init["η"][1] = mcmc_conf["p_e"][1] : nothing
    length(mcmc_conf["p_v"]) == 1 ? init["ν"][1] = mcmc_conf["p_v"][1] : nothing
    chain = Chain(init, S, slices, ri)
    return chain
end

function get_chain(S, slices, prior::IidRates, mcmc_conf)
    ri = get_rateindex(S)
    init = draw_from_prior(S, prior, length(Set(values(ri))))
    length(mcmc_conf["p_e"]) == 1 ? init["η"][1] = mcmc_conf["p_e"][1] : nothing
    chain = Chain(init, S, slices, ri)
    return chain
end

# Get the default rate index
function get_rateindex(S::SpeciesTree)
    ri = Dict{Int64,Int64}()
    i = 1
    for n in sort(collect(keys(S.tree.nodes)))
        if haskey(S.wgd_index, n)
            ri[n] = ri[non_wgd_child(S, n)]
        else
            ri[n] = i ; i += 1
        end
    end
    return ri
end

# Get next non wgd node in the subtree starting from a wgd node n
function non_wgd_child(S, n)
    while haskey(S.wgd_index, n)
        n = childnodes(S.tree, n)[1]
    end
    return n
end

# load a previous chain
function load_state!(chain, fname)
    df = CSV.read(fname)
    # the columns also have lhood (last column), so discard last element for λ when
    # selecting by first letter
    λ = convert(Array, df[end, [startswith(string(var), "l") for var in names(df)]])[1:end-1]
    μ = convert(Array, df[end, [startswith(string(var), "m") for var in names(df)]])
    q = convert(Array, df[end, [startswith(string(var), "q") for var in names(df)]])
    chain.lhood = df[end, :lhood]
    chain.prior = df[end, :prior]
    chain.state["λ"] = λ
    chain.state["μ"] = μ
    chain.state["q"] = q
    chain.state["η"] = [df[end, :eta]]
    haskey(chain.state, "ν") ? chain.state["ν"] = [df[end, :nu]] : nothing
end
