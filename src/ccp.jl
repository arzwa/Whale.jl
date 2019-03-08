# TODO: this is long an ugly, split up and make more tidy
"""
    read_ale_observe(ale_file, species_tree)
Read the output from ALEobserve from a file. Note that the branch lengths field
is the total sum of branchlengths for that clade in the sample!
"""
function read_ale_observe(ale_file, S::SpeciesTree)
    s = open(ale_file) do file
        join(readlines(file), "\n", )
    end
    s = split(s, "#")
    if length(s) != 10
        throw(ArgumentError("Not a a valid ale file $ale_file"))
    end

    # total count
    total = parse(Int64, String(split(s[3], "\n")[2]))

    # bipartition counts
    ss = split(s[4], "\n"); ss = ss[2:end-1]
    m1 = Dict(
        parse(Int64, String(split(x, "\t")[1])) =>
        parse(Int64, String(split(x, "\t")[2])) for x in ss
    )

    # branch lengths
    ss = split(s[5], "\n"); ss = ss[2:end-1]
    blens = Dict{Int64,Float64}()
    for γ_branch_len_sum in ss
        γbl = split(γ_branch_len_sum, "\t")
        γ = parse(Int64, String(γbl[1]))
        if haskey(m1, γ)
            count = m1[γ]
        else  # leaf clade
            count = total
        end
        bl = parse(Float64, String(γbl[2])) / count
        blens[γ] = bl
    end

    # triple counts
    # not sure if dictionary of variably sized arrays is the best solution
    ss = split(s[6], "\n"); ss = ss[2:end-1]
    m2 = Dict{Int64,Array{Tuple{Int64,Int64,Int64}}}()
    for triple in ss
        t = [parse(Int64, String(x)) for x in split(triple, "\t")]
        if !(haskey(m2, t[1]))
            m2[t[1]] = Tuple[]
        end
        push!(m2[t[1]],(t[2], t[3], t[4]))
    end

    # leaf mapping, NOTE the ID assigned to the leaf NAMES, is not the same as
    # the ID assigned to the leaf CLADES, which leads to horrible confusion!
    ss = split(s[8], "\n"); ss = ss[2:length(ss)-1]
    m3_ =  Dict(
        parse(Int64, String(split(x, "\t")[2])) =>
        String(split(x, "\t")[1]) for x in ss
    )
    # m3_ now contains the IDs for gene tree leaf NAMES

    sp = reverse_labels(S.species)  # species to node names
    g2s = gene_to_species(collect(values(m3_)))
    leaf_to_spnode = Dict{Int64,Int64}()
    for (k, v) in m3_
        leaf_to_spnode[k] = sp[g2s[v]]
    end
    # leaf_to_spnode now contains leaf NAME IDs to NODES in the species tree

    # set_ids & m3
    # Set IDs contain for every clade ID the leaf NAME IDs it contains
    # So, often a single gene clade may have an ID, say 5, which refers to
    # gene 4, which is very confusing.
    m3 = Dict{Int64,Int64}()
    ss = split(s[9], "\n")
    ss = ss[2:length(ss)-1]
    set_ids = Dict{Int64,Set{Int64}}()
    species = Dict{Int64,Set{Int64}}()
    leaves = Dict{Int64,String}()
    clades_ = Tuple{Int64,Int64}[]
    for set_id in ss
        set_line = split(set_id, "\t")
        t = [parse(Int64, String(x)) for x in set_line[3:end]]
        s = [leaf_to_spnode[x] for x in t]
        clade_id = parse(Int64, set_line[1])
        if length(t) == 1 # leaf
            leaves[clade_id] = m3_[t[1]]
        end
        set_ids[clade_id] = Set(t)
        species[clade_id] = Set(s)
        push!(clades_, (length(t), clade_id))

        if length(s) == 1
            m3[clade_id] = s[1]
            m1[clade_id] = total
        end
    end
    sort!(clades_)
    clades = [x[2] for x in clades_]

    Γ = ubiquitous_clade!(
        m1, m2, clades, total, set_ids, Set(keys(leaf_to_spnode)))
    blens[Γ] = minimum(values(blens))
    species[Γ] = Set(values(m3))
    ccp = compute_ccps(m1, m2)
    ccd = CCD(total, m1, m2, m3, leaves, blens, clades, species, Γ, ccp)
    return ccd
end

# A dummy CCD object for running MCMC chains without data (only prior)
function get_dummy_ccd()
    Γ = -1
    total = -1
    m1 = Dict{Int64,Int64}()
    m2 = Dict{Int64,Array{Tuple{Int64,Int64,Int64}}}()
    m3 = Dict{Int64,Int64}()
    ccp = Dict{Tuple,Float64}()
    leaves = Dict{Int64,String}()
    blens = Dict{Int64,Float64}()
    clades = Array{Int64,1}()
    species = Dict{Int64,Set{Int64}}()
    return CCD(total, m1, m2, m3, leaves, blens, clades, species, Γ, ccp)
end

#= Bug that was in way too long!
here I acccidentaly count all bipartitions of the root twice
"""
    ubiquitous_clade!(m1, m2, clades, total, set_ids, all_leaves::Set{Int64})

Adds the ubiquitous clade to ALEobserve output. I checked with the cpp ALE
implementation and it should be correct.
"""
function ubiquitous_clade!(m1, m2, clades, total, set_ids, all_leaves::Set{Int64})
    # for some weirdo reason ALEobserve does not return the ubiquitous clade
    # triples, however since trees are unrooted, every clade is a daughter clade
    # of the ubiquitous clade, with as count for the sister clade the total
    # count for that sister clade in the sample

    # in the unrooted case, any pair of non-overlapping clades that cover all
    # leaves is a gamma' gamma'' pair for the ubiquitous clade
    @warn "bugged version"
    Γ = maximum(collect(keys(m1))) + 1
    push!(clades, Γ)
    m1[Γ] = total
    m2[Γ] = Tuple[]
    for (clade, leaves) in set_ids
        for (sister, s_leaves) in set_ids
            if length(intersect(s_leaves, leaves)) == 0 && union(s_leaves, leaves) == all_leaves
                push!(m2[Γ], (clade, sister, m1[clade]))
            end
        end
    end
    return Γ
end =#

"""
    ubiquitous_clade!(m1, m2, clades, total, set_ids, all_leaves::Set{Int64})
Adds the ubiquitous clade to ALEobserve output. I checked with the cpp ALE
implementation and it should be correct.
"""
function ubiquitous_clade!(m1, m2, clades, total, set_ids, all_leaves::Set{Int64})
    # for some weirdo reason ALEobserve does not return the ubiquitous clade
    # triples, however since trees are unrooted, every clade is a daughter clade
    # of the ubiquitous clade, with as count for the sister clade the total
    # count for that sister clade in the sample

    # in the unrooted case, any pair of non-overlapping clades that cover all
    # leaves is a gamma' gamma'' pair for the ubiquitous clade
    Γ = maximum(collect(keys(m1))) + 1
    push!(clades, Γ)
    m1[Γ] = total
    Γset = Set{Tuple}()
    for (clade, leaves) in set_ids
        for (sister, s_leaves) in set_ids
            if length(intersect(s_leaves, leaves)) == 0 && union(s_leaves, leaves) == all_leaves
                γ1, γ2 = sort([clade, sister])
                push!(Γset, (γ1, γ2, m1[clade]))
            end
        end
    end
    m2[Γ] = [x for x in Γset]
    return Γ
end

"""
    compute_ccps(m1, m2)
Compute CCP-like values for ALE (it is not the actual CCP but p(γ',γ''|γ))
"""
function compute_ccps(m1, m2)
    ccps = Dict{Tuple,Float64}()
    for (γ, triples) in m2
        for (γ1, γ2, count) in triples
            ccps[(γ, γ1, γ2)] = count / m1[γ]
        end
    end
    return ccps
end

"""
    read_ale_from_dir(dir, species_tree)
Get CCDs from a directory with ALEobserve files
"""
function read_ale_from_dir(dir::String, S::SpeciesTree)
    samples = readdir(dir)
    ccds = CCD[]
    p = Progress(length(samples), 0.1, " ⧐ Reading ALE files ...")
    for f in samples
        try
            push!(ccds, read_ale_observe(dir * "/" * f, S))
        catch
            @warn "Failed reading ALE file $(dir * "/" * f)"
        end
        next!(p)
    end
    n = length(ccds)
    @info " .. read $n ALE files"
    return ccds
end

"""
    read_ale_from_list(list, species_tree)
Get CCDs from a text file with paths to ALEobserve files
"""
function read_ale_from_list(list::String, S::SpeciesTree)
    ccds = CCD[]
    open(list, "r") do f
        lines = readlines(f)
        p = Progress(length(lines), 0.1, " ⧐ Reading ALE files ...")
        for line in lines
            try
                push!(ccds, read_ale_observe(line, S))
            catch
                @warn "Failed reading ALE file $line"
            end
            next!(p)
        end
    end
    n = length(ccds)
    @info " .. read $n ALE files"
    return ccds
end

"""
    get_ccd(ale_in::file, species_tree::SpeciesTree)
Get CCDs either a single file, list of files or directory with files generated by
ALEobserve (ALE package).
"""
function get_ccd(ale_in::String, S::SpeciesTree)
    if isfile(ale_in) && endswith(ale_in, ".ale")
        ccd = [read_ale_observe(ale_in, S)]
    elseif isfile(ale_in)
        if filesize(ale_in) == 0
            @warn "$ale_in is an empty file, will create a dummy CCD"
            ccd = [get_dummy_ccd()]
        else
            ccd = read_ale_from_list(ale_in, S)
        end
    elseif isdir(ale_in)
        ccd = read_ale_from_dir(ale_in, S)
    else
        @error "Could not read ale files, either provide a single file ending
                with `.ale`, a file with one path to a `.ale` file per line or
                a path to a directory with only `.ale` files in there."
        exit(1)
    end
end
