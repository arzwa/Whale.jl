#= A test suite ===============================================================#
using Distributed
@everywhere using DistributedArrays
addprocs(3)
@everywhere using Whale
using Test
using Random
using PhyloTrees
using Distributions
using Printf

@testset "Tree related basics" begin
    s_nw = "((A:3,B:2):1,C:2);"
    g_nw = "(((A_1:1,A_2:1):1,B_1:3):1,(C_1:1,C_2:9):1);"
    s, ls = read_nw(s_nw)
    g, lg = read_nw(g_nw)
    gene2sp = gene_to_species(collect(values(lg)))
    S = SpeciesTree(s, ls)
end

@testset "DL model likelihood" begin
    S = read_sp_tree("./Whale/test/data/sp_tree.nw")
    slices = get_slices(S.tree, 0.05, 5)
    ccd = read_ale_observe("./Whale/test/data/OG0001894.ale", S)
    out = nm_aledl(S, [ccd], slices, 0.95)
    @test isapprox(out.minimum, 16.98705, rtol=0.0001)
    out = nm_aledl(S, [ccd], slices, 0.95, one_in_both=false)
    @test isapprox(out.minimum, 17.07028, rtol=0.0001)
    out = nm_aledl(S, [ccd], slices, 0.75, one_in_both=false)
    @test isapprox(out.minimum, 17.30566, rtol=0.0001)
end

@testset "DL + WGD model likelihood" begin
    S = read_sp_tree("./Whale/test/data/sp_tree.nw")
    wgd_node, τ = add_wgd_node!(S, lca_node(["ATHA","ATRI"], S), τ=1.6)
    wgd_node, τ = add_wgd_node!(S, lca_node(["PPAT"], S), τ=0.9)
    slices = get_slices(S.tree, 0.05, 5)
    ccd = read_ale_observe("./Whale/test/data/OG0001894.ale", S)
    λ = 0.2 ; μ = 0.1 ; q = [1., 1.]; η = 0.75
    matrix, l = whale_likelihood(S, ccd, slices, λ, μ, q, η)
    @test isapprox(l, -16.0362, rtol=0.0001)
    out = nm_whale(S, [ccd], slices, 0.75, [-1., -1.], one_in_both=true)
    @test isapprox(-out.minimum, -14.86550; rtol=0.0001)
end

@testset "Rate at root has no influence" begin
    S = read_sp_tree("./Whale/test/data/sp_tree.nw")
    slices = get_slices(S.tree, 0.05, 5)
    ccd = read_ale_observe("./Whale/test/data/OG0001894.ale", S)
    ri = Whale.get_rateindex(S)
    λ = rand(length(keys(ri))) ; μ = rand(length(keys(ri))) ; η = 0.75
    matrix, l1 = whale_likelihood_bw(S, ccd, slices, λ, μ, Float64[], η, ri)
    λ[1] = 7.; μ[1] = 5.
    matrix, l2 = whale_likelihood_bw(S, ccd, slices, λ, μ, Float64[], η, ri)
    @test isapprox(l1, l2, rtol=0.0001)
end

#=@testset "DL + WGD model backtracking" begin
    S = read_sp_tree("./Whale/test/data/sp_tree.nw")
    wgd_node, τ = add_wgd_node!(S, lca_node(["ATHA","ATRI"], S), τ=1.6)
    wgd_node, τ = add_wgd_node!(S, lca_node(["PPAT"], S), τ=0.9)
    slices = get_slices(S.tree, 0.05, 5)
    ccd = read_ale_observe("./Whale/test/data/OG0001894.ale", S)
    λ = 0.2 ; μ = 0.1 ; q = [1., 1.]; η = 0.75
    matrix, l = whale_likelihood(S, ccd, slices, λ, μ, q, η)
    ε = get_extinction_probabilities(S, slices, λ, μ, q)
    ϕ = get_propagation_probabilities(S, slices, λ, μ, ε)
    rectree = backtrack(matrix, S, ccd, slices, λ, q, ε, ϕ, η)
    @test rectree.labels[4]  == "wgd"
    @test rectree.labels[19] == "wgd"
    @test rectree.labels[7]  == "duplication"
end =#

# This testset evaluates whether the partial likelihood evaluation is correct
@testset "Parallell likelihood evaluation" begin
    S = read_sp_tree("./Whale/test/data/10taxa.nw")
    slices = get_slices(S.tree, 0.05, 5)
    ccd = get_ccd("./Whale/test/data/sim/", S)
    D = distribute(ccd)
    rate_index = Dict(x => x for x in 1:length(S.tree.nodes))
    η = 0.75
    q = Float64[]
    λ = [0.2 for i in 1:length(S.tree.nodes)]
    μ = [0.1 for i in 1:length(S.tree.nodes)]
    @show l1 = Whale.evaluate_lhood!(D, S, slices, λ, μ, q, η, rate_index)
    Whale.set_recmat!(D)
    μ[3] = 0.4
    λ[3] = 0.3
    @show l2 = Whale.evaluate_partial!(D, 3, S, slices, λ, μ, q, η, rate_index)
    @show l3 = Whale.evaluate_lhood!(D, S, slices, λ, μ, q, η, rate_index)
    @test isapprox(l2, l3; rtol=0.0001)
    λ = [0.2 for i in 1:length(S.tree.nodes)]
    μ = [0.1 for i in 1:length(S.tree.nodes)]
    μ[9] = 0.7
    λ[9] = 0.6
    @show l2 = Whale.evaluate_partial!(D, 9, S, slices, λ, μ, q, η, rate_index)
    @show l3 = Whale.evaluate_lhood!(D, S, slices, λ, μ, q, η, rate_index)
    @test isapprox(l2, l3; rtol=0.0001)
end

@testset "Speed test" begin
    S = read_sp_tree("example/morris-9taxa.nw")
    slices = get_slices(S.tree, 0.05, 5)
    ccd = get_ccd("/home/arzwa/whale/data/9taxon-50ale/", S)
    D = distribute(ccd)
    rate_index = Dict(x => x for x in 1:length(S.tree.nodes))
    η = 0.75
    q = Float64[]
    λ = [0.2 for i in 1:length(S.tree.nodes)]
    μ = [0.1 for i in 1:length(S.tree.nodes)]
    @time Whale.evaluate_lhood!(D, S, slices, λ, μ, q, η, rate_index)
    @time Whale.evaluate_lhood!(D, S, slices, λ, μ, q, η, rate_index)
    @time Whale.evaluate_lhood!(D, S, slices, λ, μ, q, η, rate_index)
    addprocs(3)
    @everywhere using DistributedArrays
    @everywhere using Whale
    D = distribute(ccd)
    @time Whale.evaluate_lhood!(D, S, slices, λ, μ, q, η, rate_index)
    @time Whale.evaluate_lhood!(D, S, slices, λ, μ, q, η, rate_index)
    @time Whale.evaluate_lhood!(D, S, slices, λ, μ, q, η, rate_index)
    rmprocs([2,3,4])
end

