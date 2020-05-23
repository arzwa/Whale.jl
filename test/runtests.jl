using Pkg; Pkg.activate(@__DIR__)
using Whale, NewickTree
using Test
using Random

@testset "likelihood" begin
    t = deepcopy(Whale.extree)
    n = length(postwalk(t))
    l = length(getleaves(t))
    insertnode!(getlca(t, "ATHA", "ATHA"), name="wgd_1")
    insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_2")
    r = Whale.RatesModel(
        DLWGD(λ=ones(n), μ=ones(n), q=[0.2, 0.1], η=0.9, p=zeros(l)),
        fixed=(:p,))
    w = WhaleModel(r, t, 0.05)
    ccd = read_ale("example/example-1/ale", w)
    @test logpdf!(w, ccd) ≈ -588.0647568294178
    @test logpdf(w, ccd) ≈ -588.0647568294178
    ccd = read_ale("example/example-1/ale", w, false)
    @test logpdf!(w, ccd) ≈ -588.0647568294178
    @test logpdf(w, ccd) ≈ -588.0647568294178
end
