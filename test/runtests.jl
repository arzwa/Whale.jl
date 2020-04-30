using Pkg; Pkg.activate(@__DIR__)
using Whale, DistributedArrays, NewickTree
using Test
using Random

@testset "logpdf!" begin
    t = Whale.extree2
    n = length(postwalk(t))
    r = Whale.RatesModel(
        DLWGD(λ=ones(n), μ=ones(n), q=[0.2, 0.1], η=0.9, p=zeros((n+1)÷2)),
        fixed=(:p,))
    w = WhaleModel(r, t)
    ccd = read_ale("example/example-1/ale", w)
    @test logpdf!(w, ccd) == -590.3417709062691
    @test logpdf(w, ccd) == -590.3417709062691
end

# include("dhmc.jl")
