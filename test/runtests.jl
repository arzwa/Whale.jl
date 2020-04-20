using Pkg; Pkg.activate(@__DIR__)
using Whale, DistributedArrays
using Test
using Random

@testset "logpdf!" begin
    t = Whale.extree2
    n = length(postwalk(t))
    r = DLWGD(λ=ones(n), μ=ones(n), q=[0.2, 0.1], η=0.9)
    w = WhaleModel(r, t)
    ccd = read_ale("example/example-1/ale", w)
    @test logpdf!(w, ccd) == -538.8237337383428
    @test logpdf(w, ccd) == -538.8237337383428
end

# include("dhmc.jl")
