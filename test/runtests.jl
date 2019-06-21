using Whale
using Test
using Random

const tests = ["slicedtree", "ccd"]

for t in tests
    @testset "Test $t" begin
        Random.seed!(345679)
        include("$t.jl")
    end
end
