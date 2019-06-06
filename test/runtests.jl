using Whale
using Test

const tests = ["slicedtree.jl"]

for t in tests
    @testset "Test $t" begin
        Random.seed!(345679)
        include("$t.jl")
    end
end
