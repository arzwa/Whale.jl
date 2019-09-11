using Whale
using Test
using Random
using Distributions
using Logging

const tests = ["slicedtree", "ccd", "ale", "mle", "mcmc-prior"]
st = Whale.example_tree()
ccd = read_ale("../example/example-ale/", st)

for t in tests
    @testset "Test $t" begin
        Random.seed!(345679)
        include("$t.jl")
    end
end
