using Whale
using Test
using Random

const tests = ["slicedtree", "ccd", "ale", "mle"]
st = Whale.example_tree()
ccd = read_ale("../example/example-ale/", st)

for t in tests
    @testset "Test $t" begin
        Random.seed!(345679)
        include("$t.jl")
    end
end
