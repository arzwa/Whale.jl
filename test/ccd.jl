using Whale
using PhyloTrees
using Test

st = Whale.example_tree()
ccd = read_ale("../example/example-ale/", st)
@test length(ccd) == 12

x = ccd[1]
@test x.Γ == 83
for (triple, p) in x.ccp
    if (isleaf(x, triple[2]) || isleaf(x, triple[3])) && triple[1] == x.Γ
        @test p == 1.0
    else
        @test p <= 1.0
    end
end
