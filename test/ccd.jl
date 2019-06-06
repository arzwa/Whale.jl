using Whale
using PhyloTrees
using Test

conf = read_whaleconf("./example/whalebay.conf")
tree = readtree("/home/arzwa/Whale.jl/example/morris-9taxa.nw")
st = SlicedTree(tree, conf)

ccd = read_ale("/home/arzwa/Whale.jl/example/example-ale/", st)
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
