using Whale
using PhyloTrees
using Test

#tree = readtree("../example/morris-9taxa.nw")
st = Whale.example_tree()

@test nslices(st, 1) == 1
@test nslices(st, 6) == 91
