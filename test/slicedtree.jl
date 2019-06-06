using Whale
using PhyloTrees
using Test

tree = readtree("../example/morris-9taxa.nw")
conf = Dict{String,Any}(
    "PPAT" => ("PPAT", 0.655, -1.0)
    "CPAP" => ("CPAP", 0.275, -1.0)
    "BETA" => ("ATHA", 0.55, -1.0)
    "ANGI" => ("ATRI,ATHA", 3.08, -1.0)
    "SEED" => ("GBIL,ATHA", 3.9, -1.0)
    "MONO" => ("OSAT", 0.91, -1.0)
    "ALPH" => ("ATHA", 0.501, -1.0)
)

add_wgds!(tree, conf)
