using Pkg; Pkg.activate("./test")
using Whale, DistributedArrays
using Test
using Random

include("dhmc.jl")
