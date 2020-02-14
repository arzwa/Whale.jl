using Pkg; Pkg.activate(@__DIR__)
using Whale, DistributedArrays
using Test
using Random

include("dhmc.jl")
