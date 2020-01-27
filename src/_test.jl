using Pkg; Pkg.activate("./test")
using Parameters
using NewickTree
using BenchmarkTools
include("_model.jl")
include("_ccd.jl")
include("_core.jl")
