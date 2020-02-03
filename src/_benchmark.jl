using Test
using Distributed, BenchmarkTools
using Pkg; Pkg.activate("./test/")
using Whale, DistributedArrays

wm = Whale.WhaleModel(Whale.extree)
addwgd!(wm, 5, 0.25, rand())
ccd = read_ale("./example/example-ale", wm)
D = distribute(repeat(ccd, 100))
prior = IRPrior(Ψ=[.2 0.; 0. .2])
problem = WhaleProblem(wm, D, prior)
@btime Whale.gradient(problem, randn(36))

addprocs(3)
@everywhere begin
    using Pkg; Pkg.activate("./test/")
    using Whale, DistributedArrays
end
D = distribute(repeat(ccd, 100))
problem = WhaleProblem(wm, D, prior)
@time Whale.gradient(problem, randn(36))
@time Whale.logpdf!(wm, D)

# mapreduce over DArray
# (1) 9.177 s (1032026 allocations: 17.46 GiB)
# (2) 8.071 s (1372 allocations: 86.88 KiB)
# (3?) 5.133 s (1703 allocations: 113.77 KiB)

# spawnat
# (3) 4.662 s (1790 allocations: 115.56 KiB)

using Distributed
addprocs(3)
@everywhere using Distributed, BenchmarkTools
@everywhere using Pkg;
@everywhere Pkg.activate("./test/")
@everywhere using Whale, DistributedArrays
wm = Whale.WhaleModel(Whale.extree)
addwgd!(wm, 5, 0.25, rand())
ccd = read_ale("./example/example-ale", wm)
D = distribute(repeat(ccd, 10))
prior = IRPrior(Ψ=[.2 0.; 0. .2])
problem = WhaleProblem(wm, D, prior)
