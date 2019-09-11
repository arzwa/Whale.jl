using Distributed
using DistributedArrays
import Whale: CCDArray

st = Whale.example_tree()
ccd = read_ale("example/example-ale/", st, d=false)
w = WhaleModel(st)

# benchmark of mapreduce vs. ppeval
logpdf1(m::WhaleModel, x::CCDArray, node::Int64=-1; matrix=false) =
    mapreduce((x)->logpdf(m, x, node, matrix=matrix), +, x)

logpdf2(m::WhaleModel, x::CCDArray, node::Int64=-1; matrix=false) =
    sum(ppeval((x)->logpdf(m, x[1], node, matrix=matrix), x))

for i=1:4
    addprocs(i)
    println("Workers ($(length(workers()))): $(workers())")
    @everywhere using Whale
    D = distribute(vcat([ccd for i=1:100]...))
    println("→ logpdf1")
    logpdf1(w, D)
    for j=1:5
        @time logpdf1(w, D)
    end
    println("→ logpdf2")
    logpdf2(w, D)
    for j=1:5
        @time logpdf2(w, D)
    end
    rmprocs(workers())
end

#=
Mapreduce approach is more efficient:
Workers (1): [22]
→ logpdf1
  1.315542 seconds (367 allocations: 52.297 KiB)
  1.200037 seconds (360 allocations: 51.656 KiB)
  1.144087 seconds (360 allocations: 51.656 KiB)
  1.187364 seconds (360 allocations: 51.656 KiB)
  1.054081 seconds (360 allocations: 51.656 KiB)
→ logpdf2
  1.070780 seconds (749 allocations: 69.656 KiB)
  1.093602 seconds (748 allocations: 69.172 KiB)
  1.066963 seconds (751 allocations: 69.297 KiB)
  1.057493 seconds (746 allocations: 69.563 KiB)
  1.116675 seconds (746 allocations: 69.141 KiB)
Workers (2): [23, 24]
→ logpdf1
  0.584191 seconds (688 allocations: 102.266 KiB)
  0.612497 seconds (685 allocations: 101.750 KiB)
  0.606580 seconds (685 allocations: 101.750 KiB)
  0.602477 seconds (690 allocations: 102.234 KiB)
  0.603560 seconds (693 allocations: 102.031 KiB)
→ logpdf2
  0.633401 seconds (1.31 k allocations: 129.281 KiB)
  0.627622 seconds (1.30 k allocations: 129.875 KiB)
  0.626569 seconds (1.35 k allocations: 130.984 KiB)
  0.622540 seconds (1.32 k allocations: 129.953 KiB)
  0.608926 seconds (1.31 k allocations: 130.000 KiB)
Workers (3): [25, 26, 27]
→ logpdf1
  0.644268 seconds (7.99 k allocations: 497.094 KiB, 3.39% gc time)
  0.910332 seconds (1.01 k allocations: 152.203 KiB)
  0.708245 seconds (1.02 k allocations: 152.422 KiB)
  0.640960 seconds (1.02 k allocations: 152.219 KiB)
  0.583540 seconds (1.02 k allocations: 152.547 KiB)
→ logpdf2
  0.582000 seconds (1.86 k allocations: 190.172 KiB)
  0.587359 seconds (1.85 k allocations: 190.125 KiB)
  0.604743 seconds (1.85 k allocations: 190.141 KiB)
  0.631299 seconds (1.84 k allocations: 189.453 KiB)
  0.614458 seconds (1.85 k allocations: 189.984 KiB)
Workers (4): [28, 29, 30, 31]
→ logpdf1
  0.523556 seconds (1.34 k allocations: 202.641 KiB)
  0.552655 seconds (1.34 k allocations: 202.125 KiB)
  0.538408 seconds (1.34 k allocations: 202.250 KiB)
  0.549300 seconds (1.34 k allocations: 202.703 KiB)
  0.544134 seconds (1.34 k allocations: 202.188 KiB)
→ logpdf2
  0.571677 seconds (2.41 k allocations: 250.234 KiB)
  0.547531 seconds (2.40 k allocations: 249.984 KiB)
  0.564947 seconds (2.40 k allocations: 250.063 KiB)
  0.578885 seconds (2.40 k allocations: 250.031 KiB)
  0.557662 seconds (2.41 k allocations: 250.688 KiB)
=#
