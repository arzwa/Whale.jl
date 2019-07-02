using Whale
using DistributedArrays
using Test

st = Whale.example_tree()
ccd = read_ale("../example/example-ale/", st)

λ = repeat([0.2], 17)
μ = [repeat([0.1], 10); repeat([0.2], 7)]
q = repeat([0.2], 7)
w = WhaleModel(st, λ, μ, q, 0.9)

w_ = mle(w, DArray(ccd[1:1]), show_trace=false)
@test w_.λ[1] ≈ 0.295297477
@test sum(w_.λ) ≈ 2.158270905
@test sum(w_.μ) ≈ 4.138256349
@test sum(w_.q) ≈ 1.000002886
