# The most basic analysis with Turing, for simulated data
using Pkg; Pkg.activate("docs")
using Whale, Distributions, Turing, DataFrames, FakeFamily, Random
Random.seed!(31)

t = deepcopy(Whale.extree)
θ = ConstantDLWGD(λ=0.1, μ=0.1, q=Float64[], η=0.66)
r = Whale.RatesModel(θ, fixed=(:p,:η))
w = WhaleModel(r, t, .1)

ts, ps = FakeFamily.dlsimbunch(Whale.root(w), w.rates, 10)
ale = FakeFamily.aleobserve(ts)
ccd = read_ale(ale, w)

@model constantrates(model, ccd, ::Type{T}=Float64) where T = begin
    r  ~ MvLogNormal(ones(2))
    ccd ~ model((λ=r[1], μ=r[2], q=T[]))
end

bmodel = constantrates(w, ccd)
chain = sample(bmodel, NUTS(0.65), 100)

pdf = DataFrame(chain)
fun = (m, x)-> Array(x) |> x->m((λ=x[1], μ=x[2], q=Float64[]))

tt = TreeTracker(w, ccd, pdf, fun)
trees = track!(tt)
_, ev = Whale.summarize(trees)
