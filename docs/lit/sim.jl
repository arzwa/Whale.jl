using Pkg; Pkg.activate(@__DIR__)
using Whale, NewickTree, Distributions, Turing, Optim
using Plots, StatsPlots, LaTeXStrings
theme(:wong2)
default(grid=:dot, legend=false, framestyle=:box, titlefont=8, title_loc=:left, guidefont=10)


# 1. Very basic simulation
t = nw"(((A:0.3,B:0.3):0.5,(C:0.6,D:0.6):0.2):0.2,E:1.);"
r = ConstantDLWGD(λ=.2, μ=0.2, η=0.7)
w = WhaleModel(r, t, 0.01)
trees, df = Whale.simulate(w, 1000)
aledir = Whale.aleobserve(trees)

data = read_ale(aledir, w)

@model constant(model, ccd, ::Type{T}=Float64) where T = begin 
    η ~ Beta()
    λ ~ Exponential()
    μ ~ Exponential()
    ccd ~ model((λ=λ, μ=μ, η=η, q=T[]))
end

chain1 = sample(constant(w, data), NUTS(), 200)


# 2. Simulation with WGD and branch rates
# =======================================
t = nw"(((((A:0.3,B:0.3):0.2)wgd_1:0.3,(C:0.6,D:0.6):0.2):0.2,E:1.);"
n = length(postwalk(t))-1
θ = rand(MvNormal(fill(log(0.1), n-1), .5), 2)
q = 0.1
r = DLWGD(λ=[θ[:,1]; NaN], μ=[θ[:,2]; NaN], q=[q], η=0.7)
w = WhaleModel(r, t, 0.01)

trees, df = Whale.simulate(w, 1000)
aledir = Whale.aleobserve(trees)

# 2.1 Analysis with branch rates model
@model branch(model, ccd, n, ::Type{T}=Float64) where T = begin 
    η ~ Beta()
    r ~ Normal(log(0.2), 1)
    τ ~ Exponential()
    d = MvNormal(fill(r, n-1), τ)
    λ ~ d
    μ ~ d
    q ~ Beta()
    ccd ~ model((λ=λ, μ=μ, η=η, q=[q]))
end

# sample
r1 = DLWGD(λ=θ[:,1], μ=θ[:,2], q=[0.1], η=0.7)
w1 = WhaleModel(r1, t, 0.01)
data = read_ale(aledir, w1)
logpdf(w1, data)

chain1 = sample(branch(w1, data, n), NUTS(), 500)

# optimize
ml = optimize(branch(w1, data, n), MLE())

# plot
l = hcat(get(chain1, :λ).λ...)
m = hcat(get(chain1, :μ).μ...)
q = vcat(get(chain1, :q).q...)

p1 = quantileplot(θ[:,2], ms=4, m, xlabel=L"\log\mu", ylabel=L"\log\hat{\mu}", color=:salmon)
p1 = quantileplot!(θ[:,1], l, ms=4, xlabel=L"\log\theta", ylabel=L"\log\hat{\theta}", title="(A)")
p2 = plot(sort(xs, rev=true), ms=3, xlabel="family", ylabel="# trees", title="(B)", yscale=:log10, ylims=(0.5,100))
p3 = plot(plot(q, ylabel=L"q", xlabel="iteration", title="(C)"), 
          histogram(q, xlabel=L"q", ylabel="density", xlims=(0,0.2),
                    color=:white), layout=(2,1))
plot(p1, p2, p3, size=(900,280), bottom_margin=5Plots.mm, left_margin=3Plots.mm, layout=(1,3))

savefig("/home/arzwa/vimwiki/phd/img/whale/simulation-example.pdf")

posterior = DataFrame(chain1)
function modelfun(model, x)
    l = [v for (k,v) in zip(names(x), values(x)) if startswith(k, "λ")]
    m = [v for (k,v) in zip(names(x), values(x)) if startswith(k, "μ")]
    model((λ=[l ; NaN], μ=[m ; NaN], η=x[:η], q=[x[:q]])) 
end
tt = TreeTracker(w1, data, posterior, modelfun)
rectrees = track(tt)

using Serialization
serialize("docs/data/sim1.jls", (θ=θ, M=w1, data=data, trees=trees, chain=chain1, rectrees=rectrees))


xs = map(1:1000) do i
    pp, t2 = rectrees[i].trees[1]
    j = parse(Int, split(rectrees[i].fname, ".")[1])
    t1 = trees[j]
    length(rectrees[i].trees)
end

# this is flawed, because when there are multiple clades at the root there
# branching is arbitrary
function getrootclades(tree)
    ns = filter(x->x.data.label ∈ ["sploss", "speciation"] && x.data.e == 9, prewalk(tree))
    f(n) = sort([name(x) for x in getleaves(n) if x.data.label != "loss"])
    sort(map(f, ns))
end

rootinfo = map(1:1000) do i
    j = parse(Int, split(rectrees[i].fname, ".")[1])
    t1 = trees[j] 
    r1 = getrootclades(t1)
    ptrue = 0.
    for (pp, t2) in rectrees[i].trees
        r2 = getrootclades(t2)
        @show r1 r2
        (r1 == r2) && (ptrue += pp)
    end
    #length(getleaves(t1)), ptrue
    i, j, ptrue
end

Whale.summarize(rectrees)


# 2.2 Analysis with constant rates model
@model constant(model, ccd, ::Type{T}=Float64) where T = begin 
    η ~ Beta()
    λ ~ Normal(log(0.2), 1)
    μ ~ Normal(log(0.2), 1)
    q ~ Beta()
    ccd ~ model((λ=exp(λ), μ=exp(μ), η=η, q=[q]))
end

data = read_ale(aledir, w)
r2 = ConstantDLWGD(λ=0.2, μ=0.2, q=[0.1], η=0.7)
w2 = WhaleModel(r2, t, 0.01)
chain2 = sample(constant(w2, data), NUTS(), 500)


# 3. Simulation to assess distance estimation
# ===========================================
using Turing, Distributions

t = nw"(((A:0.3,B:0.3):0.5,(C:0.6,D:0.6):0.2):0.2,E:1.);"
n = length(postwalk(t))
r = ConstantDLWGD(λ=1., μ=1., η=0.9)
w = WhaleModel(r, t, 0.01)
trees, df = Whale.dlsimbunch(t, r, 200, condition=:root)
aledir = Whale.aleobserve(trees)

t = nw"(((A,B),(C,D)),E);"
for n in prewalk(t)
    isroot(n) && continue
    n.data.distance = 1.
end
n = length(postwalk(t))
r = DLWGD(λ=zeros(n), μ=zeros(n), η=0.9)
w = WhaleModel(r, t, 0.01)
ccd = read_ale(aledir, w)

@model distance(model, ccd, n) = begin 
    λ ~ filldist(Exponential(), n-1)
    μ ~ filldist(Exponential(), n-1)
    q = μ[1:-1]
    ccd ~ model((λ=[log.(λ); NaN], μ=[log.(μ); NaN], q=q))
end

chain = sample(distance(w, ccd, n), NUTS(), 200)

# we should recover the branch lengths


