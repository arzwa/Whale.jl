using Whale, Distributions
using Plots, StatsPlots

# prior check
begin
    tree = Whale.example_tree()
    w = WhaleChain(tree,
            IRModel(λ=LogNormal(-1, 0.5),
                    μ=LogNormal(-1, 0.5),
                    ν=Exponential(0.1),
                    η=Beta(16,4)))
    w.da = true
    for i=1:100
        @time mcmc!(w, 100, show_every=100, backtrack=false)
        p = plot(stephist(w.df[!,:λ1],alpha=0.2,fill=true,color=:black,normalize=true),
                 stephist(w.df[!,:λ7],alpha=0.2,fill=true,color=:black,normalize=true),
                 stephist(w.df[!,:q1],alpha=0.2,fill=true,color=:black,normalize=true),
                 stephist(w.df[!,:η], alpha=0.2,fill=true,color=:black,normalize=true),
                 plot(log.(w.df[!,:μ4]), color=:black, linewidth=1.5),
                 plot(log.(w.df[!,:μ9]), color=:black, linewidth=1.5),
                 legend=false, grid=false)
        plot!(p[1], color=:black, w.prior.λ, linewidth=2)
        plot!(p[2], color=:black, w.prior.λ, linewidth=2)
        plot!(p[3], color=:black, w.prior.q, linewidth=2)
        plot!(p[4], color=:black, w.prior.η, linewidth=2)
        display(p)
    end
end

begin
    tree = Whale.example_tree()
    ccd = read_ale("example/example-ale/", tree)
    w = WhaleChain(tree,
            IRModel(λ=LogNormal(-1, 0.5),
                    μ=LogNormal(-1, 0.5),
                    ν=Exponential(0.1),
                    η=Beta(16,4)))
    w.da = true
    for i=1:100
        @time mcmc!(w, ccd, 100, show_every=100, backtrack=false)
        p = plot(stephist(w.df[!,:λ1],alpha=0.2,fill=true,color=:black,normalize=true),
                 stephist(w.df[!,:λ7],alpha=0.2,fill=true,color=:black,normalize=true),
                 stephist(w.df[!,:q1],alpha=0.2,fill=true,color=:black,normalize=true),
                 stephist(w.df[!,:η], alpha=0.2,fill=true,color=:black,normalize=true),
                 plot(log.(w.df[!,:μ4]), color=:black, linewidth=1.5),
                 plot(log.(w.df[!,:μ9]), color=:black, linewidth=1.5),
                 legend=false, grid=false)
        plot!(p[1], color=:black, w.prior.λ, linewidth=2)
        plot!(p[2], color=:black, w.prior.λ, linewidth=2)
        plot!(p[3], color=:black, w.prior.q, linewidth=2)
        plot!(p[4], color=:black, w.prior.η, linewidth=2)
        display(p)
    end
end
