# sample from prior and check whether prior is recovered
Random.seed!(123)
w = WhaleChain(st, IRModel(st))
@info "Doing 2000 MCMC iterations..."
chain = mcmc!(w, 2000, show_trace=false)

@testset "Independent rate model - q" begin
    for i=1:nwgd(st)
        q = w.df[Symbol("q$i")]
        @test isapprox(mean(q), 0.5, atol=0.02)
        @test isapprox(std(q), std(Beta()), atol=0.02)
    end
end

@testset "Independent rate model - η" begin
    @test isapprox(mean(w.df[:η]), mean(w.prior.η), atol=0.02)
    @test isapprox(std(w.df[:η]), std(w.prior.η), atol=0.02)
end

@testset "Independent rate model - ν" begin
    @test isapprox(mean(w.df[:ν]), mean(w.prior.ν), atol=0.02)
    @test isapprox(std(w.df[:ν]), std(w.prior.ν), atol=0.02)
end

@testset "Independent rate model - λ, μ (1)" begin
    λ1 = mean(w.df[:λ1])
    μ1 = mean(w.df[:μ1])
    ν = mean(w.df[:ν])
    @test isapprox(λ1, mean(w.prior.λ), atol=0.2)
    @test isapprox(std(w.df[:λ1]), std(w.prior.λ), atol=0.2)
    @test isapprox(μ1, mean(w.prior.μ), atol=0.2)
    @test isapprox(std(w.df[:μ1]), std(w.prior.μ), atol=0.2)
end
