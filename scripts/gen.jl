using Gen

@gen function iidmodel(xs::Vector)
    ν = @trace(inv_gamma(100., 1.), :ν)
    η = @trace(beta(10, 2), :η)
    q = zeros(nwgd(st))
    for i in eachindex(q)
        q[i] = @trace(beta(1, 1), "q-$i")
    end
    r = @trace(exponential(0.2), :r)
    λ = zeros(nrates(st))
    μ = zeros(nrates(st))
    for i in eachindex(λ)
        λ[i] = @trace(exponential(r), "λ-$i")
        μ[i] = @trace(exponential(r), "μ-$i")
    end
end
