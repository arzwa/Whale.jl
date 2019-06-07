const ConDict = Dict{Int64,Float64}

"""
    Constraints(λ, μ, q)

Equality constraints on λ, μ or q parameters, should be expressed as
dictionaries mapping branch numbers to rates, e.g.

```julia
constraints = Constraints(λ=Dict(2=>0.2, 3=>0.4), μ=Dict(1=>0.1))
```
"""
struct Constraints
    λ::ConDict
    μ::ConDict
    q::ConDict
end

Constraints(;λ=ConDict(), μ=ConDict(), q=ConDict()) = Constraints(λ, μ, q)

"""
    nmwhale(D::CCDArray, w::WhaleModel, cons::Constraints, η::Float64)

Maximum likelihood estimation using the Nelder-Mead (downhill simplex)
optimizer. Takes a `DistributedArray` of CCD objects.
"""
function nmwhale(D::CCDArray, w::WhaleModel, cons=Constraints(), η=0.9)
    var_inds = var_indices(cons, w.S)
    v = pvector(w)
    apply_constraints!(v, cons, w.S)

    function _logpdf(x::Array{Float64})
        v_ = Array(v[var_inds] .= x)
        apply_bounds!(v_, 2*nrates(w.S)+1)
        print(v_)
        m = WhaleModel(v_, w.S, η)
        lp = logpdf(D, m)
        @printf "⤷ log[P(Γ)] = %.3f\n"  lp; flush(stdout)
        return -lp
    end

    xinit = v[var_inds]
    optimizer = NelderMead(initial_simplex=Optim.AffineSimplexer())
    options = Optim.Options(g_tol=1e-6, iterations=5000)
    out = optimize(_logpdf, xinit, optimizer, options)
    minimizer = out.minimizer
    lhood = _logpdf(minimizer)
    set_recmat!(D)
    v_ = Array(v[var_inds] .= minimizer)
    apply_bounds!(v_, 2*nrates(w.S)+1)
    m = WhaleModel(v_, w.S, η)
    return m
end

# another constructor for use in numerical optimization of likelihood
function WhaleModel(x::Array{Float64}, s::SlicedTree, η::Float64)
    n = nrates(s)
    p = WhaleParams(x[1:n], x[n+1:2n], x[2n+1:end], η)
    return WhaleModel(s, p)
end

pvector(w::WhaleModel) = [w.M.λ ; w.M.μ ; w.M.q]

function var_indices(constraints::Constraints, s::SlicedTree)
    n = nrates(s)
    inds = [collect(keys(constraints.λ)) ;
            collect(keys(constraints.μ)) .+ n ;
            collect(keys(constraints.q)) .+ 2n]
    return [i for i in 1:2n+nwgd(s) if !(i in inds)]
end

function apply_constraints!(x::Array{Float64}, cons::Constraints, s::SlicedTree)
    n = nrates(s)
    for (k, off) in Dict(:λ=>0, :μ=>n, :q=>2n)
        for (i, r) in getfield(cons, k)
            x[i+off] = r
        end
    end
end

function apply_bounds!(x::Array{Float64}, qindex::Int64)
    for i in eachindex(x)
        if x[i] < 1e-8
            x[i] = 0.
        elseif x[i] > 0.9999 && i >= qindex
            x[i] = 1.
        end
    end
end
