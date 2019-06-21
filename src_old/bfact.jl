#= Arthur Zwaenepoel - 2019
Some functions for computing the Savage-Dickey density ratio's after a run has finished =#

function computebfs(df, wgdids)
    qcols = [col for col in names(df) if startswith(string(col), "q")];
    K = [bayesfactor(df[q]) for q in qcols]
    wgds = ["" for i =1:length(wgdids)]
    for (k, v) in wgdids; wgds[v] = k ; end
    decide.(K, wgds)
end

"""
    bayesfactor(x; reflect=true)

Compute the Bayes factor for a particular retention rate by means of the
Savage-Dickey density ratio. This uses a Kernel density estimate with
boundary correction (by reflection) to approximate the marginal posterior
density of the retention rate based on an MCMC sample for the parameter
(`x` vector).
"""
function bayesfactor(x; reflect=true)
    x = Array{Float64}(x)
    xkde = reflect ? kde([x ; -1. .* x]) : kde(x)
    k = reflect ? 2 : 1
    # when reflecting, the density values are halved (density integrates to 1)
    # k is a factor for the density dependent on whether we reflect or not
    return (pdf(xkde, 0.0)*k) / pdf(Beta(1., 1.), 0.0)
end

# print out suggested 'conclusions' based on Bayes factor
decide(K, id) = decide(K, id=id)

function decide(K; id::String="")
    @printf "WGD %8s: K = %7.3f " id K
    if K < 1/100 ;            @printf "< 1/100 → decisive evidence against H₀\n"
    elseif K < 1/(10^(3/2)) ; @printf "< 10³/² → very strong evidence against H₀\n"
    elseif K < 1/10 ;         @printf "< 1/10  → strong evidence against H₀\n"
    elseif K < 1/√10 ;        @printf "< 1/√10 → substantial evidence against H₀\n"
    elseif K < 1 ;            @printf "< 1     → H₁ supported, not worth more than a bare mention\n"
    else;                     @printf "> 1     → H₀ supported\n"; end
end

# Translated from PyMC3
function hpd(x; alpha=0.10)
    n = length(x)
    x_ = sort(x)
    credmass = 1.0 - alpha
    interval_idx_inc = floor(Int64, credmass * n)
    n_intervals = n - interval_idx_inc + 1
    interval_width = x_[interval_idx_inc:end] - x_[1:n_intervals]

    if length(interval_width) == 0
        error("Too few elements for interval calculation")
    end

    min_idx = argmin(interval_width)
    hdi_min = x_[min_idx]
    hdi_max = x_[min_idx + interval_idx_inc]
    return hdi_min, hdi_max
end
