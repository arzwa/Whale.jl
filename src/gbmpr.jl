# Arthur Zwaenepoel - 2018
# Geometric Brownian Motion Prior
abstract type PriorSettings end

"""
    GeometricBrownianMotion(ν, θ, q, η)
Parametrization of the Geometric Brownian Motion priors.
"""
struct GeometricBrownianMotion <: PriorSettings
    prior_ν::Distribution{Univariate,Continuous}
    prior_λ::Distribution{Univariate,Continuous}  # prior on rate at root
    prior_μ::Distribution{Univariate,Continuous}  # prior on rate at
    prior_q::Distributions.Beta{Float64}    # retention rates ∼ Beta
    prior_η::Distributions.Beta{Float64}    # geometric prior on root ∼ Beta
    fixed_ν::Bool
    fixed_η::Bool

    # Gamma priors on rates
    function GeometricBrownianMotion(ν::Tuple, λ::Tuple{Float64,Float64},
            μ::Tuple{Float64,Float64}, q::Tuple{Float64,Float64}, η::Tuple)
        @info "Geometric brownian motion with lognormal priors on λ & μ"
        @info "ν fixed = $(length(ν) == 1); $ν"
        prior_λ = LogNormal(log(λ[1]), λ[2])
        prior_μ = LogNormal(log(μ[1]), μ[2])
        prior_q =  Beta(q[1], q[2])
        prior_η = length(η) == 1 ? Beta(η[1], 1.) : Beta(η[1], η[2])
        prior_ν = length(ν) == 1 ? LogNormal(log(ν[1]), 1.) : LogNormal(log(ν[1]), ν[2])
        new(prior_ν, prior_λ, prior_μ, prior_q, prior_η, length(ν) == 1, length(η) == 1)
    end

    # exponential priors
    function GeometricBrownianMotion(ν::Tuple, λ::Tuple{Float64},
            μ::Tuple{Float64}, q::Tuple{Float64,Float64}, η::Tuple)
        @info "Geometric brownian motion with exponential priors on λ & μ"
        @info "ν fixed = $(length(ν) == 1); $ν"
        prior_λ = Exponential(λ)
        prior_μ = Exponential(μ)
        prior_q =  Beta(q[1], q[2])
        prior_η = length(η) == 1 ? Beta(η[1], 1.) : Beta(η[1], η[2])
        prior_ν = length(ν) == 1 ? LogNormal(log(ν[1]), 1.) : LogNormal(log(ν[1]), ν[2])
        new(prior_ν, prior_λ, prior_μ, prior_q, prior_η, length(ν) == 1, length(η) == 1)
    end
end

# For the GBM prior we need rate sassigned to **nodes**
"""
    draw_from_prior(S::SpeciesTree, prset::GeometricBrownianMotion)
Take a draw from the GBM prior, return **node**-wise rates (λ, μ) and q.
"""
function draw_from_prior(S::SpeciesTree, prset::GeometricBrownianMotion, args...)
    ν = rand(prset.prior_ν)
    λroot = rand(prset.prior_λ)
    μroot = rand(prset.prior_μ)
    λ = draw_from_gbm(S, λroot, ν)
    μ = draw_from_gbm(S, μroot, ν)
    q = rand(prset.prior_q, length(S.wgd_index))
    η = rand(prset.prior_η)
    return Dict("λ" => λ, "μ" => μ, "q" => q, "η" => [η], "ν" => [ν])
end

"""
    draw_from_prior(S::SpeciesTree, prset::GeometricBrownianMotion)
Take a draw from the GBM prior, return **node**-wise rates (λ, μ) and q.
"""
function draw_from_prior(S::SpeciesTree, prset::GeometricBrownianMotion, ν::Float64)
    λroot = rand(prset.prior_λ)
    μroot = rand(prset.prior_μ)
    λ = draw_from_gbm(S, λroot, ν)
    μ = draw_from_gbm(S, μroot, ν)
    q = rand(prset.prior_q, length(S.wgd_index))
    η = rand(prset.prior_η)
    return Dict("λ" => λ, "μ" => μ, "q" => q, "η" => [η], "ν" => [ν])
end

"""
    draw_from_gbm(S::SpeciesTree, r0::Float64, σ::Float64)
Take a draw from the GBM model for the rates. `r0` is the rate at the root and
`σ` is the standard deviation of the Brownian motion.
"""
function draw_from_gbm(S::SpeciesTree, r0::Float64, ν::Float64)
    r = zeros(length(S.tree.nodes) - length(S.wgd_index)) .- 1
    r[1] = r0
    function walk(node)
        if !isroot(S.tree, node) && !(haskey(S.wgd_index, node))  # skip WGDs
            pnode = non_wgd_parent(S, node)
            t = distance(S.tree, node, pnode)
            # we draw a log-rate, so don't forget to exponentiate!
            r[node] = exp(rand(Normal(log(r[pnode]) - ν^2*t/2, √t*ν)))
        end
        if isleaf(S.tree, node)
            return
        end
        for c in childnodes(S.tree, node)
            walk(c)
        end
    end
    walk(1)
    return r
end

"""
    evaluate_prior!(chain, gbm::GeometricBrownianMotion)
    evaluate_prior(chain, ν::Float64, gbm::GeometricBrownianMotion)
    evaluate_prior(S, λ, μ, ν, gbm::GeometricBrownianMotion)
    evaluate_prior(S, λ, μ, q, state, gbm::GeometricBrownianMotion)
Evaluate the prior for the chain and store it in the `prior` field.
"""
function evaluate_prior!(chain, gbm::GeometricBrownianMotion)
    logp = evaluate_prior(chain.S, chain.state["λ"], chain.state["μ"],
        chain.state["q"], chain.state, gbm)
    chain.prior = logp
end

function evaluate_prior(chain, ν::Float64, gbm::GeometricBrownianMotion)
    logp = evaluate_prior(chain.S, chain.state["λ"], chain.state["μ"], ν, gbm)
    return logp
end

function evaluate_prior(S, λ, μ, ν, gbm::GeometricBrownianMotion)
    logp  = logpdf(gbm.prior_ν, ν)
    logp += logpdf(gbm.prior_λ, λ[1]) + logpdf(gbm.prior_μ, μ[1])
    logp += mcmctree_gbm_lnprior(S, λ, ν) + mcmctree_gbm_lnprior(S, μ, ν)
    return logp
end

function evaluate_prior(S, λ, μ, q, state, gbm::GeometricBrownianMotion)
    ν = state["ν"][1]
    logp  = logpdf(gbm.prior_λ, λ[1]) + logpdf(gbm.prior_μ, μ[1])
    logp += mcmctree_gbm_lnprior(S, λ, ν) + mcmctree_gbm_lnprior(S, μ, ν)
    logp += sum(logpdf.(gbm.prior_q, q))        # WGD model
    return logp
end

"""
    mcmctree_gbm_lnprior(S::SpeciesTree, r::Array{Float64}, ν::Float64)
Compute the log prior density for the GBM prior on rates. Implementation of the
GBM prior on rates based on Ziheng Yang's MCMCTree. Described in Rannala & Yang
(2007) (syst. biol.). This uses the approach whereby rates are defined for midpoints
of branches, and where a correction is performed to ensure that the correlation is
proper (in contrast with Thorne et al. 1998). See Rannala & Yang 2007 for detailed
information.
"""
function mcmctree_gbm_lnprior(S::SpeciesTree, r::Array{Float64}, ν::Float64)
    logp = -log(2π)/2.0*(2*length(S.leaves)-2)  # every branch has a factor from the Normal.
    for n in keys(S.tree.nodes) # NOTE: nodes are naturally in preorder → OK
        (isleaf(S.tree, n) || haskey(S.wgd_index, n)) ? continue : nothing
        babies = non_wgd_children(S, n)  # should be non-wgd children!
        n == 1 ? ta = 0. : ta = distance(S.tree, parentnode(S.tree, n), n) / 2
        t1 = distance(S.tree, n, babies[1])/2
        t2 = distance(S.tree, n, babies[2])/2
        dett = t1*t2 + ta*(t1+t2)  # determinant of the var-covar matrix Σ up to factor σ^2
        tinv0 = (ta + t2) / dett   # correction terms for correlation given rate at ancestral b
        tinv1 = tinv2 = -ta/dett
        tinv3 = (ta + t1) / dett
        ra = r[n]
        r1 = r[babies[1]]
        r2 = r[babies[2]]
        y1 = log(r1/ra) + (ta + t1)*ν^2/2  # η matrix
        y2 = log(r2/ra) + (ta + t2)*ν^2/2
        zz = (y1*y1*tinv0 + 2*y1*y2*tinv1 + y2*y2*tinv3)
        logp -= zz/(2*ν^2) + log(dett*ν^4)/2 + log(r1*r2);
        #= power 4 is from determinant (which is computed up to the factor from the variance)
        i.e. Σ = [ta+t1, ta ; ta, ta + t2] × ν^2, so the determinant is:
        |Σ| = (ta + t1)ν^2 × (ta + t2)ν^2 - ta ν^2 × ta ν^2 = ν^4[ta × (t1 + t2) + t1 × t2] =#
    end
    return logp
end

"""
    non_wgd_child(S::SpeciesTree, node::Int64)
Get the child nodes that are not WGD nodes of a node.
"""
function non_wgd_children(S::SpeciesTree, node::Int64)
    children = []
    for c in childnodes(S.tree, node)
        haskey(S.wgd_index, c) ? push!(children, non_wgd_child(S, c)) : push!(children, c)
    end
    return children
end

"""
    non_wgd_parent(S::SpeciesTree, node::Int64)
Get the parent node that is not a WGD node of a node.
"""
function non_wgd_parent(S::SpeciesTree, node::Int64)
    if node == 1 ; return 1 ; end
    x = parentnode(S.tree, node)
    while haskey(S.wgd_index, x)
        x = parentnode(S.tree, x)
    end
    return x
end

#=
double lnpriorRates (void)
{
/* This calculates the log of the prior of branch rates under the two rate-drift models:
   the independent rates (clock=2) and the geometric Brownian motion model (clock=3).

   clock=2: the algorithm cycles through the branches, and add up the log
   probabilities.
   clock=3: the root rate is mu or data.rgene[].  The algorithm cycles through
   the ancestral nodes and deals with the two daughter branches.
*/
   int i, inode, locus, dad=-1, g=data.ngene, s=sptree.nspecies, sons[2];
   double lnpR=-log(2*Pi)/2.0*(2*s-2)*g, t,tA,t1,t2,Tinv[4], detT;
   double zz, r=-1, rA,r1,r2, y1,y2;
   double a, b;

   if(com.clock==3 && data.priorrate==1)
      error2("gamma prior for rates for clock3 not implemented yet.");
   else if(com.clock==2 && data.priorrate==1) {   /* clock2, gamma rate prior */
      ...
   }
   else if(com.clock==2 && data.priorrate ==0) {  /* clock2, LN rate prior */
      ...
   }
   else if(com.clock==3 && data.priorrate ==0) {  /* clock3, LN rate prior */
      for(inode=0; inode<sptree.nnode; inode++) {
        if(sptree.nodes[inode].nson==0) continue; /* skip the tips */
        dad = sptree.nodes[inode].father;
        for(i=0; i<2; i++) sons[i] = sptree.nodes[inode].sons[i];
        t = sptree.nodes[inode].age;
        if(inode==sptree.root) tA = 0;
        else                   tA = (sptree.nodes[dad].age - t)/2;
        t1 = (t-sptree.nodes[sons[0]].age)/2;
        t2 = (t-sptree.nodes[sons[1]].age)/2;
        detT = t1*t2+tA*(t1+t2);
        Tinv[0] = (tA+t2)/detT;
        Tinv[1] = Tinv[2] = -tA/detT;
        Tinv[3] = (tA+t1)/detT;
        for(locus=0; locus<g; locus++) {
            rA = (inode==sptree.root ? data.rgene[locus] : sptree.nodes[inode].rates[locus]);
            r1 = sptree.nodes[sons[0]].rates[locus];
            r2 = sptree.nodes[sons[1]].rates[locus];
            y1 = log(r1/rA)+(tA+t1)*data.sigma2[locus]/2;
            y2 = log(r2/rA)+(tA+t2)*data.sigma2[locus]/2;
            zz = (y1*y1*Tinv[0]+2*y1*y2*Tinv[1]+y2*y2*Tinv[3]);
            lnpR -= zz/(2*data.sigma2[locus]) + log(detT*square(data.sigma2[locus]))/2 + log(r1*r2);
        }
    }
 }
 return lnpR;
}=#
