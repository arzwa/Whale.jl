using Soss, NewickTree, Whale

struct SossPrior
    model::Soss.JointDistribution
    ℓ::Function
end

function EqualDupLoss(wm)
    m = @model tree begin
        rmean ~ Exponential()
        r_ ~ For(1:Whale.nnonwgd(tree)) do i
            Exponential(rmean)
        end
        r = permutedims([r_ r_])
        q ~ For(1:Whale.nwgd(tree)) do i
            Beta()
        end
        η ~ Beta(3,1)
    end
    SossPrior(m(tree=wm), (x)->()
end

t = "((A:1.2,B:1.4)86:0.2,C:0.6);"
wm = WhaleModel(t)
model = EqualDupLoss(wm)
θ = rand(model)

# problem is that we have to provide all variables to obtain a logpdf
# we could wrap the models with some kind of closure?
π = logpdf(model, (r=θ.r, rmean=θ.rmean, η=θ.η, q=θ.q))

# define transformation
@show gradient(prior, trans, randn())
