# MLE using gradient and KissThreading

function constraints(w::WhaleModel)
    lower = [0. for i=1:length(asvector1(w))]
    upper = [[Inf for i=1:2*nrates(w.S)] ; [1. for i=1:nwgd(w.S)]]
    return lower, upper
end

function lbfgs(w::WhaleModel, ccd::Array{CCD,1}; show_every=10)
    t = w.S
    x = asvector1(w)

    function f(x::Vector)
        w = WhaleModel(t, x)
        v0 = -logpdf(w, ccd[1])
        return @views tmapreduce(+, ccd[2:end], init=v0) do c
            -logpdf(w, c)
        end
    end

    lower, upper = constraints(w)
    opts = Optim.Options(show_trace=true, show_every=show_every)
    optimize(f, lower, upper, x, Fminbox(LBFGS()), autodiff=:forward, opts)
end
