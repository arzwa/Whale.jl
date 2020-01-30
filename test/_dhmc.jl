

wm = WhaleModel(Whale.extree)
Whale.addwgd!(wm, 5, 0.25, rand())
D = distribute(read_ale("./example/example-ale", wm))


prior = CRPrior(MvNormal(ones(2)), Beta(3,1), Beta())
problem = WhaleProblem(wm, D, prior)
logdensity_and_gradient(problem, zeros(4))
