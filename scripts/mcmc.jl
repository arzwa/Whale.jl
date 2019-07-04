using Whale

st = Whale.example_tree()
D = read_ale("/home/arzwa/Whale.jl/example/example-ale/", st)

w = WhaleChain(st)
w = WhaleChain(st, IRModel(st, 0.1, 0.9))
chain = mcmc!(w, D, 100, :ν, :η)
