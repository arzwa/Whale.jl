using Whale
using ConsensusTrees
using PalmTree
using PhyloTrees
using BirthDeathProcesses
using Printf

function logem(wem)
    println(join([@sprintf "%.3f" x.λ for x in wem.θ], ", "))
    println("\t" * join([@sprintf "%.3f" x.μ for x in wem.θ], ", "))
end

S = read_sp_tree("./example/morris-9taxa.nw")
conf = read_whaleconf("./example/whalemle.conf")
slices = get_slices_conf(S, conf["slices"])
rate_index = Whale.constant_ri(S)
ccd = get_ccd("./example/example-ale", S)

# whale EM MAP
wem = WhaleMapEM(S, slices, ccd, get_rateindex(S), (20., 0.01), (20, 0.01))
wem = WhaleMapEM(S, slices, ccd, get_rateindex(S), (2., 0.1), (2, 0.1), N=10)
for i=1:50
    Whale.emiter!(wem, nt=10, nk=5)
    @printf "%3d: " i; logem(wem)
end

# whale EM ML
wem = WhaleMlEM(S, slices, ccd, get_rateindex(S))
for i=1:50
    Whale.emiter!(wem)
    @printf "%3d: " i; logem(wem)
end


# EM steps
wem = WhaleMlEM(S, slices, ccd, get_rateindex(S))
@time Whale.evaluate_lhood!(wem)
@time Whale.backtrack!(wem)
@time ys = Whale.get_transitions(wem.S, wem.T)
@time Whale.whale_emstep!(wem, ys, nt=10, nk=5)
