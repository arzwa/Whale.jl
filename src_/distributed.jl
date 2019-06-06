# Distributed CCD array
const CCDArray = DArray{CCD,1,Array{CCD,1}}
const CCDSub = SubArray{CCD,0,Array{CCD,1},Tuple{Int64},false}

logpdf(x::CCDArray, m::WhaleModel, node::Int64=-1) =
    sum(ppeval(logpdf_, x, [w], [node]))

logpdf_(x::CCDSub, w, n) = logpdf(x[1], w[1], n[1])
