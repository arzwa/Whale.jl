# Distributed CCD array
const CCDArray = DArray{CCD,1,Array{CCD,1}}
const CCDSub = SubArray{CCD,0,Array{CCD,1},Tuple{Int64},false}

logpdf(x::CCDArray, m::WhaleModel, node::Int64=-1) =
    sum(ppeval(_logpdf, x, [m], [node]))

_logpdf(x::CCDSub, w, n) = logpdf(x[1], w[1], n[1])

set_recmat!(D::CCDArray) = ppeval(_set_recmat!, D)

function _set_recmat!(x::CCDSub)
    x[1].recmat = x[1].tmpmat
    return 0.
end
