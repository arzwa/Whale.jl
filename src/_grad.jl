
# gradient depends on gradient wrt which parameters
gradient(wm::WhaleModel, r::RatesModel, data::CCDArray) =
    mapreduce((x)->gradient(wm, r, x), + , data)

function gradient(wm::WhaleModel, r::RatesModel, ccd::CCD)
    v = asvec(r)
    f = (u) -> logpdf(wm(r(u)), ccd)
    ForwardDiff.gradient(f, v)
end
