
set_constantrates!(st)
w = WhaleModel(st)

w_, o = mle(w, ccd, show_trace=false)
@test o.minimizer[1] ≈ 0.08839734901061815
@test o.minimizer[2] ≈ 0.15058644779285166
