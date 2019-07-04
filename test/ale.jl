
λ = repeat([0.2], 17)
μ = [repeat([0.1], 10); repeat([0.2], 7)]
q = repeat([0.2], 7)
w = WhaleModel(st, λ, μ, q, 0.9)
@test logpdf(w, ccd) ≈ -298.2029564  # 02/07/2019
w = WhaleModel(st, λ, μ, q, 0.6)
@test logpdf(w, ccd) ≈ -300.5494354  # 02/07/2019
