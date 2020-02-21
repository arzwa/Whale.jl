using Documenter, Whale, Literate

fnames = String[]

ignore = [
	"branchrates.jl",
	"distances.jl",
	"mle-sims.jl",
	"mle.jl",
	"rectree.jl"
]

outdir = joinpath(@__DIR__, "src")
srcdir = joinpath(@__DIR__, "lit")
mkpath(outdir)
for f in readdir(srcdir)
    if endswith(f, ".jl") && f ∉ ignore
        @info "Literating $f"
        target = string(split(f,".")[1])
        outpath = joinpath(outdir, target*".md")
        push!(fnames, relpath(outpath, joinpath(@__DIR__, "src")))
        Literate.markdown(joinpath(srcdir, f), outdir, documenter=true)
        x = read(`tail -n +4 $outpath`)
        write(outpath, x)
    end
end

makedocs(
    modules = [Whale],
    format = :html,
    sitename = "Whale.jl",
    authors = "Arthur Zwaenepoel",
    doctest = :fix,
	pages = ["Index"=>"index.md",
			 "Examples"=>fnames],
)

deploydocs(
    repo = "github.com/arzwa/Whale.jl.git",
    target = "build",
)
