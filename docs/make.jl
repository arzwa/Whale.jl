using Documenter, Whale, Literate

fnames = String[]

ignore = [
	"mapsD2.jl",
	"distances.jl",
	"branchrates.jl",
	"mle-sims.jl",
	"mle.jl",
	"rectree.jl"
]

outdir = joinpath(@__DIR__, "src")
srcdir = joinpath(@__DIR__, "lit")
mkpath(outdir)
for f in readdir(srcdir)
    if endswith(f, ".jl") && !(startswith(f, "_"))
        target = string(split(f,".")[1])
        outpath = joinpath(outdir, target*".md")
		if f ∈ ignore
			try rm(outpath) ; catch ; end
			continue
		end
        push!(fnames, relpath(outpath, joinpath(@__DIR__, "src")))
		@info "Literating $f"
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
