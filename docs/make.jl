using Documenter, Whale, Literate

outdir = joinpath(@__DIR__, "src")
srcdir = joinpath(@__DIR__, "lit")
mkpath(outdir)

fnames = ["tutorial.jl", "wgd-turing.jl"]
output = String[]

for f in fnames
	target = string(split(f, ".")[1])
	outpath = joinpath(outdir, target*".md")
    push!(output, relpath(outpath, joinpath(@__DIR__, "src")))
	@info "Literating $f"
    Literate.markdown(joinpath(srcdir, f), outdir, documenter=true)
    x = read(`tail -n +4 $outpath`)
    write(outpath, x)
end

@info "makedocs"
makedocs(
    modules = [Whale],
    sitename = "Whale.jl",
    authors = "Arthur Zwaenepoel",
    doctest = false,
	pages = ["Index"=>"index.md",
             "Tutorial"=>"tutorial.md",
             "Examples"=>filter(x->!startswith(x, "tutorial"), output)],)

deploydocs(
    repo = "github.com/arzwa/Whale.jl.git",
    target = "build",)
