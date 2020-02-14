using Documenter, Whale, Literate

fnames = String["index.md"]

outdir = joinpath(@__DIR__, "src", "generated")
srcdir = joinpath(@__DIR__, "src", "jl")
mkpath(outdir)
for f in readdir(srcdir)
    if endswith(f, ".jl")
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
    pages = fnames,
)

deploydocs(
    repo = "github.com/arzwa/Whale.jl.git",
    target = "build",
)
