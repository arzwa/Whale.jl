using Documenter, Whale #, DocumenterMarkdown, DocumenterLaTeX

makedocs(
    modules = [Whale],
    #format = LaTeX(),
    #format = Markdown(),
    format = :html,
    sitename = "Whale.jl",
    authors = "Arthur Zwaenepoel",
    doctest = :fix,
    pages = [
        "Introduction" => "index.md",
        "Manual" => "manual.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/arzwa/Whale.jl.git",
    target = "build",
)
