using Documenter, Whale

makedocs(
    modules = [Whale],
    format = Documenter.HTML(),
    sitename = "Whale.jl",
    doctest = :fix,
    pages = [
        "Introduction" => "index.md"
        "Manual" => "manual.md"
        "API" => "api.md"
    ],
)
