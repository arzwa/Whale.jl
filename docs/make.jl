using Documenter, Whale

makedocs(
    modules = [Whale],
    format = Documenter.HTML(),
    sitename = "Whale.jl",
    pages = [
        "Introduction" => "index.md"
        "Manual" => "manual.md"
        "API" => "api.md"
    ],
)
