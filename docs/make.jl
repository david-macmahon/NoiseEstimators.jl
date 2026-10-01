using NoiseEstimators
using Documenter

makedocs(;
    modules = [NoiseEstimators],
    authors = "David MacMahon <davidm@astro.berkeley.edu> and contributors",
    sitename = "NoiseEstimators.jl",
    format = Documenter.HTML(;
        canonical = "https://david-macmahon.github.io/NoiseEstimators.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Thresholding statistics" => "thresholding.md",
        "Theory of operation" => "theory.md",
        "Accuracy and tuning" => "accuracy.md",
        "Worked example" => "example.md",
    ],
)

deploydocs(;
    repo = "github.com/david-macmahon/NoiseEstimators.jl",
    devbranch = "main",
)
