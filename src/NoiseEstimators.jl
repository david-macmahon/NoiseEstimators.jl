module NoiseEstimators

using Statistics
using SpecialFunctions: gamma_inc, gamma_inc_inv
using FastQuantiles: fast_quantile

# noisefloor.jl
export noisefloor

include("noisefloor.jl")

end # module NoiseEstimators
