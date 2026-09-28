module NoiseEstimators

using Statistics
using SpecialFunctions: gamma_inc, gamma_inc_inv
using FastQuantiles: fast_quantile

# noisefloor.jl
export noisefloor

# noisestats.jl
export noisestats, noisenormalize!, noisenormalize, noisedenormalize

include("noisefloor.jl")
include("noisestats.jl")

end # module NoiseEstimators
