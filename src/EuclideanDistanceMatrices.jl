module EuclideanDistanceMatrices

using SCS
using Convex
using Random
using Distances
using LinearAlgebra
using Distributions
using TripletEmbeddings

import Base: size, getindex, *, ndims, show
import LinearAlgebra: rank

export EDM, MaskedEDM, NoisyEDM, NoisyMaskedEDM,
       J,
       
       # EDMs.jl
       masked, isedm, nitems, 

       # utilities.jl
       relativeerror, randommask, idxmask, 

       # completion.jl
       complete,
       AlternatingDescent, SDP # losses

include("EDMs.jl")
include("masks.jl")
include("triplets.jl")
include("utilities.jl")
include("completion.jl")

end # module
