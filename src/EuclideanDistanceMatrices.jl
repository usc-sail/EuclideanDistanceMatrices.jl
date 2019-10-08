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

export EuclideanDistanceMatrix, MaskedEuclideanDistanceMatrix, 
       NoisyEuclideanDistanceMatrix, NoisyMaskedEuclideanDistanceMatrix,
       J,
       
       # EDMs.jl
       masked, isedm,

       # utilities.jl
       relativeerror, randommask,

       # completion.jl
       complete,
       AlternatingDescent, SDP # losses

include("EDMs.jl")
include("masks.jl")
include("triplets.jl")
include("utilities.jl")
include("completion.jl")

end # module
