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
import TripletEmbeddings: label

export EuclideanDistanceMatrix, MaskedEuclideanDistanceMatrix, J, 
       
       # triplets.jl
       label, randommask, isedm,

       # utilities.jl
       relativeerror,

       # completion.jl
       complete, AlternatingDescent, SDP

include("EDMs.jl")
include("triplets.jl")
include("utilities.jl")
include("completion.jl")

end # module
