module EuclideanDistanceMatrices

using Random
using Distances
using LinearAlgebra
using Distributions
using TripletEmbeddings

import Base: size, getindex, *, ndims
import LinearAlgebra: rank
import TripletEmbeddings: label

export EuclideanDistanceMatrix, MaskedEuclideanDistanceMatrix, J, label, randommask, isedm

include("EDMs.jl")
# include("maskedEDMs.jl")
include("triplets.jl")
include("utilities.jl")


end # module
