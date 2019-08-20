module EuclideanDistanceMatrices

using Random
using Distances
using LinearAlgebra
using Distributions
using TripletEmbeddings

import Base: size, getindex, *, ndims
import LinearAlgebra: rank
import TripletEmbeddings: label

export EuclideanDistanceMatrix, MaskedEuclideanDistanceMatrix, J, 
       
       # triplets.jl
       label, randommask, isedm,

       # utilities.jl
       relativeerror,

       # completion.jl
       complete

include("EDMs.jl")
include("triplets.jl")
include("utilities.jl")
include("completion.jl")

end # module
