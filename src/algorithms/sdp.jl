struct SDP <: AbstractLoss end

function complete(D::MaskedEDM{T}, loss::SDP, dims::Int) where T <: Real
    return SDP(D)
    # return EuclideanDistanceMatrix(D, distances=true)
end

function SDP(D::MaskedEDM{T}) where T <: Real
    ##
    # D = sdr_complete_D(D, W, dims)
    #
    # Completes a partially observed D (with observed entries assumed noiseless)
    # using the semidefinite relaxation (SDR).
    #
    # INPUT:  D ... input partial EDM
    #         W   ... observation mask (1 for observed, 0 for non-observed entries)
    #
    # OUTPUT: D   ... completed EDM
    #
    # Author: Ivan Dokmanic, 2014

    Dm = D.D .* D.mask
    n = size(Dm, 1)

    # The old SVD code to do the following is also good.
    x = -1/(n + sqrt(n))
    y = -1/sqrt(n)
    V = [y * ones(1, n-1); x * ones(n-1, n-1) + Matrix(I, n-1, n-1)]
    e = ones(n, 1)

    G = Semidefinite(n-1)
    B = V * G * V'
    E = diag(B) * e' + e * diag(B)' - 2 * B

    problem = maximize(tr(G))
    problem.constraints += E .* D.mask == D.D .* D.mask
    problem.constraints += isposdef(G)

    solve!(problem, SCSSolver(verbose=false))

    # Can I just use what's above inside the cvx block? Don't do that, do rank
    # thresholding here!
    B = V * G.value * V'
    D = B[diagind(B)] * e' + e * B[diagind(B)]' .- 2*B

    # return EuclideanDistanceMatrix(D, distances=true)
    return D
end