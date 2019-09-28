struct AlternatingDescent <: AbstractLoss end

function complete(D::MaskedEuclideanDistanceMatrix, loss::AlternatingDescent, d::Int)
    D̂, _ = alternating_descent(D, dims=d)

    return D̂
end

"""
    alternating_descent(D::MaskedEuclideanDistanceMatrix{T}; dims::Int = 2, max_iterations::Int=50) where T <: Real

Complete an EMD using Alternating Descent.

Authors: Reza Parhizkar and Ivan Dokmanic, 2014. Karel Mundnich (port to Julia)
"""
function alternating_descent(D::MaskedEuclideanDistanceMatrix{T}; dims::Int = 2, max_iterations::Int=50) where T <: Real

    n = size(D, 1)
    L = J(n)
    Dm = D.D .* D.mask # We operate over this matrix

    X₀ = zeros(n, dims) # TODO: Transpose
    X = X₀

    connected_nodes = Array{Array{Int,1},1}(undef, n)
    connectivity_vector = Array{Array{Float64,1},1}(undef, n)

    for i in 1:n
        connected_nodes[i] = findall(x -> x != 0, Dm[i, :])
        connectivity_vector[i] = Dm[i, connected_nodes[i]]
    end

    for iter_ind in 1:max_iterations, ind in 1:n
        X_connected = X[connected_nodes[ind], :]

        for coor_ind in 1:dims
            a = 4 * size(X_connected, 1)
            b = 12 * sum((broadcast(-, X[ind, coor_ind], X_connected[:,coor_ind])))
            c = 4 * sum(sum((repeat(X[ind:ind,:], size(X_connected, 1), 1) .- X_connected).^2, dims=2) 
                .+ 2 * (X[ind,coor_ind] .- X_connected[:,coor_ind]).^2 .- connectivity_vector[ind])
            d = 4 * sum((X[ind,coor_ind] .-  X_connected[:,coor_ind]) .* (sum((repeat(X[ind:ind,:], size(X_connected, 1), 1) .- X_connected).^2, dims=2) - connectivity_vector[ind]))
        
            DeltaX_vec = roots(a, b, c, d)
            if DeltaX_vec != nothing
                # If we have no solutions, we should break and return
                DeltaX_vec = real(DeltaX_vec[abs.(imag(DeltaX_vec)) .< eps()])
            else
                break
            end

            cc = zeros(Float64, length(DeltaX_vec))

            for ii = 1 : length(DeltaX_vec)
                cc[ii] = sum(((X[ind, coor_ind] + DeltaX_vec[ii] .- X_connected[:,coor_ind]).^2
                       + sum((repeat(X[ind:ind,:], size(X_connected, 1), 1) .- X_connected).^2, dims=2)
                       - (X[ind, coor_ind] .- X_connected[:, coor_ind]).^2
                       - connectivity_vector[ind]).^2)
            end

            DeltaX = DeltaX_vec[argmin(cc)]
            X[ind, coor_ind] = X[ind, coor_ind] + DeltaX

            X = L * X
        end
    end

    X = X'L
    D = pairwise(SqEuclidean(), X, dims=2)

    return EuclideanDistanceMatrix(D, distances=true), Embedding(X)

end

"""
    roots(a::T, b::T, c::T) where T <: Real

Solve a quadratic equation where a, b, and c are real
    
    a*x^2 + b*x + c = 0

Authors:  Nam Sun Wang (Matlab version), Karel Mundnich (Julia port)
"""
function roots(a::Real, b::Real, c::Real)
    if a == 0
        if b == 0
            # We have a non-equation therefore, we have no valid solution
            return nothing
        else
            # We have a linear equation with 1 root.
            return -c/b
        end
    else
        # We have a true quadratic equation.  Apply the quadratic formula to find two roots.
        x = zeros(Float64, 2)
        DD = b * b - 4 * a * c
        x[1] = (-b + sqrt(DD))/2/a
        x[2] = (-b - sqrt(DD))/2/a
        return x
    end
end

"""
        roots(a::T, b::T, c::T, d::T) where T <: Real

Solve a cubic equation where a, b, c, and d are real.
    
    a*x^3 + b*x^2 + c*x + d = 0

# Reference
Tuma, "Engineering Mathematics Handbook", pp, 7, McGraw Hill, 1978.
    
# Algorithm
    
        - Step 0: If a is 0, use the quadratic formula to avoid dividing by 0.
        - Step 1: Calculate p and q
                    p = ( 3*c/a - (b/a)**2 ) / 3
                    q = ( 2*(b/a)**3 - 9*b*c/a/a + 27*d/a ) / 27
        - Step 2: Calculate discriminant D
                    D = (p/3)**3 + (q/2)**2
        - Step 3: Depending on the sign of D, we follow different strategy.
                    If D<0, thre distinct real roots.
                    If D=0, three real roots of which at least two are equal.
                    If D>0, one real and two complex roots.
        - Step 3a: For D>0 and D=0,
                    Calculate u and v
                    u = cubic_root(-q/2 + sqrt(D))
                    v = cubic_root(-q/2 - sqrt(D))
                    Find the three transformed roots
                    y1 = u + v
                    y2 = -(u+v)/2 + i (u-v)*sqrt(3)/2
                    y3 = -(u+v)/2 - i (u-v)*sqrt(3)/2
        - Step 3b Alternately, for D<0, a trigonometric formulation is more convenient
                    y1 =  2 * sqrt(|p|/3) * cos(ϕ/3)
                    y2 = -2 * sqrt(|p|/3) * cos((ϕ+pi)/3)
                    y3 = -2 * sqrt(|p|/3) * cos((ϕ-pi)/3)
                    where ϕ = acos(-q/2/sqrt(|p|**3/27))
                                pi  = 3.141592654...
        - Step 4  Finally, find the three roots
                    x = y - b/a/3

Authors: Nam Sun Wang (Matlab version), Karel Mundnich (Julia port)
"""
function roots(a::Real, b::Real, c::Real, d::Real)
    # Step 0: If a is 0 use the quadratic formula
    if a == 0.0
        return roots(b, c, d)
    end

    # Cubic equation with 3 roots
    nroot = 3

    # Step 1: Calculate p and q
    p  = c / a - b * b / a / a / 3.0
    q  = (2.0 * b * b * b / a / a / a - 9.0 * b * c / a / a + 27.0 * d / a) / 27.0

    # Step 2: Calculate DD (discriminant)
    DD = p * p * p / 27.0 + q * q / 4.0

    # Step 3: Branch to different algorithms based on DD
    if DD < 0.0
        # Step 3b
        # 3 real unequal roots -- use the trigonometric formulation
        ϕ = acos(- q / 2.0 / sqrt(abs(p * p * p) / 27.0))
        temp1 = 2.0 * sqrt(abs(p)/3.0)
        y1 =  temp1 * cos(ϕ/3.0)
        y2 = -temp1 * cos((ϕ + pi)/3.0)
        y3 = -temp1 * cos((ϕ - pi)/3.0)
    else
        # Step 3a:
        # 1 real root & 2 conjugate complex roots OR 3 real roots (some are equal)
        temp1 = -q/2.0 + sqrt(DD)
        temp2 = -q/2.0 - sqrt(DD)
        u = abs(temp1)^(1/3)
        v = abs(temp2)^(1/3)
        if (temp1 < 0.0) u = -u end
        if (temp2 < 0.0) v = -v end
        y1  = u + v
        y2r = -(u+v)/2.0
        y2i =  (u-v) * sqrt(3.0)/2.0
    end

    # Step 4: Final transformation
    temp1 = b / a / 3.0
    y1 = y1 - temp1
    if DD < 0.0
        y2 = y2 - temp1
        y3 = y3 - temp1
    else
        y2r = y2r - temp1
    end

    # Assign answers
    x = zeros(Number, 3)

    if DD < 0.0
        x[1] = y1
        x[2] = y2
        x[3] = y3
    elseif DD == 0.0
        x[1] =  y1
        x[2] = y2r
        x[3] = y2r
    else
        x[1] = y1
        x[2] = y2r + y2i * 1im
        x[3] = y2r - y2i * 1im
    end

return x
end
