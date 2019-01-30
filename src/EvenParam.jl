module EvenParam

using Interpolations

"""
    reparam(X, dst, closed)

Given a curve in R^d discretized at N points, represented by the N-by-d matrix X,
outputs in the M-by-d matrix dst the result of resampling the curve with a fixed
arclength step with M points, where M = N if new_N is not specified, otherwise M = new_N

The starting point is always preserved. If closed is false, the ending point
is also preserved. If true, then the corresponding curve is assumed to be closed,
and computations are done assuming periodicity of the coordinates. The only preserved
point is the first one, since it conceptually also corresponds to the ending point.
"""
function reparam(X::Array{Float64,2}, dst::Array{Float64,2}; closed::Bool=true, new_N::Int64=0)
    N, d = size(X)

    eff_N = new_N == 0 ? N : new_N

    if closed
        X_e = X[[1:N; 1], :]
    else
        X_e = X
    end

    dX = [zeros(1, d); diff(X_e; dims=1)]

    l = vec(cumsum(sqrt.(sum(abs2, dX; dims=2)); dims=1))
    l_max = l[end]

    if closed
        new_nodes = collect(range(0; stop=l_max, length=eff_N+1)[1:eff_N])
    else
        new_nodes = collect(range(0; stop=l_max, length=eff_N))
    end

    for i in 1:d
        dst[:,i] = interpolate((l,), X_e[:,i], Gridded(Linear()))(new_nodes)
    end
end # reparam

function reparam(X::Array{Float64,2}; closed::Bool=true, new_N::Int64=0)
    N, d = size(X)
    if new_N == 0
        dst = zeros(N, d)
    else
        dst = zeros(new_N, d)
    end
    reparam(X, dst; closed=closed, new_N=new_N)
    return dst
end

function reparam!(X::Array{Float64,2}; closed::Bool=true)
    reparam(X, X; closed=closed)
end

end # module
