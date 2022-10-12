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

"""
    Given X which same sampled at nodes old_nodes, return the linear interpolation evaluated at new_nodes.
    If periodicise if true, X[1] is appended to X and one node is appended to old_nodes, assuming a constent step
"""
function resample(X::Array{T}, old_nodes::AbstractVector{T}, new_nodes::AbstractVector{T}; periodicise::Bool=false, append::Union{Nothing,Array{T}}=nothing) where T <: Real
    @assert !periodicise || isnothing(append)

    new_N = length(new_nodes)
    nd = ndims(X)

    (X_ext, old_nodes_ext) = if periodicise
        if nd == 1
            (vcat(X, X[1]), vcat(old_nodes, 2old_nodes[end] - old_nodes[end-1]))
        else
            (vcat(X, X[1,:]'), vcat(old_nodes, 2old_nodes[end] - old_nodes[end-1]))
        end
    elseif !isnothing(append)
        if nd == 1
            (vcat(X, append), vcat(old_nodes, 2old_nodes[end] - old_nodes[end-1]))
        else
            (vcat(X, vec(append)'), vcat(old_nodes, 2old_nodes[end] - old_nodes[end-1]))
        end
    else
        (X, old_nodes)
    end

    if nd == 1
        dst = vec(interpolate((old_nodes_ext,), X_ext, Gridded(Linear()))(new_nodes))
    elseif nd == 2
        d = size(X, 2)
        dst = zeros(new_N, d)

        for i in 1:d
            dst[:,i] = interpolate((old_nodes_ext,), X_ext[:,i], Gridded(Linear()))(new_nodes)
        end
    else
        throw(ArgumentError("X must have at most 2 dimensions"))
    end

    return dst
end

"""
    Given X which same sampled at the nodes corresponding to old_range, return the linear interpolation evaluated
    the range with the same bounds with length new_N.
    If periodicise if true, X[1] is appended to X and one node is appended to old_range, assuming a constent step
"""
function resample(X::Array{T}, old_range::AbstractVector{T}, new_N::Int64; periodicise::Bool=false, append::Union{Nothing,Array{T}}=nothing) where T <: Real
    @assert !periodicise || isnothing(append)
    new_range = if periodicise || !isnothing(append)
        s = if isa(old_range, StepRangeLen)
            step(old_range)
        else
            old_range[2] - old_range[1]
        end
        range(old_range[1], old_range[end]+s; length=new_N+1)[1:new_N]
    else
        range(old_range[1], old_range[end]; length=new_N)
    end

    return resample(X, old_range, new_range; periodicise=periodicise, append=append)
end

end # module
