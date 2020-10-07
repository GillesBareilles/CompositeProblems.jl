###############################################################################
## Least squares with nuclear norm regularization
#    min_x  0.5 Σᴹ (⟨Aᵢ, X⟩-yᵢ)² + λ ||X||_*
struct TracenormPb{Tr, Tm} <: CompositeProblem
    A::Vector{Matrix{Float64}}
    y::Vector{Float64}
    n1::Int
    n2::Int                         # matrix variable, dimension (n1,n2)
    m::Int
    regularizer::Tr
    x0::Matrix{Float64}
    M_x0::Tm
end

problem_dimension(pb::TracenormPb) = pb.n1*pn.n2

# 0th order
function f(pb::TracenormPb, X::AbstractMatrix)
    return 0.5 * sum( (dot(pb.A[i], X) - pb.y[i])^2 for i in 1:pb.m)
end

# function f(pb::TracenormPb, x::AbstractVector)
#     X = reshape(pb, x)
#     return 0.5 * sum( (dot(pb.A[i], X) - pb.y[i])^2 for i in 1:pb.m)
# end

# 1st order
∇f(pb::TracenormPb, X::AbstractMatrix) = sum( pb.A[i] * (dot(pb.A[i], X) - pb.y[i]) for i in 1:pb.m)
function ∇f!(pb::TracenormPb, ∇f::AbstractMatrix, X::AbstractMatrix)
    ∇f .= 0.0
    for i in 1:pb.m
        ∇f .+= pb.A[i] * (dot(pb.A[i], X) - pb.y[i])
    end
    return ∇f
end

function ∇f(pb::TracenormPb, x::AbstractVector)
    X = reshape(pb, x)
    return vec(sum( pb.A[i] * (dot(pb.A[i], X) - pb.y[i]) for i in 1:pb.m))
end

# 2nd order
∇²f_h(pb::TracenormPb, X::AbstractMatrix, H::AbstractMatrix) = sum( pb.A[i] * dot(pb.A[i], H) for i in 1:pb.m)
function ∇²f_h!(pb::TracenormPb, ∇²f::AbstractMatrix, X::AbstractMatrix, H::AbstractMatrix)
    ∇²f .= 0
    for i in 1:pb.m
        ∇²f .+= pb.A[i] * dot(pb.A[i], H)
    end
    return ∇²f
end


function ∇²f_h(pb::TracenormPb, x::AbstractVector, h::AbstractVector)
    X = reshape(pb, x)
    H = reshape(pb, h)
    return vec(sum( pb.A[i] * dot(pb.A[i], H) for i in 1:pb.m))
end


# conditioning
# get_gradlips(pb::TracenormPb) = opnorm(pb.A)^2
# get_μ_cvx(pb::TracenormPb) = (svdvals(pb.A)[end])^2


## Instance generation
function get_tracenorm_MLE(;n1=10, n2=12, m=5, sparsity=0.6, seed=1234, δ=0.01)
    # A is drawn from the standard normal distribution
    # b = Ax0 + e where e is taken from the normal distribution with standard deviation 0.001.
    # We set λ1 so that the original sparsity is ultimately recovered.
    @assert 0 ≤ sparsity ≤ 1

    A = Vector{Matrix{Float64}}(undef, m)
    for i in 1:m
        Random.seed!(seed+i)
        A[i] = rand(Normal(), n1, n2)
    end

    ## Generating structured signal
    nsingvals = min(n1, n2)
    rank = Int(ceil((1-sparsity) * nsingvals))
    optstructure = FixedRankMatrices(n1, n2, rank)

    Random.seed!(seed-1)
    x0 = project(optstructure, rand(Normal(), n1, n2))

    ## Noised measurements
    Random.seed!(seed)
    e = rand(Normal(0, δ^2), m)

    y = [dot(A[i], x0) for i in 1:m] .+ e

    return TracenormPb(A, y, n1, n2, m, regularizer_lnuclear(δ), x0, optstructure)
end
