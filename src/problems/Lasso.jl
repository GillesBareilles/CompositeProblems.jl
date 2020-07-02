###############################################################################
## LASSO Problem parametrized by regularizer reg
#    min_x  0.5 * ||Ax-y||² + λ ||x||_1
struct LassoPb <: CompositeProblem
    A::Matrix{Float64}
    y::Vector{Float64}
    regularizer::regularizer_l1
    n::Int64
end

problem_dimension(pb::LassoPb) = pb.n

## f
# 0th order
f(pb::LassoPb, x) = 0.5 * norm(pb.A*x-pb.y)^2

# 1st order
∇f!(pb::LassoPb, res, x) = (res .= transpose(pb.A)*(pb.A*x - pb.y))
∇f(pb::LassoPb, x) = transpose(pb.A)*(pb.A*x - pb.y)

# 2nd order
∇²f_h!(pb::LassoPb, res, x, h) = (res .= transpose(pb.A) * (pb.A*h))
∇²f_h(pb::LassoPb, x, h) = transpose(pb.A) * (pb.A*h)

# conditioning
get_gradlips(pb::LassoPb) = opnorm(pb.A)^2
get_μ_cvx(pb::LassoPb) = (svdvals(pb.A)[end])^2



## instance generation:
function get_lasso(n, m, sparsity; seed=1234)
    # A is drawn from the standard normal distribution
    # b = Ax0 + e where e is taken from the normal distribution with standard deviation 0.001.
    # We set λ1 so that the original sparsity is ultimately recovered.

    Random.seed!(seed)
    A = rand(Normal(), m, n)

    Random.seed!(seed+1)
    x0 = rand(Normal(), n)

    Random.seed!(seed+2)
    inds_nz = Set(randperm(n)[1:Int(floor(sparsity*n))])

    for i in 1:n
        !(i in inds_nz) && (x0[i] = 0)
    end

    delta = 0.01
    Random.seed!(seed+3)
    e = rand(Normal(0, delta^2), m)

    y = A*x0+e
    pb = LassoPb(A, y, regularizer_l1(delta), n)

    return pb
end
