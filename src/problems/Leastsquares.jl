###############################################################################
## Least squares problem parametrized by regularizer reg
#    min_x  0.5 * ||Ax-y||² + g(reg, x)
struct LeastsquaresPb{Tr,Tm} <: CompositeProblem
    A::Matrix{Float64}
    y::Vector{Float64}
    regularizer::Tr
    n::Int64
    x0::Vector{Float64}
    M_x0::Tm
end

problem_dimension(pb::LeastsquaresPb) = pb.n

## f
# 0th order
f(pb::LeastsquaresPb, x) = 0.5 * norm(pb.A * x - pb.y)^2

# 1st order
∇f!(pb::LeastsquaresPb, res, x) = (res .= transpose(pb.A) * (pb.A * x - pb.y))
∇f(pb::LeastsquaresPb, x) = transpose(pb.A) * (pb.A * x - pb.y)

# 2nd order
∇²f_h!(pb::LeastsquaresPb, res, x, h) = (res .= transpose(pb.A) * (pb.A * h))
∇²f_h(pb::LeastsquaresPb, x, h) = transpose(pb.A) * (pb.A * h)

# conditioning
get_gradlips(pb::LeastsquaresPb) = opnorm(pb.A)^2
get_μ_cvx(pb::LeastsquaresPb) = (svdvals(pb.A)[end])^2



## instance generation:
function build_original_signal(::regularizer_l1, n, sparsity, seed)
    Random.seed!(seed)
    x0 = rand(Normal(), n)

    Random.seed!(seed)
    inds_nz = Set(randperm(n)[1:sparsity])

    nnz_coords = BitVector(zeros(n))
    for ind in inds_nz
        nnz_coords[ind] = 1
    end

    x0 .*= nnz_coords

    return x0, l1Manifold(nnz_coords)
end

function build_original_signal(reg::regularizer_distball, n, sparsity, seed)
    Random.seed!(seed)
    x0 = rand(Normal(), n)
    x0 *= reg.r / norm(x0, reg.p)

    return x0, PSphere(reg.p, reg.r, n)
end

function build_original_signal(reg::regularizer_group{Tr}, n, sparsity, seed) where {Tr}

    x0 = zeros(n)
    Ms = []
    for (i, group) in enumerate(reg.groups)
        x0i, M0 = build_original_signal(reg.regs[i], length(group), sparsity, seed)
        x0[group] .= x0i
        push!(Ms, M0)
    end

    return x0, ProductManifold(Ms...)
end

function update_regularizationstrength!(regularizer, delta)
    regularizer.λ = delta
    return
end
function update_regularizationstrength!(
    regularizer::regularizer_group{Tr},
    delta,
) where {Tr}
    for i in 1:length(regularizer.regs)
        update_regularizationstrength!(regularizer.regs[i], delta)
    end
    return
end


"""
    get_random_qualifiedleastsquares(n, m, regularizer, sparsity; seed=1234)

Build a least-squares problem that fits the signal recovery framework. The optimization
problem solution should be close to the original signal `x0` and lie exactly on the same
optimal manifold `M_x0`.
"""
function get_random_qualifiedleastsquares(n, m, regularizer, sparsity; seed = 1234)
    # A is drawn from the standard normal distribution
    # b = Ax0 + e where e is taken from the normal distribution with standard deviation 0.001.
    # We set λ1 so that the original sparsity is ultimately recovered.

    Random.seed!(seed)
    A = rand(Normal(), m, n)

    x0, M0 = build_original_signal(regularizer, n, sparsity, seed)

    delta = 0.01
    Random.seed!(seed + 3)
    e = rand(Normal(0, delta^2), m)

    y = A * x0 + e

    update_regularizationstrength!(regularizer, delta)
    return LeastsquaresPb(A, y, regularizer, n, x0, M0)
end

function get_randomlasso(n, m, sparsity; seed = 1234)
    return get_random_qualifiedleastsquares(
        n,
        m,
        regularizer_l1,
        NamedTuple(),
        sparsity;
        seed = seed,
    )
end




# function get_randomlasso(n, m, sparsity; reg=regularizer_l1, seed=1234)
#     # A is drawn from the standard normal distribution
#     # b = Ax0 + e where e is taken from the normal distribution with standard deviation 0.001.
#     # We set λ1 so that the original sparsity is ultimately recovered.


#     elseif reg <: regularizer_l12
#         T = reg.parameters[1]
#         ngroups = Int(ceil(n/T))

#         Random.seed!(seed)
#         inds_nz = Set(randperm(ngroups)[1:sparsity])

#         for i in 1:ngroups-1
#             !(i in inds_nz) && (x0[T*i-3:T*i] .= 0)
#         end
#         !(ngroups in inds_nz) && (x0[T*ngroups-3:end] .= 0)

#     elseif reg == regularizer_lnuclear
#         n_mat = Int(sqrt(n))

#         Random.seed!(seed)
#         basisvecs = rand(Normal(), n_mat, sparsity)

#         x0 = zeros(n)
#         for sp in 1:sparsity
#             x0 += vec(basisvecs[:, sp] * transpose(basisvecs[:, sp]))
#         end

#     else
#         @error "hunhandled regularizer $reg"
#     end

#     delta = 0.01
#     Random.seed!(seed)
#     e = rand(Normal(0, delta^2), m)

#     y = A*x0+e
#     pb = LassoPb{reg}(A, y, delta, n, x0)

#     return pb
# end
