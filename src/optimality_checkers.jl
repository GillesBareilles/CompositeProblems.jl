
function compute_minnorm_subgradient(pb::CompositeProblem, x, M, ∇f_x)
    model = Model(with_optimizer(Ipopt.Optimizer, tol=1e-15, print_level=1))

    ḡ_normal = model_g_subgradient!(model, pb.regularizer, M, x)
    ḡ = build_subgradient_from_normalcomp(pb.regularizer, M, x, ḡ_normal)

    if length(ḡ_normal) == 0
        return convert.(Float64, ḡ)
    else
        @objective(model, Min, 0.5*dot(ḡ+∇f_x, ḡ+∇f_x))

        optimize!(model)
        # @show termination_status(model)

        return value.(ḡ)
    end
end

function firstorder_optimality_tangnorm(pb::CompositeProblem, x, M, ∇f_x)

    x_man = x
    if !isa(x, SVDMPoint)
        x_man = project(M, x)
    end

    ## Computing minimal norm subgradient
    ḡ = compute_minnorm_subgradient(pb, x_man, M, ∇f_x)

    fḡ_tangent = project(M, x_man, ∇f_x+ḡ)
    fḡ_normal = ∇f_x+ḡ - fḡ_tangent

    # println("|fḡ_tangent|:  ", norm(fḡ_tangent))
    # println("|fḡ_normal|:   ", norm(fḡ_normal))

    grad_fgₖ = egrad_to_rgrad(M, x_man, ∇f_x) + ∇M_g(pb, M, x_man)
    # println("|grad_fgₖ|:    ", norm(M, x_man, grad_fgₖ))

    if !isapprox(norm(fḡ_tangent), norm(M, x_man, grad_fgₖ))
        @warn "riemannian gradient and tangent subdiff mismatch" norm(fḡ_tangent) norm(M, x_man, grad_fgₖ) norm(project(M, x_man, ḡ)) norm(M, x_man, ∇M_g(pb, M, x_man))
    end

    return norm(fḡ_tangent), norm(fḡ_normal)
end

# function dist_int_subdiff(::regularizer_l1, x_man, M::l1Manifold{n}, ḡ) where {n}

#     d_norm = n - manifold_dimension(M)

#     if d_norm == 0
#         return 0.0
#     end

#     ḡ_normal =
#     @show x_man
#     @show ḡ
#     ḡ_normal = ḡ - project(M, x_man, ḡ)

#     @show ḡ_normal

#     return 0.0
# end

