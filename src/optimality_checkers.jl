function firstorder_optimality_tangnorm(pb::CompositeProblem, x, M, ∇f_x)
    x_man = x
    if !isa(x, SVDMPoint)
        x_man = project(M, x)
    end
    !is_manifold_point(M, x) && @debug "trouble here: !is manifold point" x M
    return firstorder_optimality_tangnorm(pb::CompositeProblem, pb.regularizer, x_man, M, ∇f_x)
end

function compute_minnorm_subgradient(pb::CompositeProblem, regularizer, x, M, ∇f_x)
    model = Model(with_optimizer(
        Mosek.Optimizer,
        QUIET = true,
        INTPNT_CO_TOL_REL_GAP = 1e-15,
        INTPNT_CO_TOL_PFEAS = 1e-13,
        INTPNT_CO_TOL_DFEAS = 1e-13,
    ))
    # Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => false, "INTPNT_CO_TOL_DFEAS" => 1e-7))

    ḡ_normal = model_g_subgradient!(model, regularizer, M, x)
    ḡ = build_subgradient_from_normalcomp(regularizer, M, x, ḡ_normal)

    ḡres = zeros(pb.n1, pb.n2)
    if length(ḡ_normal) == 0
        ḡres .= convert.(Float64, ḡ)
        # return convert.(Float64, ḡ)
    else
        @objective(model, Min, 0.5*dot(ḡ+∇f_x, ḡ+∇f_x))

        optimize!(model)
        # @show termination_status(model)

        ḡres .= value.(ḡ)
        # return value.(ḡ)
    end

    @show manifold_dimension(M)
    display(ḡres)

    ḡexpl = zeros(pb.n1, pb.n2)
    # for i in 1:pb.n
    #     # @show i, i in M.nnz_coords, ∇f_x[i]
    #     if M.nnz_coords[i]
    #         # printstyled("$i\n", color=:red)
    #         ḡexpl[i] = pb.regularizer.λ * sign(x[i])
    #     else
    #         # printstyled("$i\n", color=:blue)
    #         ḡexpl[i] = min(pb.regularizer.λ, max(-pb.regularizer.λ, -∇f_x[i]))
    #     end
    # end

    # @show norm(ḡres - ḡexpl)
    # @show M
    # @show M.nnz_coords
    # display(ḡres)
    # display(ḡexpl)

    if norm(ḡres - ḡexpl) > 1e-7
        @error "--"
        @show norm(ḡres - ḡexpl)
        @show M
        display(ḡres)
        display(ḡexpl)
        display(-∇f_x)
    end

    if manifold_dimension(M) < 9
        @assert false
    end
    return ḡres
end



#
### l1 regularizer
#
function firstorder_optimality_tangnorm(pb::CompositeProblem, regularizer::regularizer_l1, x, M, ∇f_x)
    ḡ = compute_minnorm_subgradient(pb, regularizer, x, M, ∇f_x)

    fḡ_tangent = project(M, x, ∇f_x+ḡ)
    fḡ_normal = ∇f_x+ḡ - fḡ_tangent

    grad_fgₖ = egrad_to_rgrad(M, x, ∇f_x) + ∇M_g(pb, M, x)

    if !isapprox(norm(fḡ_tangent), norm(M, x, grad_fgₖ))
        @debug "riemannian gradient and tangent subdiff mismatch" norm(fḡ_tangent) norm(M, x, grad_fgₖ) norm(project(M, x, ḡ)) norm(M, x, ∇M_g(pb, M, x))
    end

    return norm(fḡ_tangent), dist_int_subdiff(pb, x, M, ḡ)
end

function compute_minnorm_subgradient(pb::CompositeProblem, regularizer::regularizer_l1, x, M, ∇f_x)
    ḡ = zeros(pb.n)
    for i in 1:pb.n
        if M.nnz_coords[i]
            ḡ[i] = pb.regularizer.λ * sign(x[i])
        else
            ḡ[i] = min(pb.regularizer.λ, max(-pb.regularizer.λ, -∇f_x[i]))
        end
    end
    return ḡ
end

function dist_int_subdiff(pb, x_man, M, ḡ)
    if manifold_dimension(M) == problem_dimension(pb)
        return 0
    end

    ḡ_normal = build_normalcomp_from_subgradient(pb.regularizer, M, x_man, ḡ)
    res = maximum(dist_int_subdiff_l1.(ḡ_normal, pb.regularizer.λ)) / pb.regularizer.λ

    return res
end

dist_int_subdiff_l1(x, λ) = max(x-λ, -x-λ)



#
### Nuclear regularizer
#
function firstorder_optimality_tangnorm(pb::CompositeProblem, regularizer::regularizer_lnuclear, x, M::FixedRankMatrices{m,n,k}, ∇f_x) where {m,n,k}
    #! normal component not implemented here.

    ḡ_tan = regularizer.λ .* x.U[:, 1:k] * x.Vt[1:k, :]
    # fḡ_tangent = project(M, x, ∇f_x+ḡ_tan)

    return norm(project(M, x, ∇f_x) + ḡ_tan), -1
end
