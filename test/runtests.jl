using CompositeProblems
using Test
using ForwardDiff
using LinearAlgebra

@testset "CompositeProblems.jl" begin
    # Write your tests here.

    n, m = 5, 3
    n1, n2 = 5, 4

    testcollection_pb = [
        ("Lasso", get_lasso_MLE(n=n, m=m, sparsity=0.5), :vec),
        ("Logistic", get_logit_MLE(n=n, m=m), :vec),                   # Beware, hessian of logistic is often null for n, m > 10, 10
        ("Tracenorm", get_tracenorm_MLE(n1=n1, n2=n2, m=m), :mat),
    ]

    @testset "$testcollection" for (testcollection, pb, itertype) in testcollection_pb
        x = itertype==:vec ? rand(n) : rand(n1, n2)
        h = itertype==:vec ? rand(n) : rand(n1, n2)

        # First order
        ∇f_x = itertype==:vec ? zeros(n) : zeros(n1,n2)
        ∇f!(pb, ∇f_x, x)

        @test isapprox(∇f_x, ∇f(pb, x))
        @test isapprox(∇f_x, ForwardDiff.gradient(x->f(pb, x), x))

        # Second order
        ∇²f_xh = itertype==:vec ? zeros(n) : zeros(n1,n2)
        ∇²f_h!(pb, ∇²f_xh, x, h)

        @test isapprox(∇²f_xh, ∇²f_h(pb, x, h))

        if itertype == :vec
            ∇²f_xh_AD = ForwardDiff.hessian(x->f(pb, x), x) * h
            @test isapprox(∇²f_xh, ∇²f_xh_AD)
        else
            ∇²f_xh_AD = ForwardDiff.hessian(x->f(pb, x), x) * vec(h)
            ∇²f_xh_AD = reshape(∇²f_xh_AD, (n1, n2))
            @test isapprox(∇²f_xh, ∇²f_xh_AD)
        end
    end
end
