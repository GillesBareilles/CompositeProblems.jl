using Statistics
using BenchmarkTools
using LinearAlgebra


# Looking into pointers from http://fa.bianp.net/blog/2019/evaluate_logistic/
#



σ(x) = 1 / (1 + exp(-x))

## Order 0
function f_naive(x, A, b)
    z = A*x
    tmp = σ.(z)
    @. tmp = - b * log(tmp) - (1 - b) * log(1 - tmp)
    return sum(tmp)/length(tmp)
end


## blog logistic
function logsig_naive(t)
    return log(1 / (1 + exp(-t)))
end

function logsig_log1pexp(t)
    if t < -33.3
        return t
    elseif t <= -18
        return t - exp(t)
    elseif t <= 37
        return -log1p(exp(-t))
    else
        return -exp(-t)
    end
end

function f_logsumexp(x, A, b, logsig)
    m = size(A, 1)

    Ax = A * x
    fval = 0.0
    @inbounds @simd for i in 1:m
        fval += -logsig(b[i] * Ax[i])
    end
    fval /= m

    return fval
end



### Order 1
function ∇f_current(x, A, b)
    m = size(A, 1)

    σbAx = σ.(-b .* (A * x))
    res = transpose(A) * (σbAx .* -b)

    res /= m
    return res
end



function expit_b(x, b)
    out = zeros(size(x))

    @inbounds for (i, xi) in enumerate(x)
        if xi < 0
            out[i] = ((1 - b[i]) * exp(xi) - b[i]) / (1 + exp(xi))
        else
            out[i] = ((1 - b[i]) - b[i] * exp(-xi)) / (1 + exp(-xi))
        end
    end
    return out
end


function ∇f_prop(x, A, b)
    """Computes the gradient of the logistic loss.

    Parameters
    ----------
    x: array-like, shape (n_features,)
        Coefficients

    A: array-like, shape (n_samples, n_features)
        Data matrix

    b: array-like, shape (n_samples,)
        Labels

    Returns
    -------
    grad: array-like, shape (n_features,)
    """
    z = A*x
    s = expit_b(z, b)
    return A'*s / size(A, 1)
end

function main()
    A = [1 1;]
    b = [1]
    x = [20, 20]

    # mode = [:fvals]
    mode = [:∇f]

    if :fvals in mode
        @show f_naive(x, A, b)
        @show f_logsumexp(x, A, b, logsig_naive)
        @show f_logsumexp(x, A, b, logsig_log1pexp)
    elseif :∇f in mode
        @show ∇f_current(x, A, b)
        @show ∇f_prop(x, A, b)
    end

    println("")
    println("")

    println("Larger test")
    n, m = 500, 400
    A = rand(m, n)
    x = rand(n)
    b = rand(m)

    if :fvals in mode
        @show f_naive(x, A, b)
        @show f_logsumexp(x, A, b, logsig_naive)
        @show f_logsumexp(x, A, b, logsig_log1pexp)

        @btime f_naive($x, $A, $b)
        @btime f_logsumexp($x, $A, $b, $logsig_naive)
        @btime f_logsumexp($x, $A, $b, $logsig_log1pexp)
    elseif :∇f in mode
        ∇f_cu = ∇f_current(x, A, b)
        ∇f_pr = ∇f_prop(x, A, b)

        @show norm(∇f_cu - ∇f_pr)

        @btime ∇f_current($x, $A, $b)
        @btime ∇f_prop($x, $A, $b)
    end

    return
end


main()
