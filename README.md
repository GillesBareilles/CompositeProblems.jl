# CompositeProblems

Layout composite problems and associated oracles. The nonsmooth part is based on `StructuredProximalOperators`.

## Implemented problems

- least squares

## Example: least-squares with l1 penalization.

```julia
julia> using StructuredProximalOperators
julia> n, m, sparsity = 10, 8, 0.8
julia> pb = get_random_qualifiedleastsquares(n, m, regularizer_l1(2.5), sparsity; seed = 1234);
julia> x = rand(n);
julia> h = rand(n);
julia> res = zeros(n);

julia> f(pb, x)

julia> ∇f!(pb, res, x);
julia> ∇f(pb, x);

julia> ∇²f_h!(pb, res, x, h);
julia> ∇²f_h(pb, x, h);

julia> get_gradlips(pb);
julia> get_μ_cvx(pb);
```

### About instances

Some problems use the data from [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/):
- `ionosphere`: automatic download;
- `gisette`: download by hand of file [gisette_scale.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2), which should be decompressed in folder `instances`.
