# CompositeProblems

Layout (additive) composite problems and associated oracles. The nonsmooth part is based on `StructuredProximalOperators`.

## Implemented problems

- least squares + l1 regularization
    - `get_lasso_MLE`
- logistic + l1 regularization
    - `get_logit_MLE`
- trace norm + l regularization
    - `get_tracenorm_MLE`



## Example: least-squares with l1 penalization.

```julia
julia> using CompositeProblems
julia> n, m, sparsity = 10, 8, 0.8
julia> pb = get_logit_MLE(n=n, m=m, sparsity=0.8);
julia> x = rand(n);
julia> h = rand(n);
julia> res = zeros(n);

julia> f(pb, x)

julia> ∇f!(pb, res, x);
julia> ∇f(pb, x);

julia> ∇²f_h!(pb, res, x, h);
julia> ∇²f_h(pb, x, h);

julia> get_gradlips(pb);
```

### About instances

Some problems use the data from [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/):
- `ionosphere`: automatic download;
- `gisette`: download by hand of file [gisette_scale.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2), which should be decompressed in folder `instances`.
