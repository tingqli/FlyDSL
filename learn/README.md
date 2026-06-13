
# composition `R = A o B`
 - `R(c) = A(B(c))`
 - B ⪯ R, 
   B Coarsens R, thus R accepts all coordinates of B:
    - for example, if shape of B is (4, (2,3)), shape of R maybe ((2,2),(2,3)) but not ((2,2),2,3)
   B determines the shape and coordinate sets of the resulting layout by defining the domain of R
 - A determines the codomain of R

Composition Properties:
 - Identity Layouts: ∀c ∈ ZS, IS(c) = c.
   - ID ◦ B = B 
   - A ◦ IS = A
 - Associative Property: given image(C) ⊆ Z(B) and image(B) ⊆ Z(A),
   - A ◦ (B ◦ C) = (A ◦ B) ◦ C

composition can also be done mode by mode or even in a nested structure style.

# complement `A∗`
R = complement(A, shapeM)

 - size(R) and cosize(R) ≤ size(M).
 - R has positive, increasing strides → unique.
 - A and R operate on disjoint mode sets. R completes the missing modes.

R fills the dimensional gaps not covered by A, following the coalescing assumption.

for example, 30=2x3x5, Possible layouts:
 - (2,3,5):(1,2,6)
 - (3,2,5):(1,3,5)
 - (2,5,3):(1,2,10)
 - ......

complement = `any split of modes into two complementary groups.`

# division `A⊘B = A o (B, B∗|A|) = A o B⋆`

Given layouts A and B, their division produces a layout R such that A is split into modes:
 - first mode : elements inside each tile pointed to by B : `A o B`
 - second mode : B-tiles pointed to by B*                  : `A o B*`

After division, size of A is unchanged, cosize is also unchanged(in most cases), 
the only changed is the way we traversal/indexing the elements of A.


```python
"""
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
"""
```

# product `A ⊗ B = (A, A∗ o B)`

The logical product of two layouts A and B is a layout R where “each element of layout B has been replaced with a
uniquely shifted version of the layout A.”
 - first mode is the layout A
 - the second mode is the layout B but with each element replaced by a “unique replication” of layout A.

```python
"""
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_product : ((M,TileM), (N,TileN), L, ...)
zipped_product  : ((M,N), (TileM,TileN,L,...))
tiled_product   : ((M,N), TileM, TileN, L, ...)
flat_product    : (M, N, TileM, TileN, L, ...)
"""
```

# coalesce only works when all shape/strides are static (compile-time constant)
```python
    fx.printf("before coalesce: {}", arg_a.layout) # (512,4):(1,512d)  d means dynamic(runtime)
    arg_a = fx.coalesce(arg_a)
    fx.printf("after coalesce: {}", arg_a.layout)  # (512,4):(1,512d)  no runtime coalesce

    a = fx.make_layout((512,4), (1,512))
    b = fx.coalesce(a)
    fx.printf("coalesce {} -> {}", a, b) # Works:   coalesce (512,4):(1,512) -> 2048:1
```
but some other layout algebra works at runtime

# TensorAdaptor is hack to allow static shape/strides

tensor are passed into kernel with dynamic shape by default
this introduces a lot of runtime layout algebra calculations
some of them are not neccessary.

for now we have no way but hack on it within TensorAdaptor's init.

```bash
export DEBUGPY="-m debugpy --listen localhost:5678 --wait-for-client"

python3 -m debugpy --listen localhost:5678 --wait-for-client xxx.py

    {
        "name": "Python Debugger: Attach",
        "type": "debugpy",
        "justMyCode": false, // important
        "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            }
    }
```