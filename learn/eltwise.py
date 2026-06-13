# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/notebooks/elementwise_add.ipynb

import os

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float8E4M3FNUZ, Float16, Float32, T
from flydsl.expr.typing import Vector as Vec
import torch

import flydsl

if 1:
    from flydsl.utils.env import DebugEnvManager
    from flydsl._mlir import ir
    DebugEnvManager.enable_debug_info = True
    DebugEnvManager.dump_asm = True
    DebugEnvManager.dump_ir = True
    DebugEnvManager.dump_dir = "kkk"
    ir._globals.register_traceback_file_inclusion(__file__)
    ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
    ir._globals.set_loc_tracebacks_frame_limit(40)
    ir._globals.set_loc_tracebacks_enabled(True)
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

N = 512

BLOCK_SIZE = 256 # number of threads in a block
VEC_WIDTH = 4

"""
kernel编写时一般会假定block-size和vector-size是已知的编译时常量
以及想要如何访问逻辑block的数据，例如eltwise的算法会假定逻辑block是1d的，
然后根据block-size和vector-size切割。但是实际外部传入的tensor的layout
则可能是任意strided，编译期已知嵌套结构，或者甚至已知shape/stride。

layout algebra允许我们的kernel代码几乎不变。例如kernel实现中，我们会始终假定：
 1. 1D输入，
 2. 固定按照64线程，每线程处理4个元素（即vector-size=4）
 3. copy_atom是UniversalCopy128b（即每次128bits=16bytes），这样每个线程访问4个float32元素正好16bytes

但是编译期外部传入的tensor的layout实际上可能是各种各样的。例如：
 1. 外部是 1D tensor,完美契合我们的要求
 2. 外部是 (4, 512):(1, 8) 带有stride，但是仍然保证4个元素内部连续
 3. 外部是 (4, 512):(2, 8) 带有stride，无法保证4个元素内部连续，我们的kernel应该检测到这种情况并报错

# logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
# zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
# tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
# flat_divide    : (TileM, TileN, RestM, RestN, L, ...)

"""

@flyc.jit
def test_partition():
    CompilationContext.get_current()
    tid = 2
    bid = 4
    layout0 = fx.make_layout((8,512,32), (1, 8, 8))
    layout1 = fx.zipped_divide(layout0, fx.make_layout((4, 64), (1, 4)))
    print(layout0, "=>", layout1)
    layout2 = fx.slice(layout1, ((None, tid), bid))
    print("   =>", layout2)
    stride0 = layout2.stride.get_static_leaf_int
    assert stride0 == 1, "inner most stride must be 1 for copy_atom with vec width 4, but got {}".format(stride0)

#test_partition() assert 0


@flyc.kernel
def kernel_eltwise1d(mA: fx.Tensor, mB: fx.Tensor):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    
    # what copy_atom can do:
    assert VEC_WIDTH == copy_atom.layout_ref_tv.shape[1].get_static_leaf_int
    print(copy_atom,
          copy_atom.val_bits,         # bits of each element : 32
          copy_atom.thr_layout,       # Layout<1:1>
          copy_atom.layout_src_tv,    # Layout<(1,4):(1,1)> : 1 thread reads 4 values
          copy_atom.layout_dst_tv,    # Layout<(1,4):(1,1)> : 1 thread reads 4 values
          copy_atom.layout_ref_tv,    # Layout<(1,4):(1,1)> : 1 thread reads 4 values
          VEC_WIDTH)

    # zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
    # all other complement modes are nested into the second Mode, which is sliced by bid
    print("before divide: ", mA, mB)
    mA = fx.zipped_divide(mA, fx.make_layout((VEC_WIDTH, BLOCK_SIZE), (1, VEC_WIDTH)))
    mB = fx.zipped_divide(mB, fx.make_layout((VEC_WIDTH, BLOCK_SIZE), (1, VEC_WIDTH)))

    if 0:
        """
        this doesn't work when tensor layout are dynamic (don't known if it's a limitation of FlyDSL)
        because unlike python, MLIR deduce/infer the return type at compile time
        in theory, we should know part of the layout after divide at compile time.
        """
        vA = mA[(None, tid), bid].load()
        vB = vA * vA
        mB[(None, tid), bid] = vB
        return
    
    # if bid == 0 and tid == 0: fx.printf("after divide: {}\n", mA.layout)

    print("        =>: ", mA, mB)
    mA = fx.slice(mA, ((None, tid), bid))
    mB = fx.slice(mB, ((None, tid), bid))
    print("        =>: ", mA, mB)


    if mA.shape.is_static:
        assert mA.shape.to_py_value() == copy_atom.layout_src_tv.shape[1].to_py_value()
        assert mA.stride.to_py_value() == copy_atom.layout_src_tv.stride[1].to_py_value()

    if mB.shape.is_static:
        assert mB.shape.to_py_value() == copy_atom.layout_dst_tv.shape[1].to_py_value()
        assert mB.stride.to_py_value() == copy_atom.layout_dst_tv.stride[1].to_py_value()

    rA = fx.make_rmem_tensor(VEC_WIDTH, fx.Float32)
    rB = fx.make_rmem_tensor(VEC_WIDTH, fx.Float32)

    print("        =>: ", rA, rB)

    fx.copy_atom_call(copy_atom, mA, rA)

    vA = Vec(fx.memref_load_vec(rA))

    vB = vA * vA  # square
    #scale = arith.constant_vector(float(0.123), T.vec(VEC_WIDTH, T.f32))
    #vOut = arith.divf(scale, vA, fastmath=fx.arith.FastMathFlags.fast)

    fx.memref_store_vec(vB, rB)    

    fx.copy_atom_call(copy_atom, rB, mB)

# tiling/divide on inner dimensions is more common
# make stride ascenting helps to make each tile more likely
# to be contiguous
def sort_stride(a: fx.Tensor, ascending=True):
    if isinstance(a, fx.Tensor):
        stride = a.layout.stride.to_py_value()
        indices = sorted(range(len(stride)), key=lambda i: stride[i], reverse=not ascending)
        tensor = fx.select(a, indices)
        return tensor

    # torch tensor
    stride = a.stride()
    indices = sorted(range(len(stride)), key=lambda i: stride[i], reverse=not ascending)
    print(indices)
    return a.permute(indices)

@flyc.jit
def test_eltwise(mA: fx.Tensor, mB: fx.Tensor, stream):
    CompilationContext.get_current()

    # kernel assumes inner most stride is 1 for vectorized load/store
    #mA = sort_stride(mA)
    #mB = sort_stride(mB)

    assert all(t.element_type == mA.element_type for t in [mA, mB])
    dtype = mA.element_type

    coalesced_ldst_bytes = 16
    thr_layout = fx.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = fx.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    #val_layout = fx.recast_layout(val_layout, dtype.width, 8)
    #tiler_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)
    print("thr_layout:", thr_layout)
    print("val_layout:", val_layout)

    # raked_product(A, B) = (A0, B0), (A1, B1), ... 
    # 
    layout_mn = fx.raked_product(thr_layout, val_layout)
    print("layout_mn:", layout_mn)
    thr_size = fx.size(thr_layout).to_py_value()
    val_size = fx.size(val_layout).to_py_value()
    print("thr_size:", thr_size)
    print("val_size:", val_size)
    tmp = fx.make_layout((thr_size, val_size), (1, thr_size))
    print("tmp:", tmp)

    layout_tv = fx.composition(fx.right_inverse(layout_mn), tmp)
    print("tv_layout:", layout_tv)

    tiler_mn = fx.int_tuple_product_each(fx.get_shape(layout_mn)).to_py_value()
    print("tiler_mn:", tiler_mn)

    assert 0

    #fx.printf("mA size:{}", fx.size(mA.layout))
    sz = fx.get_scalar(fx.size(mA.layout))
    tile_size = BLOCK_SIZE * VEC_WIDTH
    grid_x = (sz + (tile_size) - 1) // (tile_size)
    fx.printf("sz={} tile_size={} grid_x={}\n", sz, tile_size, grid_x)

    kernel_eltwise1d(mA, mB).launch(grid=(grid_x,), block=(BLOCK_SIZE,), stream=stream)

torch.set_default_device("cuda")

A = torch.randn(4, N, dtype=torch.float32)
B = torch.zeros(4, N, dtype=torch.float32)
bid = 9
tid = 6

#tA = flyc.from_dlpack(A).mark_static()
#tB = flyc.from_dlpack(B).mark_static()
#tA = flyc.from_dlpack(A.t())
#tB = flyc.from_dlpack(B.t())

tA = A[0:4:2,:]
tB = B[0:4:2,:]

tsA = sort_stride(tA)
tsB = sort_stride(tB)
stream = torch.cuda.current_stream()
hints = {"fast_fp_math": True, "waves_per_eu":8}

compiled_test_eltwise = flyc.compile[hints](test_eltwise, tsA, tsB, stream)

compiled_test_eltwise(tsA, tsB, stream)
assert torch.allclose(tA*tA, tB)
print(A)
print(B)
print(tB.shape, tB.stride())
