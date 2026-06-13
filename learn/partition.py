"""


template <class Layout>
Layout layoutTiledCopyThrValView(LayoutBuilder<Layout> &builder, CopyAtomType copyAtom,
                                 Layout trgLayout, LayoutAttr tiledLayoutThrVal,
                                 LayoutAttr ref2trg) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto *ctx = tiledLayoutThrVal.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  auto atomLayoutRef = cast<LayoutAttr>(copyAtom.getThrValLayoutRef());
  auto atomNumThr = intTupleProduct(attrBuilder, atomLayoutRef.getShape().at(0)).getLeafAsInt();
  auto atomNumVal = intTupleProduct(attrBuilder, atomLayoutRef.getShape().at(1)).getLeafAsInt();

  TileAttr atomTile = TileAttr::get(ArrayAttr::get(ctx, {atomNumThr, atomNumVal}));
  LayoutAttr atomLayoutTV = layoutZippedDivide(attrBuilder, tiledLayoutThrVal, atomTile);

  LayoutAttr firstMode = atomLayoutTV.at(0);
  LayoutAttr composedFirstMode = layoutComposition(attrBuilder, firstMode, ref2trg);
  LayoutAttr trgLayoutTV = LayoutAttr::get(
      IntTupleAttr::get(
          ArrayAttr::get(ctx, {composedFirstMode.getShape(), atomLayoutTV.getShape().at(1)})),
      IntTupleAttr::get(
          ArrayAttr::get(ctx, {composedFirstMode.getStride(), atomLayoutTV.getStride().at(1)})));

  IntTupleAttr zippedShape = intTupleZip(attrBuilder, trgLayoutTV.getShape());
  IntTupleAttr zippedStride = intTupleZip(attrBuilder, trgLayoutTV.getStride());
  auto one = IntTupleAttr::getLeafStatic(ctx, 1);
  IntTupleAttr profile = IntTupleAttr::get(
      ArrayAttr::get(ctx, {one, IntTupleAttr::get(ArrayAttr::get(ctx, {one, one}))}));
  LayoutAttr thrval2mn =
      layoutCoalesce(attrBuilder, LayoutAttr::get(zippedShape, zippedStride), profile);

  IntTuple firstModeShape = builder.at(builder.getShape(trgLayout), 0);
  IntTuple firstModeStride = builder.at(builder.getStride(trgLayout), 0);
  Layout firstModeLayout = builder.makeLayout(firstModeShape, firstModeStride);
  Layout thrval2mnLayout = builder.materializeConstantLayout(thrval2mn);
  Layout composedTV = layoutComposition(builder, firstModeLayout, thrval2mnLayout);

  typename LayoutBuilder<Layout>::ElemCollector retShapeElems;
  typename LayoutBuilder<Layout>::ElemCollector retStrideElems;
  retShapeElems.push_back(builder.getShape(composedTV));
  retStrideElems.push_back(builder.getStride(composedTV));
  retShapeElems.push_back(builder.at(builder.getShape(trgLayout), 1));
  retStrideElems.push_back(builder.at(builder.getStride(trgLayout), 1));
  Layout resultLayout =
      builder.makeLayout(builder.makeTuple(retShapeElems), builder.makeTuple(retStrideElems));

  IntTuple retShape = intTupleExpand(builder, builder.getShape(resultLayout), {0});
  IntTuple retStride = intTupleExpand(builder, builder.getStride(resultLayout), {0});
  return builder.makeLayout(retShape, retStride);
}


template <class Layout>
Layout layoutTiledCopyThrValViewSrc(LayoutBuilder<Layout> &builder, CopyAtomType copyAtom,
                                    LayoutAttr tiledLayoutThrVal, TileAttr tileMN,
                                    Layout srcLayout) {
  auto *ctx = tiledLayoutThrVal.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  auto atomLayoutRef = cast<LayoutAttr>(copyAtom.getThrValLayoutRef());
  auto atomLayoutSrc = cast<LayoutAttr>(copyAtom.getThrValLayoutSrc());
  LayoutAttr refInv = layoutRightInverse(attrBuilder, atomLayoutRef);
  LayoutAttr ref2src = layoutComposition(attrBuilder, refInv, atomLayoutSrc);

  Layout zippedDiv = layoutZippedDivide(builder, srcLayout, tileMN);
  return detail::layoutTiledCopyThrValView(builder, copyAtom, zippedDiv, tiledLayoutThrVal,
                                           ref2src);
}


  LayoutAttr thrValView =
      layoutTiledCopyThrValViewSrc(builder, copyAtom, tiledLayoutThrVal, tileMN, srcLayout);

  SmallVector<Attribute> coordElems;
  coordElems.push_back(thrIdx);
  coordElems.push_back(IntTupleAttr::getLeafNone(context));
  for (int i = 0; i < srcLayout.rank(); ++i)
    coordElems.push_back(IntTupleAttr::getLeafNone(context));
  IntTupleAttr sliceCoord = IntTupleAttr::get(ArrayAttr::get(context, coordElems));

  IntTupleAttr resultShape =
      intTupleSlice(builder, intTupleExpand(builder, thrValView.getShape(), {2}), sliceCoord);
  IntTupleAttr resultStride =
      intTupleSlice(builder, intTupleExpand(builder, thrValView.getStride(), {2}), sliceCoord);
  LayoutAttr partitioned = LayoutAttr::get(resultShape, resultStride);

  IntTupleAttr thrShape = builder.at(thrValView.getShape(), 0);
  IntTupleAttr thrStride = builder.at(thrValView.getStride(), 0);
  IntTupleAttr offset = layoutCrd2Idx(builder, thrIdx, thrShape, thrStride);
  inferredReturnTypes.assign({applyOffsetOnTensorLike(builder, srcTy, offset, partitioned)});


"""

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Generate Typst visualizations for Layout/ComposedLayout, TiledMma, and TiledCopy.
After the Typst source is written, run:

    typst compile layout.typ

to produce the corresponding PDF document.
"""

import os

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
    DebugEnvManager.dump_dir = "dump_ir"
    ir._globals.register_traceback_file_inclusion(__file__)
    ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
    ir._globals.set_loc_tracebacks_frame_limit(40)
    ir._globals.set_loc_tracebacks_enabled(True)
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import inspect

def EXPECT_LAYOUT(layout, shape, stride):
    stack = inspect.stack()
    caller_frame = stack[1]
    if caller_frame.code_context:
        src_line = caller_frame.code_context[0].strip()
        name = src_line.split("(")[1].split(",")[0].strip()
    else:
        name = "<unknown>"
    assert layout.shape.to_py_value() == shape, f"Expected shape {shape}, got {layout.shape.to_py_value()}"
    assert layout.stride.to_py_value() == stride, f"Expected stride {stride}, got {layout.stride.to_py_value()}"
    print(f"   {name}: {layout}")

def div_up(x, y):
    return (x + y - 1) // y



@flyc.jit
def print_layout(layout):
    """
     use range_constexpr if use @cute.jit, otherwise use range
     if layout(i,j) can be computed at compile time, use get_static_leaf_int,
     this is a pure compile-time operation generating no runtime code, 
    """
    #os.environ["DDD"] = "1"
    print(f"{layout}", layout.shape)

    if layout.rank == 1:
        M = layout.shape.get_static_leaf_int
        for i in fx.range_constexpr(M):
            # Layout.__call__ always construct coordinates a Tuple, which is incompatible rank1 tensor
            crd = fx.make_int_tuple(i)
            idx = fly.crd2idx(crd, layout)
            print(f"{idx.get_static_leaf_int:6d}", end=" ")
        print("")

    if layout.rank == 2:
        M = fx.size(layout.shape[0]).get_static_leaf_int
        N = fx.size(layout.shape[1]).get_static_leaf_int
        
        for i in fx.range_constexpr(M):
            print("\t", end="")
            for j in fx.range_constexpr(N):
                idx = layout(i, j).get_static_leaf_int
                idx_m = idx % 32
                idx_n = idx // 32
                print(f"({idx_m:2},{idx_n:2})", end=" ")
            print("")

@flyc.kernel
def kernel_partition(arg_a: fx.Tensor, bid: fx.Int32, tid: fx.Int32):
    """

Value(
    %131 = "fly.tiled_copy.partition_src"(%arg16, %68, %87) :
    
    : (
        %arg16 ： !fly.tiled_copy<!fly.copy_atom<!fly.universal_copy<128>, 16>,
                                  !fly.layout<((8,32),(1,8)):((256,1),(1,32))>,
                                  !fly.tile<[32|64]>>,
         

         %68  ：!fly.memref<bf16, #fly_rocdl.buffer_desc, (256,64,2):(128,1,64)>,


         %87  : !fly.int_tuple<?>) 
         
         (256,64,2):(128,1,64)
         ((32,8),64,2):(128,1,64)


         -> !fly.memref<bf16, #fly_rocdl.buffer_desc, ((8,1),8,1,2):((1,0),4096,0,64)>)

>    
    """
    # logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
    # zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
    # tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
    # flat_divide    : (TileM, TileN, RestM, RestN, L, ...)

    a = fx.make_layout(((2, 4), 4, 2, (2,3)),((6,10),5,6,(2,6)))
    b=fx.coalesce(a)
    print(a, b)

    b = fx.group(a, 1, a.rank)
    c = fx.slice(b, fx.make_coord(None, 1))
    print(b)
    d = b[None, 1]
    
    print(c, c.rank)
    print(d, d.rank)

    M, K = 8129, 128
    src_layout = fx.make_layout((M, K), (K, 1)) # src_layout[M, K]

    tile_m, tile_k = 256, 64
    src_layout_div = fx.flat_divide(src_layout, fx.make_tile(tile_m, tile_k))
    
    # divide [M,K] by [tile_m, tile_k]]
    num_tile_m = div_up(M, tile_m)
    num_tile_k = div_up(K, tile_k)

    EXPECT_LAYOUT(src_layout_div, (tile_m, tile_k, num_tile_m, num_tile_k), (K, 1, tile_m*K, tile_k))
    
    src_block_layout = fx.slice(src_layout_div, fx.make_coord(None, None, bid, None))

    EXPECT_LAYOUT(src_block_layout, (tile_m, tile_k, num_tile_k), (K, 1, tile_k))
    
    mem = fx.Tensor(fx.make_view(fx.get_iter(arg_a), src_block_layout))
    mem = fx.rocdl.make_buffer_tensor(mem)
    # tv layouts has only 2 modes, (thread-mode, value-mode)
    # 

    copy_atom = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(16,64), BFloat16)
    print("=== CopyAtom fx.rocdl.cdna4.LDSReadTrans(16,64) ===")
    print("\t val_bits=", copy_atom.val_bits)           # 16
    print("\t thr_layout=", copy_atom.thr_layout)       # Layout<16:1>
    print("\t layout_ref_tv=", copy_atom.layout_ref_tv) # Layout<(16,(1,4)):(1,(1,16))>
    print("\t layout_src_tv=", copy_atom.layout_src_tv) # Layout<(16,4):(4,1)>
    print("\t layout_dst_tv=", copy_atom.layout_dst_tv) # Layout<(16,(1,4)):(1,(1,16))>

    # CopyAtom
    #   CopyOpUniversalCopyType/CopyOpCDNA3BufferCopyType : Layout<(1,bitsize):(1,1)>
    #   CopyOpCDNA4LdsReadTransposeType : 
    #      bitSize == 64 && transGranularity == 16 (DS_READ_B64_TR_B16) : Layout<(16, (16,4)):(16, (1,256))>
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), BFloat16)
    print("=== CopyAtom UniversalCopy128b ===")
    print("\t val_bits=", copy_atom.val_bits)           # 16
    print("\t thr_layout=", copy_atom.thr_layout)       # Layout<1:1>
    print("\t layout_ref_tv=", copy_atom.layout_ref_tv) # Layout<(1,8):(1,1)>
    print("\t layout_src_tv=", copy_atom.layout_src_tv) # Layout<(1,8):(1,1)>
    print("\t layout_dst_tv=", copy_atom.layout_dst_tv) # Layout<(1,8):(1,1)>

    # TiledCopy
    # 这个layout决定了256个线程如何协作复制一个逻辑shape为(32,64)大小的数据块
    # layout_thr_val(tid, vid) 得到的就是线程tid，变量vid所要复制的 (32, 64) 坐标
    # 但是copy_atom里面还有一层映射
    copy_tile_m = 32
    copy_tile_n = 64
    layout_thr_val = fx.make_layout(
            ((8, 32), (1, 8)),      # 8*32=256 threads, each thread loads 8*1=8 values
            ((256, 1), (1, 32)),    # how (thr_id, value_id) is mapped into (32, 64) tile?
    )
    tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, fx.make_tile(copy_tile_m, copy_tile_n))

    print("=== TiledCopy ===")
    print("\t tile_mn =", tiled_copy.tile_mn)                   # Tile<[32|64]>
    print("\t layout_tv_tiled =", tiled_copy.layout_tv_tiled)        # Layout<((8,32),(1,8)):((256,1),(1,32))>
    print("\t layout_src_tv_tiled=", tiled_copy.layout_src_tv_tiled) # Layout<((8,32),(8,1)):((256,1),(32,0))> tiledCopyGetTiledThrValLayoutSrc  
    print("\t layout_dst_tv_tiled=", tiled_copy.layout_dst_tv_tiled) # Layout<((8,32),(8,1)):((256,1),(32,0))> tiledCopyGetTiledThrValLayoutDst
    
    thr_g2s = tiled_copy.get_slice(tid)
    # partition is two-fold:
    #   1) partition copy tile
    #   2) partition tiled copy thread view
    part_mem = thr_g2s.partition_S(mem)
    part_mem_D = thr_g2s.partition_D(mem)

    frag_copy_A = fx.make_fragment_like(part_mem_D[None, None, None, 0])

    smem_ptr = fx.recast_iter(
        fx.PointerType.get(BFloat16.ir_type, fx.AddressSpace.Shared, 512),
        fx.get_dyn_shared(),
    )
    swz = fx.SwizzleType.get(3, 3, 3)
    mem_lds = fx.make_view(smem_ptr,
                         fx.make_composed_layout(fx.static(swz), 
                                                 fx.make_layout((256,64,2), (128,1,64))))
    part_mem_D = thr_g2s.partition_D(mem_lds)

    copy_atom1 = fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
    # fx.copy(copy_atom1, part_mem[None, None, None, 0], frag_copy_A)
    print(part_mem)
    print(part_mem_D)

    # fx.copy is the final user of all these layouts
    # and its behavior is quite simple, which is just emit copy_atom_calls
    fx.copy(copy_atom, part_mem[None, None, None, 0], part_mem_D[None, None, None, 0])

    # now we reproduce what partition_S does to "mem.layout"

    # trgLayout((tile_m_off, tile_n_off),(tile_m_loc, tile_n_loc)) -> physical offsets in mem
    trgLayout = fx.zipped_divide(mem.layout, tiled_copy.tile_mn) # Layout<((32,64),(8,1,2)):((128,1),(4096,0,64))>
    # EXPECT_LAYOUT(src_block_layout, (tile_m, tile_k, num_tile_k), (K, 1, tile_k))
    EXPECT_LAYOUT(trgLayout, 
                 (
                     (copy_tile_m, copy_tile_n),            # (32,64) first-mode: elements in tile
                     (div_up(tile_m, copy_tile_m),
                      div_up(tile_k, copy_tile_n),
                      num_tile_k)                           # (8,1,2) second-mode: tiles in 3D grid
                 ),
                 (
                     (K,1),
                     (K*copy_tile_m, 0, tile_k)
                 ))

    # 根据common case right_inverse定义:  ∀k∈Z|L‡|, L(L‡(k))=k,
    # refInv是 copy_atom layout_ref_tv的反向映射，带入offset可以求原始1d坐标
    # composition(refInv, layout_src_tv)的结果layout应该是src坐标映射为ref坐标
    #    layout_src_tv(tid, vid) -> src_coord : coordinate into src tile
    #    refInv(src_coord) -> ref_coord : coordinate into copy atom thread-value layout
    refInv = fx.right_inverse(copy_atom.layout_ref_tv)
    ref2trg = fx.composition(refInv, copy_atom.layout_src_tv) # Layout<(1,8):(0,1)>

    tiledLayoutThrVal = tiled_copy.layout_tv_tiled

    def layoutTiledCopyThrValView(copyAtom, trgLayout, tiledLayoutThrVal, ref2trg):
        atomNumThr, atomNumVal = copyAtom.layout_ref_tv.shape.to_py_value()
        atomTile = fx.make_tile(atomNumThr, atomNumVal)
        atomLayoutTV = fx.zipped_divide(tiledLayoutThrVal, atomTile)
        # atomLayoutTV = Layout<((1,8),((8,32),1)):((0,32),((256,1),0))>
        EXPECT_LAYOUT(tiledLayoutThrVal, ((8,32),(1,8)), ((256,1),(1,32)))
        EXPECT_LAYOUT(atomLayoutTV, ((1,8),((8,32),1)), ((0,32),((256,1),0)))
        firstMode = atomLayoutTV[0]
        composedFirstMode = fx.composition(firstMode, ref2trg)
        print(">>>>>>>>>> firstMode=", firstMode)
        print(">>>>>>>>>> composedFirstMode=", composedFirstMode)
        print(">>>>>>>>>> atomLayoutTV=", atomLayoutTV)

        # (1,8) 
        shape0 = composedFirstMode.shape.to_py_value()
        shape1 = atomLayoutTV.shape[1].to_py_value()
        stride0 = composedFirstMode.stride.to_py_value()
        stride1 = atomLayoutTV.stride[1].to_py_value()

        zippedShape = list(zip(shape0, shape1))
        zippedStride = list(zip(stride0, stride1))

        print("zippedShape/zippedStride= ", zippedShape, zippedStride)
        # zippedShape/zippedStride=  [(1, (8, 32)), (8, 1)] [(0, (256, 1)), (32, 0)]
        # trgLayoutTV    = fly<layout((1,8),((8,32),1))   : ((0,32),((256,1),0))>
        thrval2mn = fx.make_layout(zippedShape, zippedStride)
        thrval2mn = fx.coalesce(thrval2mn, fx.make_int_tuple((1, fx.make_int_tuple((1,1)))))

        # === Debug layoutTiledCopyThrValView ===
        #fly<layout((32,64),(8,1,2)):((128,1),(4096,0,64))>
        #fly<layout(32,64):(128,1)>
        #fly<layout((8,32),(8,1)):((256,1),(32,0))>
        #fly<layout((8,32),(8,1)):((8,128),(1,0))>

        print("trgLayout=", trgLayout)
        firstModeLayout = fx.make_layout(trgLayout.shape[0], trgLayout.stride[0])

        print("firstModeLayout=", firstModeLayout)

        # thrval2mn= Layout<((8,32),(8,1)):((256,1),(32,0))>
        # fly<layout((8,32),(8,1)):((256,1),(32,0))>
        composedTV = fx.composition(firstModeLayout, thrval2mn)

        print("composedTV=", composedTV)

        shape = []
        stride = []
        shape.extend(composedTV.shape.to_py_value())
        stride.extend(composedTV.stride.to_py_value())
        shape.extend(trgLayout.shape[1].to_py_value())
        stride.extend(trgLayout.stride[1].to_py_value())

        return fx.make_layout(shape, stride) # Layout<((8,32),(8,1), 8,1,2):((8,128),(1,0), 4096,0,64)>
    
    layout = layoutTiledCopyThrValView(copy_atom, trgLayout, tiled_copy.layout_tv_tiled, ref2trg)
    layout = fx.slice(layout, fx.make_coord(tid, None, None, None, None))
    print(">>>>>>> 1 ", layout)
    print(">>>>>>> 2 ", part_mem.layout)

@flyc.jit
def test_partition(arg_a: fx.Tensor, bid: fx.Int32, tid: fx.Int32):
    CompilationContext.get_current()
    kernel_partition(arg_a, bid, tid).launch(grid=(1,), block=(1,))

A = torch.randn(256, 64, 2, dtype=torch.bfloat16).cuda()
bid = 9
tid = 6
test_partition(A, bid,tid)

