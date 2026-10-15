use cubecl_core::{
    self as cubecl,
    frontend::barrier::Barrier,
    ir::dialect::barrier::{
        CommitCopyAsyncOp, CopyAsyncCommitGroupOp, CopyAsyncOp, CopyAsyncWaitGroupOp,
    },
    prelude::*,
};
use pliron::{derive::op_interface_impl, value::Value};

use crate::{
    cuda::{
        cuda_op,
        ptx::{barrier_native_handle, generic_to_shared},
    },
    shared::lowering::LowerOp,
    target::Cuda,
};

// Ptr type doesn't matter

#[cube]
pub fn cp_async_global_to_shared(
    src: *const u32,
    smem: *const u32,
    #[comptime] cache: &str,
    #[comptime] copy_size: usize,
) {
    let smem = generic_to_shared::<u32>(smem);
    gpu_asm!(
        "cp.async.{cache}.shared::cta.global [{}], [{}], {size}, {size};",
        mem_out(_) smem, mem_in(_) src, size = const copy_size, options(explicit_mem)
    );
}

#[cube]
pub fn cp_async_global_to_shared_checked(
    src: *const u32,
    smem: *const u32,
    src_size: usize,
    #[comptime] cache: &str,
    #[comptime] copy_size: usize,
) {
    let smem = generic_to_shared::<u32>(smem);
    gpu_asm!(
        "cp.async.{cache}.shared::cta.global [{}], [{}], {copy_size}, {len};",
        mem_out(_) smem, mem_in(_) src, len = in(_) src_size, options(explicit_mem)
    );
}

#[cube]
pub fn commit_copy_async(bar: &Barrier) {
    let bar_handle = barrier_native_handle(bar);
    gpu_asm!("cp.async.mbarrier.arrive.shared::cta.b64 [{}];", mem_inout(_) bar_handle, options(explicit_mem));
}

#[op_interface_impl]
impl LowerOp<Cuda> for CopyAsyncOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let copy_size = self.copy_length(scope.ctx()).0;
        let src = self.source(scope.ctx()).into();
        let smem = self.destination(scope.ctx()).into();
        let cache = if copy_size == 16 { "cg" } else { "ca" };
        if self.checked(scope.ctx()).0 {
            let src_size = self.source_length(scope.ctx()).into();
            cp_async_global_to_shared_checked::expand(
                scope, &src, &smem, src_size, cache, copy_size,
            );
        } else {
            cp_async_global_to_shared::expand(scope, &src, &smem, cache, copy_size);
        }
        vec![]
    }
}

#[op_interface_impl]
impl LowerOp<Cuda> for CommitCopyAsyncOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let bar = self.barrier(scope.ctx()).into();
        commit_copy_async::expand(scope, &bar);
        vec![]
    }
}

cuda_op!(CopyAsyncCommitGroupOp, |_, _| {
    r#"asm volatile("cp.async.commit_group;" ::: "memory");"#.into()
});
cuda_op!(CopyAsyncWaitGroupOp, |op, ctx| {
    let max_pending = op.max_pending(ctx).0;
    format!(r#"asm volatile("cp.async.wait_group {max_pending};" ::: "memory");"#)
});
