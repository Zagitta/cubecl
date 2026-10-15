//! NVPTX barriers, and the copies that complete on them.
//!
//! A cube barrier is an `mbarrier`, one 64-bit word in shared memory, which Ampere introduced.
//! Arriving returns the phase it arrived in as a token, and waiting spins until that phase
//! completes: on `mbarrier.try_wait` from Hopper, which suspends the unit for a while in each
//! attempt, and on `mbarrier.test_wait` before it. A phase completes once both its arrivals and
//! the transaction bytes it expects, written by TMA and bulk copies, are all in. Transactions are
//! Hopper's; Ampere's phases count arrivals alone.
//!
//! `copy_async` is Ampere's `cp.async`, a copy of 4, 8 or 16 bytes from global to shared memory
//! that completes in the background. `commit_copy_async` makes the phase the unit arrives in next
//! wait for every copy the unit has issued, without counting as an arrival itself.
//!
//! `memcpy_async` copies synchronously: each unit's part of the copy is done before the unit
//! next arrives, and the release of its arrival with the acquire of the wait carries the copy to
//! the units that wait. A unit barrier only ever waits on its own unit's copies, so it has
//! nothing to wait for and its operations are dropped.

use crate::{
    nvptx::{
        address::NvptxSpace,
        inline_asm::InlineAsm,
        ptx_version::PtxVersion,
        tma::{no_multicast_or_cache_hint, wait_for_groups},
    },
    prelude::*,
    shared::to_llvm::ty::pointee_size,
};
use cubecl_core::ir::{
    dialect::barrier::{
        ArriveAndExpectTxOp, ArriveAndWaitOp, ArriveOp, CommitCopyAsyncOp, CopyAsyncCommitGroupOp,
        CopyAsyncOp, CopyAsyncWaitGroupOp, ExpectTxOp, InitOp, MemCopyAsyncOp, MemCopyAsyncTxOp,
        WaitOp, WaitParityOp,
    },
    nvidia::SmArch,
    types::barrier::{BarrierLevel, BarrierType},
};
use pliron::location::Location;

// LLVM has no scoped form of `init`; arrivals and transactions use the scoped forms, which
// take a count.
const INIT: &str = "llvm.nvvm.mbarrier.init.shared";
const ARRIVE_COUNT: &str = "llvm.nvvm.mbarrier.arrive.scope.cta.space.cta";
// Ampere's arrival, which has no scope or count operand and arrives once.
const ARRIVE_ONCE: &str = "llvm.nvvm.mbarrier.arrive.shared";
const COPY_ASYNC_ARRIVE: &str = "llvm.nvvm.cp.async.mbarrier.arrive.shared";
const COPY_ASYNC_COMMIT_GROUP: &str = "llvm.nvvm.cp.async.commit.group";
const COPY_ASYNC_WAIT_GROUP: &str = "llvm.nvvm.cp.async.wait.group";
const EXPECT_TX: &str = "llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cta";
const BULK_GLOBAL_TO_SHARED: &str = "llvm.nvvm.cp.async.bulk.global.to.shared.cluster";

#[derive(Debug, Error)]
#[error(
    "a TMA or bulk copy completes on an `mbarrier`, which only a cube barrier in shared memory \
     is; a unit barrier has none"
)]
pub struct UnitBarrierUnsupported;

#[derive(Debug, Error)]
#[error("a copy length is an integer, not {0}")]
pub struct CopyLengthType(String);

#[derive(Debug, Error)]
#[error("`cp.async` copies 4, 8 or 16 bytes, not {0}")]
pub struct CopyAsyncLength(usize);

#[derive(Debug, Error)]
#[error(
    "transaction bytes on an `mbarrier` are Hopper's; this device's barrier counts arrivals only"
)]
pub struct TransactionsUnsupported;

#[derive(Debug, Error)]
#[error(
    "waiting on a barrier phase by parity needs PTX 7.1 on Ampere; the driver's PTX version is older"
)]
pub struct ParityWaitUnsupported;

#[derive(Debug, Error)]
#[error(
    "`copy_async` completes on an `mbarrier`, which only a cube barrier in shared memory is; a \
     unit barrier has none"
)]
pub struct CopyAsyncUnitBarrier;

/// The `mbarrier` an architecture has. Ampere introduced it; Hopper added the scoped forms,
/// transaction bytes and `try_wait`, which suspends the unit in hardware between attempts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mbarrier {
    /// `sm_80` to `sm_89`. `parity_wait` is whether the PTX version has `test_wait.parity`,
    /// from 7.1.
    Ampere { parity_wait: bool },
    /// `sm_90` on.
    Hopper,
}

impl Mbarrier {
    /// The `mbarrier` of `arch` under `ptx`, or `None` before Ampere. `ptx` is `None` when
    /// LLVM picks the version, the oldest `arch` accepts: 7.0 on `sm_80`, 7.1 from `sm_86`.
    pub fn of(arch: SmArch, ptx: Option<PtxVersion>) -> Option<Self> {
        match arch.version() {
            ..80 => None,
            80..90 => Some(Mbarrier::Ampere {
                parity_wait: ptx.map_or(arch.version() >= 86, |ptx| ptx.at_least(71)),
            }),
            _ => Some(Mbarrier::Hopper),
        }
    }
}

impl CtxMbarrier for Context {}

pub trait CtxMbarrier: ContextExt {
    fn mbarrier(&self) -> Mbarrier {
        *self
            .aux_ty::<Option<Mbarrier>>()
            .as_ref()
            .expect("barrier ops are only compiled for devices that have an `mbarrier`")
    }
    fn set_mbarrier(&mut self, mbarrier: Option<Mbarrier>) {
        self.set_aux_ty(mbarrier);
    }
}

/// What a barrier operand is on NVPTX, read off the cube type it was converted from.
pub(crate) enum Barrier {
    /// An `mbarrier`, at its address in shared memory.
    Cube(Value),
    /// A unit barrier, which guards only the unit's own synchronous copies.
    Unit,
}

impl Barrier {
    pub(crate) fn new(
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        info: &OperandsInfo,
        barrier: Value,
    ) -> Self {
        let level = cube_pointee(ctx, info, barrier, |inner| {
            inner.downcast_ref::<BarrierType>().map(|barrier| barrier.0)
        })
        .expect("a barrier operand points at a barrier");
        match level {
            BarrierLevel::Cube => Barrier::Cube(NvptxSpace::Shared.cast(ctx, rw, barrier)),
            BarrierLevel::Unit => Barrier::Unit,
        }
    }

    /// The `mbarrier` a copy that counts its bytes as transactions completes on.
    pub(crate) fn mbarrier(self, loc: Location) -> Result<Value> {
        match self {
            Barrier::Cube(barrier) => Ok(barrier),
            Barrier::Unit => input_err!(loc, UnitBarrierUnsupported),
        }
    }
}

/// How a wait names the phase it waits for.
enum Wait {
    /// The state an arrival returned.
    Token(Value),
    /// The parity of the phase.
    Parity(Value),
}

impl Wait {
    /// Spins until the phase completes. The label is scoped to the braces, so a kernel may hold
    /// any number of these, and the memory clobber keeps the reads the barrier guards after it.
    fn emit(
        self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        barrier: Value,
        loc: Location,
    ) -> Result<()> {
        let i32_ty = i32_ty(ctx);
        let address = llvm::PtrToIntOp::new(ctx, barrier, i32_ty);
        let address = insert(ctx, rw, &address);

        // `test_wait` and the plain `.shared` space are what Ampere's PTX has.
        let (test, space) = match ctx.mbarrier() {
            Mbarrier::Hopper => ("try_wait", "shared::cta"),
            Mbarrier::Ampere { parity_wait } => {
                if matches!(self, Wait::Parity(_)) && !parity_wait {
                    return input_err!(loc, ParityWaitUnsupported);
                }
                ("test_wait", "shared")
            }
        };
        let mut asm = InlineAsm::new(Vec::new()).clobbers_memory();
        let address = asm.input(address);
        let (test, phase) = match self {
            Wait::Token(token) => (format!("mbarrier.{test}.{space}.b64"), asm.input(token)),
            Wait::Parity(parity) => {
                let parity = u32_operand(ctx, rw, parity);
                (
                    format!("mbarrier.{test}.parity.{space}.b64"),
                    asm.input(parity),
                )
            }
        };
        let template = format!(
            "{{\n.reg .pred done;\nwait:\n{test} done, [{address}], {phase};\n@!done bra wait;\n}}"
        );
        asm.emit(ctx, rw, &template);
        Ok(())
    }
}

pub(crate) fn init(
    op: &InitOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        let count = u32_operand(ctx, rw, op.arrival_count(ctx));
        call_void(ctx, rw, INIT, vec![barrier, count]);
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn arrive_op(
    op: &ArriveOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    let token = match Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        Barrier::Cube(barrier) => arrive(ctx, rw, barrier),
        Barrier::Unit => unit_token(ctx, rw),
    };
    rw.replace_operation_with_values(ctx, op.get_operation(), vec![token]);
    Ok(())
}

/// Raises the bytes the current phase expects, then arrives `arrive_count_update` times. Two
/// instructions rather than `mbarrier.arrive.expect_tx`, which arrives only once.
pub(crate) fn arrive_and_expect_tx(
    op: &ArriveAndExpectTxOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if ctx.mbarrier() != Mbarrier::Hopper {
        return input_err!(op.loc(ctx), TransactionsUnsupported);
    }
    let token = match Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        Barrier::Cube(barrier) => {
            let count = u32_operand(ctx, rw, op.arrive_count_update(ctx));
            let bytes = u32_operand(ctx, rw, op.transaction_count_update(ctx));
            call_void(ctx, rw, EXPECT_TX, vec![barrier, bytes]);
            let i64_ty = i64_ty(ctx);
            call_intrinsic(ctx, rw, ARRIVE_COUNT, i64_ty, vec![barrier, count])
        }
        Barrier::Unit => unit_token(ctx, rw),
    };
    rw.replace_operation_with_values(ctx, op.get_operation(), vec![token]);
    Ok(())
}

pub(crate) fn expect_tx(
    op: &ExpectTxOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if ctx.mbarrier() != Mbarrier::Hopper {
        return input_err!(op.loc(ctx), TransactionsUnsupported);
    }
    if let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        let bytes = u32_operand(ctx, rw, op.transaction_count_update(ctx));
        call_void(ctx, rw, EXPECT_TX, vec![barrier, bytes]);
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn wait(
    op: &WaitOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        Wait::Token(op.token(ctx)).emit(ctx, rw, barrier, op.loc(ctx))?;
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn wait_parity(
    op: &WaitParityOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        Wait::Parity(op.phase(ctx)).emit(ctx, rw, barrier, op.loc(ctx))?;
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn arrive_and_wait(
    op: &ArriveAndWaitOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        let token = arrive(ctx, rw, barrier);
        Wait::Token(token).emit(ctx, rw, barrier, op.loc(ctx))?;
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

/// Copies `source_length` elements of `source` to `destination`, synchronously. A cooperative
/// copy was already split into each unit's share at the cube level.
pub(crate) fn memcpy_async(
    op: &MemCopyAsyncOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    debug_assert!(
        !op.cooperative(ctx).0,
        "a cooperative copy is split into each unit's share before it is lowered"
    );
    let (source, destination) = (op.source(ctx), op.destination(ctx));
    let bytes = copy_bytes(ctx, rw, info, source, op.source_length(ctx), op.loc(ctx))?;
    let name = format!(
        "llvm.memcpy.{}.{}.i32",
        llvm_mangled_ty(ctx, destination.get_type(ctx)),
        llvm_mangled_ty(ctx, source.get_type(ctx)),
    );
    let volatile = insert_bool_const(ctx, rw, false);
    call_void(ctx, rw, &name, vec![destination, source, bytes, volatile]);

    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

/// A `cp.async` of `copy_length` bytes from global to shared memory. A checked copy reads only
/// `source_length` bytes and fills the rest of the destination with zeros.
pub(crate) fn copy_async(
    op: &CopyAsyncOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _info: &OperandsInfo,
) -> Result<()> {
    let bytes = op.copy_length(ctx).0;
    // `.cg` caches in L2 only, and copies only 16 bytes.
    let cache = match bytes {
        16 => "cg",
        4 | 8 => "ca",
        _ => return input_err!(op.loc(ctx), CopyAsyncLength(bytes)),
    };
    let destination = NvptxSpace::Shared.cast(ctx, rw, op.destination(ctx));
    let source = NvptxSpace::Global.cast(ctx, rw, op.source(ctx));
    let mut name = format!("llvm.nvvm.cp.async.{cache}.shared.global.{bytes}");
    let mut args = vec![destination, source];
    if op.checked(ctx).0 {
        name.push_str(".s");
        args.push(u32_operand(ctx, rw, op.source_length(ctx)));
    }
    call_void(ctx, rw, &name, args);

    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

/// Ties the unit's `cp.async` copies to the barrier's current phase.
pub(crate) fn commit_copy_async(
    op: &CommitCopyAsyncOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) else {
        return input_err!(op.loc(ctx), CopyAsyncUnitBarrier);
    };
    call_void(ctx, rw, COPY_ASYNC_ARRIVE, vec![barrier]);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

/// Closes a group of the unit's `cp.async` copies.
pub(crate) fn copy_async_commit_group(
    op: &CopyAsyncCommitGroupOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _info: &OperandsInfo,
) -> Result<()> {
    call_void(ctx, rw, COPY_ASYNC_COMMIT_GROUP, vec![]);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

/// Waits until at most `max_pending` of the unit's `cp.async` groups still run.
pub(crate) fn copy_async_wait_group(
    op: &CopyAsyncWaitGroupOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _info: &OperandsInfo,
) -> Result<()> {
    let max_pending = op.max_pending(ctx).0;
    wait_for_groups(
        ctx,
        rw,
        op.get_operation(),
        COPY_ASYNC_WAIT_GROUP,
        max_pending,
    )
}

/// A one-dimensional bulk copy of `source_length` elements from global to shared memory, which
/// completes on the barrier as a transaction of the bytes it wrote.
pub(crate) fn memcpy_async_tx(
    op: &MemCopyAsyncTxOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    let loc = op.loc(ctx);
    if ctx.mbarrier() != Mbarrier::Hopper {
        return input_err!(loc, TransactionsUnsupported);
    }
    let barrier = Barrier::new(ctx, rw, info, op.barrier(ctx)).mbarrier(loc.clone())?;
    let bytes = copy_bytes(ctx, rw, info, op.source(ctx), op.source_length(ctx), loc)?;
    let destination = NvptxSpace::SharedCluster.cast(ctx, rw, op.destination(ctx));
    let source = NvptxSpace::Global.cast(ctx, rw, op.source(ctx));

    let mut args = vec![destination, barrier, source, bytes];
    args.extend(no_multicast_or_cache_hint(ctx, rw));
    call_void(ctx, rw, BULK_GLOBAL_TO_SHARED, args);

    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

/// A `u32` operand as the 32-bit integer the instructions take.
fn u32_operand(ctx: &mut Context, rw: &mut DialectConversionRewriter, value: Value) -> Value {
    resize_int(ctx, rw, value, 32, Extension::Zero).expect("barrier counts are integers")
}

fn arrive(ctx: &mut Context, rw: &mut DialectConversionRewriter, barrier: Value) -> Value {
    let i64_ty = i64_ty(ctx);
    match ctx.mbarrier() {
        Mbarrier::Ampere { .. } => call_intrinsic(ctx, rw, ARRIVE_ONCE, i64_ty, vec![barrier]),
        Mbarrier::Hopper => {
            let once = insert_i32_const(ctx, rw, 1);
            call_intrinsic(ctx, rw, ARRIVE_COUNT, i64_ty, vec![barrier, once])
        }
    }
}

/// The token of an arrival on a unit barrier, which no wait reads.
fn unit_token(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
    insert_int_const(ctx, rw, 64, 0)
}

/// The bytes `length` elements of what `source` points at take, as the 32-bit count the copy
/// instructions read.
fn copy_bytes(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
    source: Value,
    length: Value,
    loc: Location,
) -> Result<Value> {
    let elem_size =
        pointee_size(ctx, info, source).expect("a copy source points at sized elements");
    let Some(length) = resize_int(ctx, rw, length, 32, Extension::Zero) else {
        let ty = length.get_type(ctx).disp(ctx).to_string();
        return input_err!(loc, CopyLengthType(ty));
    };
    let elem_size = insert_i32_const(ctx, rw, elem_size as i32);
    let bytes = llvm::MulOp::new_with_overflow_flag(
        ctx,
        length,
        elem_size,
        IntegerOverflowFlagsAttr::default(),
    );
    Ok(insert(ctx, rw, &bytes))
}
