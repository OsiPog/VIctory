//! Whole-wave sections (decision 0003): code between `s_or_saveexec -1`
//! (or `s_nand_saveexec 0`) and the restore of the saved mask runs in every lane of the console's
//! wave, the inactive ones included, which is how the compiler expands
//! HLSL wave reductions and prefix sums (an identity put into the inactive
//! lanes with `v_cndmask`, then DPP, `v_permlanex16` and `v_readlane`
//! steps). A host lane that is elsewhere cannot run that code, so each
//! section is simulated on the console's lanes with symbolic values, and a
//! value it leaves is lowered by what it turned out to be: the reduction or
//! prefix of one operation over the lanes' (masked) inputs, which DXIL's
//! wave operations compute over the host's active lanes. No instruction
//! sequence is matched; any step the simulation does not model leaves the
//! values it writes unknown.

use std::collections::HashMap;

use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};

/// An associative operation and its identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    AddU,
    AddF,
    MinU,
    MaxU,
    MinI,
    MaxI,
    MinF,
    MaxF,
    Or,
    And,
    Xor,
    /// An f16 sum in the low half.
    AddF16,
    /// The high dword of a 64-bit sum whose low dwords are the value's
    /// (the low dword itself is an `AddU` of them).
    AddU64Hi(Value),
    /// The low dword of an f64 sum whose high dwords are the value's.
    AddF64Lo(Value),
    /// The high dword of an f64 sum whose low dwords are the value's.
    AddF64Hi(Value),
}

impl Op {
    fn of(name: &str) -> Option<Op> {
        Some(match name {
            "v_add_nc_u32" | "v_add_co_u32" => Op::AddU,
            "v_add_f16" => Op::AddF16,
            "v_add_f32" => Op::AddF,
            "v_min_u32" => Op::MinU,
            "v_max_u32" => Op::MaxU,
            "v_min_i32" => Op::MinI,
            "v_max_i32" => Op::MaxI,
            "v_min_f32" => Op::MinF,
            "v_max_f32" => Op::MaxF,
            "v_or_b32" => Op::Or,
            "v_and_b32" => Op::And,
            "v_xor_b32" => Op::Xor,
            _ => return None,
        })
    }

    /// Whether `k` leaves the other operand unchanged.
    fn identity(self, k: u32) -> bool {
        match self {
            Op::AddU | Op::Or | Op::Xor | Op::MaxU => k == 0,
            Op::AddF | Op::AddF64Hi(_) => k == 0 || k == 0x8000_0000,
            Op::AddF16 => k & 0xFFFF == 0 || k & 0xFFFF == 0x8000,
            Op::AddU64Hi(_) | Op::AddF64Lo(_) => k == 0,
            Op::MinU | Op::And => k == u32::MAX,
            Op::MinI => k == 0x7FFF_FFFF,
            Op::MaxI => k == 0x8000_0000,
            Op::MinF => k == 0x7F80_0000,
            Op::MaxF => k == 0xFF80_0000,
        }
    }

    /// Counting a lane twice changes the result.
    fn counts(self) -> bool {
        matches!(
            self,
            Op::AddU
                | Op::AddF
                | Op::Xor
                | Op::AddF16
                | Op::AddU64Hi(_)
                | Op::AddF64Lo(_)
                | Op::AddF64Hi(_)
        )
    }

    /// Half of a 64-bit operation (no prefix form).
    pub(crate) fn wide(self) -> bool {
        matches!(self, Op::AddU64Hi(_) | Op::AddF64Lo(_) | Op::AddF64Hi(_))
    }
}

/// A lane's value inside a section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sym {
    Unknown,
    K(u32),
    /// The lane's own value of a VGPR value from before the section.
    Own(Value),
    /// A uniform value from before the section.
    Uni(Value),
    /// `op` over `lanes` of `mask ? src : ident` (`op` none: one lane).
    Red {
        op: Option<Op>,
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
    },
}

/// What a value a section leaves holds in every lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    K(u32),
    Value(Value),
    /// The lane's own `mask ? src : ident`.
    Masked(Value, Value, u32),
    /// `op` over every lane.
    Active(Op, Value, Value, u32),
    /// `op` over the lanes below (`inclusive`: and including) each lane.
    Prefix(Op, Value, Value, u32, bool),
}

/// The section starting at instruction `start` of a block: its restore
/// index and the shapes of the values it writes.
pub(crate) struct Section {
    pub end: usize,
    pub shapes: HashMap<Value, Shape>,
}

/// Every whole-wave section of `block`, by start index.
pub(crate) fn sections(p: &Program, block: usize) -> HashMap<usize, Section> {
    let insts = &p.blocks[block].insts;
    let wave: u32 = if p.wave64 { 64 } else { 32 };
    let mut out = HashMap::new();
    let mut i = 0;
    while i < insts.len() {
        let inst = &insts[i];
        // EXEC becomes every lane: `s_or_saveexec -1` or `s_nand_saveexec 0`.
        let all = match inst.name() {
            "s_or_saveexec_b64" | "s_or_saveexec_b32" => Some(Operand::Const(193)),
            "s_nand_saveexec_b64" | "s_nand_saveexec_b32" => Some(Operand::Const(128)),
            _ => None,
        };
        let start = all.is_some_and(
            |k| matches!(&inst.fields, Fields::Sop { src, .. } if src.first() == Some(&k)),
        );
        let saved = match &inst.fields {
            Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                ..
            } if start => inst.write(Reg::S(*code)),
            _ => None,
        };
        let Some(saved) = saved else {
            i += 1;
            continue;
        };
        let end = (i + 1..insts.len()).find(|&j| {
            insts[j].write(Reg::S(EXEC)).is_some()
                && insts[j].read(Reg::S(saved_code(inst))) == Some(saved)
        });
        let Some(end) = end else {
            break;
        };
        let shapes = simulate(&insts[i + 1..end], wave, saved);
        out.insert(i, Section { end, shapes });
        i = end + 1;
    }
    out
}

fn saved_code(inst: &Inst) -> u8 {
    match &inst.fields {
        Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            ..
        } => *code,
        _ => NULL,
    }
}

/// The lane `l` reads with DPP, whether it reads 0 instead, and whether it
/// writes; `None` for patterns not modelled.
fn dpp_lane(d: Dpp, l: u32) -> Option<(u32, bool, bool)> {
    match d {
        Dpp::Lanes { sel, .. } => Some(((l & !7) | (sel >> (3 * (l & 7)) & 7), false, true)),
        Dpp::Row {
            ctrl,
            row_mask,
            bank_mask,
            bound_ctrl,
            ..
        } => {
            let (row, pos, n) = (l & !15, l & 15, u32::from(ctrl & 15));
            let enabled = row_mask >> (l >> 4 & 3) & 1 != 0 && bank_mask >> (l >> 2 & 3) & 1 != 0;
            let (src, inside) = match ctrl {
                0..=0xFF => ((l & !3) | (u32::from(ctrl) >> (2 * (l & 3)) & 3), true),
                0x101..=0x10F => (l + n, pos + n < 16),
                0x111..=0x11F => (l.wrapping_sub(n), pos >= n),
                0x121..=0x12F => (row | (pos.wrapping_sub(n) & 15), true),
                0x140 => (l ^ 15, true),
                0x141 => (l ^ 7, true),
                0x150..=0x15F => (row | n, true),
                0x160..=0x16F => (l ^ n, true),
                _ => return None,
            };
            match (inside, bound_ctrl) {
                (true, _) => Some((src, false, enabled)),
                (false, true) => Some((l, true, enabled)),
                (false, false) => Some((l, false, false)),
            }
        }
    }
}

fn combine(op: Op, a: Sym, b: Sym) -> Sym {
    match (a, b) {
        (Sym::K(k), x) | (x, Sym::K(k)) if op.identity(k) => x,
        (
            Sym::Red {
                op: oa,
                src,
                mask,
                ident,
                lanes: la,
            },
            Sym::Red {
                op: ob,
                src: sb,
                mask: mb,
                ident: ib,
                lanes: lb,
            },
        ) if (src, mask, ident) == (sb, mb, ib)
            && op.identity(ident)
            && oa.is_none_or(|o| o == op)
            && ob.is_none_or(|o| o == op)
            && (!op.counts() || la & lb == 0) =>
        {
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes: la | lb,
            }
        }
        _ => Sym::Unknown,
    }
}

/// A lane's term of a reduction: `None` for the identity `0`.
type Term = Option<(Option<Op>, Value, Value, u32, u64)>;

/// Both halves of an f64 reduction: low and high sources, mask, the high
/// identity and lanes; `None` for the identity.
type Pair = Option<(Value, Value, Value, u32, u64)>;

fn term(s: Sym) -> Option<Term> {
    match s {
        Sym::K(0) => Some(None),
        Sym::Red {
            op,
            src,
            mask,
            ident,
            lanes,
        } => Some(Some((op, src, mask, ident, lanes))),
        _ => None,
    }
}

/// `a + b + carry` for high dwords `a`, `b` of a 64-bit sum whose low
/// dwords' sum (and carry) is `lo`.
fn add_u64_hi(a: Sym, b: Sym, lo: Sym) -> Sym {
    let (Some(ta), Some(tb)) = (term(a), term(b)) else {
        return Sym::Unknown;
    };
    let (src, mask, ident, lanes, ops) = match (ta, tb) {
        (None, None) => return Sym::Unknown,
        // One term: the carry is the other's own low dwords'.
        (Some(t), None) | (None, Some(t)) => {
            return match (t.0, lo) {
                (None, _) => Sym::Red {
                    op: None,
                    src: t.1,
                    mask: t.2,
                    ident: t.3,
                    lanes: t.4,
                },
                (Some(Op::AddU64Hi(x)), Sym::Red { op, src, lanes, .. })
                    if x == src && lanes == t.4 && op.is_none_or(|o| o == Op::AddU) =>
                {
                    Sym::Red {
                        op: t.0,
                        src: t.1,
                        mask: t.2,
                        ident: t.3,
                        lanes: t.4,
                    }
                }
                _ => Sym::Unknown,
            };
        }
        (Some(x), Some(y)) => {
            if (x.1, x.2, x.3) != (y.1, y.2, y.3) || x.4 & y.4 != 0 || x.3 != 0 {
                return Sym::Unknown;
            }
            (x.1, x.2, x.3, x.4 | y.4, [x.0, y.0])
        }
    };
    let Sym::Red {
        op: lo_op,
        src: low,
        mask: lo_mask,
        ident: 0,
        lanes: lo_lanes,
    } = lo
    else {
        return Sym::Unknown;
    };
    if lo_mask != mask
        || lo_lanes != lanes
        || !lo_op.is_none_or(|o| o == Op::AddU)
        || !ops.iter().all(|o| o.is_none_or(|o| o == Op::AddU64Hi(low)))
    {
        return Sym::Unknown;
    }
    Sym::Red {
        op: Some(Op::AddU64Hi(low)),
        src,
        mask,
        ident,
        lanes,
    }
}

/// The f64 sum of register pairs `a` and `b` (low, high dword syms).
fn add_f64(a: (Sym, Sym), b: (Sym, Sym)) -> (Sym, Sym) {
    const NO: (Sym, Sym) = (Sym::Unknown, Sym::Unknown);
    // A pair: the identity (+-0.0), or both halves of one reduction.
    let pair = |(lo, hi): (Sym, Sym)| -> Option<Pair> {
        match (lo, hi) {
            (Sym::K(0), Sym::K(0 | 0x8000_0000)) => Some(None),
            (
                Sym::Red {
                    op: ol,
                    src: xl,
                    mask,
                    ident: 0,
                    lanes,
                },
                Sym::Red {
                    op: oh,
                    src: xh,
                    mask: mh,
                    ident: ih @ (0 | 0x8000_0000),
                    lanes: lh,
                },
            ) if mask == mh
                && lanes == lh
                && ol.is_none_or(|o| o == Op::AddF64Lo(xh))
                && oh.is_none_or(|o| o == Op::AddF64Hi(xl)) =>
            {
                Some(Some((xl, xh, mask, ih, lanes)))
            }
            _ => None,
        }
    };
    let (Some(pa), Some(pb)) = (pair(a), pair(b)) else {
        return NO;
    };
    let (xl, xh, mask, ih, lanes) = match (pa, pb) {
        (None, None) => return (Sym::K(0), Sym::K(0)),
        (Some(_), None) => return a,
        (None, Some(_)) => return b,
        (Some(p), Some(q)) => {
            if (p.0, p.1, p.2, p.3) != (q.0, q.1, q.2, q.3) || p.4 & q.4 != 0 {
                return NO;
            }
            (p.0, p.1, p.2, p.3, p.4 | q.4)
        }
    };
    (
        Sym::Red {
            op: Some(Op::AddF64Lo(xh)),
            src: xl,
            mask,
            ident: 0,
            lanes,
        },
        Sym::Red {
            op: Some(Op::AddF64Hi(xl)),
            src: xh,
            mask,
            ident: ih,
            lanes,
        },
    )
}

fn simulate(insts: &[Inst], wave: u32, saved: Value) -> HashMap<Value, Shape> {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut exec = all;
    let mut vec: HashMap<Value, Vec<Sym>> = HashMap::new();
    let mut uni: HashMap<Value, Sym> = HashMap::new();
    // Masks the section derives from the saved EXEC (`s_and` with it).
    let mut masked = vec![saved];
    // Each `v_add_co_u32`'s lanes, by the carry value it writes.
    let mut carry: HashMap<Value, Vec<Sym>> = HashMap::new();
    for inst in insts {
        let name = inst.name();
        if inst.write(Reg::S(EXEC)).is_some() {
            // Constant lane sets (prefix sums); anything else ends what
            // the simulation can follow.
            let lo = name
                .starts_with("s_mov_b")
                .then(|| constant(inst))
                .flatten();
            let Some(k) = lo else {
                return shapes(vec, uni, wave);
            };
            exec = if name == "s_mov_b64" {
                k
            } else {
                exec & !0xFFFF_FFFF | k & 0xFFFF_FFFF
            };
            continue;
        }
        if inst.write(Reg::S(EXEC + 1)).is_some() {
            let Some(k) = name
                .starts_with("s_mov_b")
                .then(|| constant(inst))
                .flatten()
            else {
                return shapes(vec, uni, wave);
            };
            exec = exec & 0xFFFF_FFFF | k << 32;
            continue;
        }
        if let Fields::Sop { src, .. } = &inst.fields {
            if name.starts_with("s_and_b")
                && src.iter().any(|o| matches!(*o, Operand::Scalar { code, .. } if inst.read(Reg::S(code)).is_some_and(|v| masked.contains(&v))))
                && let Some(&(_, w)) = inst.writes.iter().find(|w| matches!(w.0, Reg::S(_)))
            {
                masked.push(w);
            }
            if name == "s_mov_b32"
                && let (Some(&Operand::Scalar { code, .. }), [(Reg::S(_), w)]) =
                    (src.first(), inst.writes.as_slice())
                && let Some(x) = inst.read(Reg::S(code))
            {
                uni.insert(*w, uni.get(&x).copied().unwrap_or(Sym::Uni(x)));
                continue;
            }
            for &(_, w) in &inst.writes {
                uni.entry(w).or_insert(Sym::Unknown);
            }
            continue;
        }
        let Fields::Valu(v) = &inst.fields else {
            for &(_, w) in &inst.writes {
                vec.insert(w, vec![Sym::Unknown; wave as usize]);
            }
            continue;
        };
        let lanes = |op: &Operand,
                     vec: &HashMap<Value, Vec<Sym>>,
                     uni: &HashMap<Value, Sym>|
         -> Vec<Sym> {
            let one = |s: Sym| vec![s; wave as usize];
            match *op {
                Operand::Vector { reg, .. } => match inst.read(Reg::V(reg)) {
                    Some(x) => vec
                        .get(&x)
                        .cloned()
                        .unwrap_or_else(|| (0..wave).map(|_| Sym::Own(x)).collect()),
                    None => one(Sym::Unknown),
                },
                Operand::Scalar { code, .. } if code != NULL => match inst.read(Reg::S(code)) {
                    Some(x) => one(uni.get(&x).copied().unwrap_or(Sym::Uni(x))),
                    None => one(Sym::Unknown),
                },
                Operand::Const(c) => one(crate::resource::inline(c).map_or(Sym::Unknown, Sym::K)),
                Operand::Literal(x) => one(Sym::K(x)),
                _ => one(Sym::Unknown),
            }
        };
        let modifiers = v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 || v.sdwa.is_some();
        if name == "v_readlane_b32" {
            let (Some(Operand::Scalar { code, .. }), Some(src), Some(lane)) =
                (v.dst, v.src.first(), v.src.get(1))
            else {
                continue;
            };
            let l = match *lane {
                Operand::Const(c) => crate::resource::inline(c),
                Operand::Literal(x) => Some(x),
                _ => None,
            };
            let x = lanes(src, &vec, &uni);
            if let Some(w) = inst.write(Reg::S(code)) {
                let s = l
                    .and_then(|l| x.get(l as usize).copied())
                    .unwrap_or(Sym::Unknown);
                uni.insert(w, s);
            }
            continue;
        }
        let Some(Operand::Vector { reg: dreg, .. }) = v.dst else {
            for &(_, w) in &inst.writes {
                uni.insert(w, Sym::Unknown);
            }
            continue;
        };
        let Some(dst) = inst.write(Reg::V(dreg)) else {
            continue;
        };
        if name == "v_add_f64" {
            // Register pairs: lanes of each half, summed as f64 pairs.
            let half = |o: Option<&Operand>, k: u8| -> Vec<Sym> {
                let o = match o.copied() {
                    Some(Operand::Vector { reg, .. }) => Operand::Vector {
                        reg: reg + k,
                        dwords: 1,
                    },
                    Some(Operand::Scalar { code, .. }) if code != NULL => Operand::Scalar {
                        code: code + k,
                        dwords: 1,
                    },
                    _ => return vec![Sym::Unknown; wave as usize],
                };
                lanes(&o, &vec, &uni)
            };
            let (s0, s1) = (v.src.first(), v.src.get(1));
            let (alo, ahi, blo, bhi) = (half(s0, 0), half(s0, 1), half(s1, 0), half(s1, 1));
            let d = Operand::Vector {
                reg: dreg,
                dwords: 1,
            };
            let (old_lo, old_hi) = (half(Some(&d), 0), half(Some(&d), 1));
            let mut lo = Vec::with_capacity(wave as usize);
            let mut hi = Vec::with_capacity(wave as usize);
            for l in 0..wave as usize {
                let (rl, rh) = if modifiers || v.dpp.is_some() {
                    (Sym::Unknown, Sym::Unknown)
                } else {
                    add_f64((alo[l], ahi[l]), (blo[l], bhi[l]))
                };
                let on = exec >> l & 1 != 0;
                lo.push(if on { rl } else { old_lo[l] });
                hi.push(if on { rh } else { old_hi[l] });
            }
            vec.insert(dst, lo);
            if let Some(h) = inst.write(Reg::V(dreg + 1)) {
                vec.insert(h, hi);
            }
            continue;
        }
        let old = lanes(
            &Operand::Vector {
                reg: dreg,
                dwords: 1,
            },
            &vec,
            &uni,
        );
        // src0 as each lane fetches it (DPP), with the lanes that write.
        let src0 = v.src.first().map(|o| lanes(o, &vec, &uni));
        let mut writes = exec;
        let fetched: Option<Vec<Sym>> = match (v.dpp, &src0) {
            (Some(d), Some(s)) => {
                if exec != all {
                    vec.insert(dst, vec![Sym::Unknown; wave as usize]);
                    continue;
                }
                let mut f = Vec::with_capacity(wave as usize);
                let mut ok = true;
                for l in 0..wave {
                    match dpp_lane(d, l) {
                        Some((src, zero, on)) if src < wave => {
                            f.push(if zero { Sym::K(0) } else { s[src as usize] });
                            if !on {
                                writes &= !(1u64 << l);
                            }
                        }
                        _ => ok = false,
                    }
                }
                ok.then_some(f)
            }
            (None, Some(s)) => Some(s.clone()),
            _ => None,
        };
        let src = |k: usize| v.src.get(k).map(|o| lanes(o, &vec, &uni));
        let result: Vec<Sym> = match (name, fetched) {
            _ if modifiers => vec![Sym::Unknown; wave as usize],
            ("v_mov_b32", Some(a)) => a,
            ("v_bfrev_b32", Some(a)) => a
                .into_iter()
                .map(|s| match s {
                    Sym::K(k) => Sym::K(k.reverse_bits()),
                    _ => Sym::Unknown,
                })
                .collect(),
            ("v_cndmask_b32", Some(a)) => {
                let mask = match v.src.get(2) {
                    Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                    _ => None,
                };
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied(), mask) {
                        (Sym::K(ident), Some(Sym::Own(x)), Some(m)) if masked.contains(&m) => {
                            Sym::Red {
                                op: None,
                                src: x,
                                mask: m,
                                ident,
                                lanes: 1 << l,
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_permlanex16_b32", Some(a)) => {
                // Lanes of each row read the other row of their half; lane
                // i's pick is nibble i of {src2, src1}.
                let sel = |k: usize| match v.src.get(k) {
                    Some(&Operand::Const(c)) => crate::resource::inline(c),
                    Some(&Operand::Literal(x)) => Some(x),
                    _ => None,
                };
                match (sel(1), sel(2)) {
                    (Some(s1), Some(s2)) => {
                        let s = u64::from(s1) | u64::from(s2) << 32;
                        (0..wave)
                            .map(|l| {
                                let other = (l & !31) | ((l & 16) ^ 16);
                                let pick = (s >> (4 * (l & 15)) & 15) as u32;
                                a[(other | pick) as usize]
                            })
                            .collect()
                    }
                    _ => vec![Sym::Unknown; wave as usize],
                }
            }
            ("v_add_co_ci_u32", Some(a)) => {
                // The high dword of a 64-bit sum: its carry comes from the
                // `v_add_co_u32` of the low dwords.
                let b = src(1).unwrap_or_default();
                let lo = inst
                    .reads
                    .iter()
                    .find_map(|&(r, x)| match r {
                        Reg::S(_) => carry.get(&x),
                        _ => None,
                    })
                    .cloned();
                (0..wave as usize)
                    .map(|l| match (&lo, b.get(l)) {
                        (Some(c), Some(&bl)) => add_u64_hi(a[l], bl, c[l]),
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_sub_nc_u32", Some(a)) => {
                // A prefix sum less the lane's own term: the exclusive one.
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied()) {
                        (
                            Sym::Red {
                                op: None | Some(Op::AddU),
                                src,
                                mask,
                                ident: 0,
                                lanes,
                            },
                            Some(Sym::Red {
                                op: None | Some(Op::AddU),
                                src: s2,
                                mask: m2,
                                ident: 0,
                                lanes: l2,
                            }),
                        ) if (src, mask) == (s2, m2) && lanes & l2 == l2 => {
                            let rest = lanes & !l2;
                            if rest == 0 {
                                Sym::K(0)
                            } else {
                                Sym::Red {
                                    op: Some(Op::AddU),
                                    src,
                                    mask,
                                    ident: 0,
                                    lanes: rest,
                                }
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            (n, Some(a)) => match (Op::of(n), src(1)) {
                (Some(op), Some(b)) => (0..wave as usize)
                    .map(|l| combine(op, a[l], b[l]))
                    .collect(),
                _ => vec![Sym::Unknown; wave as usize],
            },
            _ => vec![Sym::Unknown; wave as usize],
        };
        let merged: Vec<Sym> = (0..wave as usize)
            .map(|l| {
                if writes >> l & 1 != 0 {
                    result[l]
                } else {
                    old[l]
                }
            })
            .collect();
        if name == "v_add_co_u32"
            && let Some(&(_, c)) = inst.writes.iter().find(|w| matches!(w.0, Reg::S(_)))
        {
            carry.insert(c, merged.clone());
        }
        vec.insert(dst, merged);
    }
    shapes(vec, uni, wave)
}

fn constant(inst: &Inst) -> Option<u64> {
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    match *src.first()? {
        Operand::Const(c) => {
            let x = crate::resource::inline(c)?;
            Some(if (x as i32) < 0 {
                u64::MAX
            } else {
                u64::from(x)
            })
        }
        Operand::Literal(x) => Some(u64::from(x)),
        _ => None,
    }
}

/// The shape of each value, where every lane agrees on one.
fn shapes(
    vec: HashMap<Value, Vec<Sym>>,
    uni: HashMap<Value, Sym>,
    wave: u32,
) -> HashMap<Value, Shape> {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut out = HashMap::new();
    let single = |s: Sym| match s {
        Sym::K(k) => Some(Shape::K(k)),
        Sym::Own(x) | Sym::Uni(x) => Some(Shape::Value(x)),
        Sym::Red {
            op: Some(op),
            src,
            mask,
            ident,
            lanes,
        } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
        _ => None,
    };
    for (v, s) in uni {
        if let Some(sh) = single(s) {
            out.insert(v, sh);
        }
    }
    for (v, lanes) in vec {
        if let Some(sh) = per_lane(&lanes, all) {
            out.insert(v, sh);
        }
    }
    out
}

fn per_lane(lanes: &[Sym], all: u64) -> Option<Shape> {
    let first = lanes[0];
    let own = |l: usize, s: Sym| match s {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if lanes == 1 << l => Some((src, mask, ident)),
        _ => None,
    };
    if let Some(m) = own(0, first)
        && lanes.iter().enumerate().all(|(l, &s)| own(l, s) == Some(m))
    {
        return Some(Shape::Masked(m.0, m.1, m.2));
    }
    if lanes.iter().all(|&s| s == first) {
        return match first {
            Sym::K(k) => Some(Shape::K(k)),
            Sym::Uni(x) | Sym::Own(x) => Some(Shape::Value(x)),
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes,
            } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
            _ => None,
        };
    }
    // Prefixes: lane l holds lanes 0..l (exclusive) or 0..=l (inclusive).
    let mut found: Option<(Op, Value, Value, u32, bool)> = None;
    for (l, &s) in lanes.iter().enumerate() {
        let below = (1u64 << l) - 1;
        let (op, src, mask, ident, set) = match s {
            Sym::Red {
                op,
                src,
                mask,
                ident,
                lanes,
            } => (op, src, mask, ident, lanes),
            // Lane 0 of an exclusive prefix holds the identity.
            Sym::K(_) if l == 0 => continue,
            _ => return None,
        };
        let inclusive = set == below | 1 << l;
        if !inclusive && set != below {
            return None;
        }
        let op = match (op, found) {
            (Some(o), _) => o,
            (None, Some(f)) => f.0,
            (None, None) => continue,
        };
        let me = (op, src, mask, ident, inclusive);
        match found {
            None => found = Some(me),
            Some(f) if f == me => {}
            _ => return None,
        }
    }
    let (op, src, mask, ident, inclusive) = found?;
    if let Sym::K(k) = lanes[0]
        && (inclusive || k != ident)
    {
        return None;
    }
    Some(Shape::Prefix(op, src, mask, ident, inclusive))
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn build(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words
            .iter()
            .copied()
            .chain(std::iter::once(S_ENDPGM))
            .flat_map(u32::to_le_bytes)
            .collect();
        crate::ir::build(&code, wave64, 256).unwrap()
    }

    #[test]
    fn dpp_lanes_follow_the_isa() {
        // row_shr:1 with bound_ctrl: lane 16 reads 0, lane 17 lane 16.
        let shr = Dpp::Row {
            ctrl: 0x111,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(shr, 16), Some((16, true, true)));
        assert_eq!(dpp_lane(shr, 17), Some((16, false, true)));
        // row_mirror: lane 3 reads lane 12.
        let mirror = Dpp::Row {
            ctrl: 0x140,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(mirror, 3), Some((12, false, true)));
        // dpp8 [1,0,3,2,5,4,7,6] swaps neighbours.
        let swap = Dpp::Lanes {
            sel: 1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21,
            fi: false,
        };
        assert_eq!(dpp_lane(swap, 8), Some((9, false, true)));
    }

    #[test]
    fn prefixes_are_recognised() {
        let (x, m) = (Value(1), Value(2));
        let red = |lanes: u64| Sym::Red {
            op: Some(Op::AddU),
            src: x,
            mask: m,
            ident: 0,
            lanes,
        };
        let inclusive: Vec<Sym> = (0..64).map(|l| red(u64::MAX >> (63 - l))).collect();
        assert_eq!(
            per_lane(&inclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, true))
        );
        let mut exclusive: Vec<Sym> = (0..64).map(|l| red((1u64 << l) - 1)).collect();
        exclusive[0] = Sym::K(0);
        assert_eq!(
            per_lane(&exclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, false))
        );
        assert_eq!(
            per_lane(&vec![red(u64::MAX); 64], u64::MAX),
            Some(Shape::Active(Op::AddU, x, m, 0))
        );
    }

    /// The shapes of the values `regs` hold after the section at the
    /// block's first instruction, with the program's input values.
    fn section_shapes(words: &[u32], wave64: bool, regs: &[Reg]) -> (Program, Vec<Option<Shape>>) {
        let p = build(words, wave64);
        let found = sections(&p, 0);
        let sec = &found[&0];
        let out = regs
            .iter()
            .map(|&r| {
                let w = p.blocks[0].insts[..sec.end]
                    .iter()
                    .rev()
                    .find_map(|i| i.write(r))?;
                sec.shapes.get(&w).copied()
            })
            .collect();
        (p, out)
    }

    fn input(p: &Program, r: Reg) -> Value {
        p.inputs.iter().find(|i| i.0 == r).unwrap().1
    }

    #[test]
    fn sixty_four_bit_integer_sum() {
        // Wave32 WaveActiveSum(uint64_t v[0:1]) as the console compiler
        // expands it: v_add_co_u32 / v_add_co_ci_u32 pairs.
        let words = [
            0xBE8B_3DC1,
            0x7E04_0280,
            0x7E06_0280,
            0xD501_0002,
            0x002E_0102,
            0xD501_0003,
            0x002E_0303,
            0x7E08_02E9,
            0xDE54_C102,
            0x7E0A_02E9,
            0xDE54_C103,
            0xD70F_6A02,
            0x0002_0504,
            0x5006_0705,
            0x7E08_02E9,
            0x9360_1202,
            0x7E0A_02E9,
            0x9360_1203,
            0xD70F_6A02,
            0x0002_0504,
            0x5006_0705,
            0x7E08_02E9,
            0x0009_2402,
            0x7E0A_02E9,
            0x0009_2403,
            0xD70F_6A02,
            0x0002_0504,
            0x5006_0705,
            0x7E08_02FA,
            0xFF09_4002,
            0x7E0A_02FA,
            0xFF09_4003,
            0xD70F_6A02,
            0x0002_0504,
            0x5006_0705,
            0xD778_0004,
            0x0201_0102,
            0xD778_0005,
            0x0201_0103,
            0xD70F_6A02,
            0x0002_0504,
            0x5006_0705,
            0xD760_0002,
            0x0001_0102,
            0xD760_0003,
            0x0001_0103,
            0xBEFE_030B,
        ];
        let (p, s) = section_shapes(&words, false, &[Reg::S(2), Reg::S(3)]);
        let (lo, hi) = (input(&p, Reg::V(0)), input(&p, Reg::V(1)));
        let m = p.blocks[0].insts[0].write(Reg::S(11)).unwrap();
        assert_eq!(s[0], Some(Shape::Active(Op::AddU, lo, m, 0)));
        assert_eq!(s[1], Some(Shape::Active(Op::AddU64Hi(lo), hi, m, 0)));
    }

    #[test]
    fn f64_sum_across_halves() {
        // Wave64 WaveActiveSum(double v[29:30]) with v_add_f64 steps, the
        // halves joined through v_readlane 32 and copied by s_mov_b32.
        let words = [
            0xBE90_25C1,
            0x7E04_0280,
            0x7E06_0280,
            0x8792_104E,
            0xD501_0002,
            0x004A_3B02,
            0xD501_0003,
            0x004A_3D03,
            0x7E08_02E9,
            0xDE54_C102,
            0x7E0A_02E9,
            0xDE54_C103,
            0xD564_0002,
            0x0002_0504,
            0x7E08_02E9,
            0x9360_1202,
            0x7E0A_02E9,
            0x9360_1203,
            0xD564_0002,
            0x0002_0504,
            0x7E08_02E9,
            0x0009_2402,
            0x7E0A_02E9,
            0x0009_2403,
            0xD564_0002,
            0x0002_0504,
            0x7E08_02FA,
            0xFF09_4002,
            0x7E0A_02FA,
            0xFF09_4003,
            0xD564_0002,
            0x0002_0504,
            0xD778_0004,
            0x0201_0102,
            0xD778_0005,
            0x0201_0103,
            0xD564_0002,
            0x0002_0504,
            0xD760_0012,
            0x0001_4102,
            0xD760_0013,
            0x0001_4103,
            0xD564_0002,
            0x0002_0412,
            0xD760_0012,
            0x0001_0102,
            0xD760_0013,
            0x0001_0103,
            0xBE94_0312,
            0xBE95_0313,
            0xBEFE_0410,
        ];
        let (p, s) = section_shapes(&words, true, &[Reg::S(20), Reg::S(21)]);
        let (lo, hi) = (input(&p, Reg::V(29)), input(&p, Reg::V(30)));
        let m = p.blocks[0].insts[3].write(Reg::S(18)).unwrap();
        assert_eq!(s[0], Some(Shape::Active(Op::AddF64Lo(hi), lo, m, 0)));
        assert_eq!(s[1], Some(Shape::Active(Op::AddF64Hi(lo), hi, m, 0)));
    }

    #[test]
    fn f16_sum() {
        // Wave64 WaveActiveSum(half v8) with v_add_f16 steps.
        let words = [
            0xBE80_25C1,
            0xD501_0001,
            0x0002_1080,
            0x6402_02E9,
            0xDE54_C101,
            0x6402_02E9,
            0x9360_1201,
            0x6402_02E9,
            0x0009_2401,
            0x6402_02FA,
            0xFF09_4001,
            0xD778_0002,
            0x0201_0101,
            0x6402_0302,
            0xD760_0002,
            0x0001_4101,
            0x6402_0202,
            0xD760_0002,
            0x0001_0101,
            0xBEFE_0400,
        ];
        let (p, s) = section_shapes(&words, true, &[Reg::S(2)]);
        let x = input(&p, Reg::V(8));
        let m = p.blocks[0].insts[0].write(Reg::S(0)).unwrap();
        assert_eq!(s[0], Some(Shape::Active(Op::AddF16, x, m, 0)));
    }
}
