
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
 
import torch
import torch.nn as nn
from collections import defaultdict
 
# ── config constants matching layout_tiny.py ────────────────────────────────
BEV_H       = 50
BEV_W       = 50
DIM         = 256
NUM_BBOXES  = 300
NUM_CLASSES = 12   # 10 + 2
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE       = torch.float32
 
print(f"\n{'='*70}")
print(f"  BEVDiffuser Profiler  |  device={DEVICE}  |  BEV={BEV_H}x{BEV_W}  |  C={DIM}")
print(f"{'='*70}\n")
 
# ── build the model from your real config ────────────────────────────────────
try:
    from model_utils import build_unet
    from mmcv import Config
    cfg_path = '../configs/bevdiffuser/layout_tiny.py'
    bev_cfg  = Config.fromfile(cfg_path)
    unet     = build_unet(bev_cfg.unet).to(DEVICE).eval()
    print("[OK] Model built from layout_tiny.py config")
except Exception as e:
    print(f"[WARN] Could not build from config ({e})")
    print("       Falling back to direct import with defaults.")
    from layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel
    from layout_diffusion.layout_encoder import LayoutTransformerEncoder
    le = LayoutTransformerEncoder(
        layout_length=NUM_BBOXES, hidden_dim=DIM, output_dim=DIM*4,
        num_layers=6, num_heads=8, use_final_ln=True,
        num_classes_for_layout_object=NUM_CLASSES, mask_size_for_layout_object=0,
        used_condition_types=['obj_class','obj_bbox','is_valid_obj'],
        resolution_to_attention=[12,25,50], use_3d_bbox=True,
        num_temporal_frames=1,
    )
    unet = LayoutDiffusionUNetModel(
        layout_encoder=le, in_channels=DIM, model_channels=DIM,
        out_channels=DIM, num_res_blocks=2, attention_ds=[4,2,1],
        encoder_channels=DIM, channel_mult=[1,2,4], num_heads=8,
        num_head_channels=32, resblock_updown=True, num_attention_blocks=1,
        use_positional_embedding_for_attention=True, image_size=BEV_H,
        attention_block_type='ObjectAwareCrossAttention',
        use_scale_shift_norm=True, use_preconditioning=True,
    ).to(DEVICE).eval()
    print("[OK] Model built with hardcoded defaults")
 
# ── fake inputs ──────────────────────────────────────────────────────────────
B = 1
x         = torch.randn(B, DIM, BEV_H, BEV_W, device=DEVICE, dtype=DTYPE)
timesteps = torch.tensor([500], device=DEVICE)
obj_class    = torch.zeros(B, NUM_BBOXES, dtype=torch.long,  device=DEVICE)
obj_bbox     = torch.zeros(B, NUM_BBOXES, 9,  device=DEVICE, dtype=DTYPE)
is_valid_obj = torch.zeros(B, NUM_BBOXES,     device=DEVICE, dtype=DTYPE)
is_valid_obj[:, 0] = 1.0
obj_time     = torch.zeros(B, NUM_BBOXES, dtype=torch.long,  device=DEVICE)
 
cond = dict(obj_class=obj_class, obj_bbox=obj_bbox,
            is_valid_obj=is_valid_obj, obj_time=obj_time)
 
# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — ARCHITECTURE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PART 1 — ARCHITECTURE OVERVIEW")
print("─"*70)
 
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
 
def human(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)
 
total_params = count_params(unet)
print(f"\nTotal trainable parameters: {human(total_params)}")
print(f"Model dtype: {next(unet.parameters()).dtype}")
print(f"Device: {next(unet.parameters()).device}\n")
 
# print named top-level modules with param counts
print(f"{'Module':<45} {'Params':>10} {'Type'}")
print("─"*70)
for name, mod in unet.named_children():
    p = count_params(mod)
    mtype = type(mod).__name__
    print(f"  {name:<43} {human(p):>10}   {mtype}")
 
# skip connection map
print("\n── SKIP CONNECTION MAP ──")
print("  U-Net encoder saves intermediate features at each level,")
print("  concatenated with decoder features at matching resolution.\n")
try:
    ch   = int(unet.channel_mult[0] * unet.model_channels)
    ds   = 1
    enc_shapes = []
    print(f"  {'Level':<8} {'DS':<6} {'Resolution':<14} {'Channels':<10} {'Has Attn'}")
    print(f"  {'─'*5:<8} {'─'*4:<6} {'─'*12:<14} {'─'*8:<10} {'─'*8}")
    for level, mult in enumerate(unet.channel_mult):
        for _ in range(unet.num_res_blocks):
            c = int(mult * unet.model_channels)
            r = unet.image_size // ds
            has_attn = ds in unet.attention_ds
            enc_shapes.append((r, c))
            print(f"  enc-{level:<4} {ds:<6} {r}x{r:<11} {c:<10} {'[ATTN]' if has_attn else ''}")
        if level != len(unet.channel_mult) - 1:
            ds *= 2
    print(f"\n  bottleneck:  {unet.image_size//ds}x{unet.image_size//ds}  channels={int(unet.channel_mult[-1]*unet.model_channels)}  [ATTN]")
    print(f"\n  Decoder mirrors encoder with skip concatenation at each level.")
    print(f"  Skip: enc features (C) concatenated to dec input -> ResBlock sees C+C channels")
except Exception as e:
    print(f"  (could not extract shape info: {e})")
 
# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — LAYER-BY-LAYER TIMING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PART 2 — LAYER-BY-LAYER TIMING PROFILER")
print("─"*70)
print("  Each named module is hooked. Times are wall-clock (ms).")
print("  GPU times are synchronised before/after each module.\n")
 
timing   = defaultdict(list)
mem_use  = defaultdict(list)
hooks    = []
 
def make_hook(name):
    def pre(module, inp):
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        module._t0 = time.perf_counter()
        if DEVICE == 'cuda':
            module._m0 = torch.cuda.memory_allocated()
    def post(module, inp, out):
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - module._t0) * 1000   # ms
        timing[name].append(dt)
        if DEVICE == 'cuda':
            dm = torch.cuda.memory_allocated() - module._m0
            mem_use[name].append(dm / 1024**2)           # MB
    return pre, post
 
for name, module in unet.named_modules():
    if name == '': continue
    pre_h, post_h = make_hook(name)
    hooks.append(module.register_forward_pre_hook(pre_h))
    hooks.append(module.register_forward_hook(post_h))
 
# warmup
print("  Warming up (3 passes)...")
with torch.no_grad():
    for _ in range(3):
        _ = unet(x, timesteps, **cond)
    timing.clear(); mem_use.clear()
 
# measure (5 passes, take mean)
print("  Measuring (5 passes)...")
with torch.no_grad():
    for _ in range(5):
        _ = unet(x, timesteps, **cond)
 
for h in hooks:
    h.remove()
 
# aggregate
results = {}
for name, times in timing.items():
    results[name] = {
        'mean_ms': sum(times)/len(times),
        'max_ms':  max(times),
        'mem_mb':  sum(mem_use.get(name,[0]))/max(len(mem_use.get(name,[1])),1),
    }
 
# total time
total_ms = results.get('', {}).get('mean_ms', None)
if total_ms is None:
    # compute from all top-level children
    top = [n for n in results if '.' not in n]
    total_ms = sum(results[n]['mean_ms'] for n in top)
 
print(f"\n  Total forward pass: {total_ms:.1f} ms\n")
 
# rank by time, show top 30
ranked = sorted(results.items(), key=lambda x: x[1]['mean_ms'], reverse=True)
print(f"  {'Rank':<5} {'Module Name':<50} {'Time(ms)':>9} {'%Total':>8} {'Mem(MB)':>9}")
print(f"  {'─'*4:<5} {'─'*48:<50} {'─'*7:>9} {'─'*6:>8} {'─'*7:>9}")
for rank, (name, stat) in enumerate(ranked[:30], 1):
    pct = 100 * stat['mean_ms'] / total_ms if total_ms else 0
    mem = stat['mem_mb']
    bar = '█' * int(pct / 2)
    print(f"  {rank:<5} {name:<50} {stat['mean_ms']:>8.2f}  {pct:>7.1f}%  {mem:>8.1f}  {bar}")
 
# group by category
print("\n  ── TIME GROUPED BY MODULE TYPE ──")
group = defaultdict(float)
for name, stat in results.items():
    t = type(dict(unet.named_modules()).get(name, nn.Module())).__name__
    group[t] += stat['mean_ms']
for k, v in sorted(group.items(), key=lambda x: -x[1])[:15]:
    pct = 100*v/total_ms if total_ms else 0
    print(f"  {k:<40} {v:>8.2f} ms  ({pct:.1f}%)")
 
# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — BOTTLENECK SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PART 3 — BOTTLENECK SUMMARY & RECOMMENDATIONS")
print("─"*70)
 
# find the layout encoder time
le_time = results.get('layout_encoder', {}).get('mean_ms', 0)
precond_time = results.get('precondition_attn', {}).get('mean_ms', 0)
middle_time  = results.get('middle_block',   {}).get('mean_ms', 0)
 
print(f"\n  Layout encoder (LFM):          {le_time:>8.2f} ms  ({100*le_time/total_ms:.1f}%)")
print(f"  Pre-conditioning attn (Opt 3): {precond_time:>8.2f} ms  ({100*precond_time/total_ms:.1f}%)")
print(f"  Middle block (bottleneck):     {middle_time:>8.2f} ms  ({100*middle_time/total_ms:.1f}%)")
 
# find all attention layers
attn_total = sum(v['mean_ms'] for k,v in results.items()
                 if 'cross' in k.lower() or 'attention' in k.lower() or 'attn' in k.lower())
res_total  = sum(v['mean_ms'] for k,v in results.items()
                 if 'resblock' in k.lower() or ('res' in k.lower() and 'block' in k.lower()))
 
print(f"\n  All attention layers combined: {attn_total:>8.2f} ms  ({100*attn_total/total_ms:.1f}%)")
print(f"  All ResBlocks combined:        {res_total:>8.2f} ms  ({100*res_total/total_ms:.1f}%)")
 
print(f"\n  TOP 3 BOTTLENECKS:")
for i, (name, stat) in enumerate(ranked[:3], 1):
    pct = 100*stat['mean_ms']/total_ms if total_ms else 0
    print(f"    {i}. [{pct:.1f}%]  {name}  ({stat['mean_ms']:.2f} ms)")
 
print(f"\n  WHAT TO LOOK FOR:")
print(f"  - If attention layers dominate → reducing num_heads or using linear attention helps")
print(f"  - If LFM (layout encoder) is large → LFM depth/width is your tuning lever")
print(f"  - If ResBlocks dominate → channel_mult or num_res_blocks is the knob")
print(f"  - If pre-conditioning is large → this is YOUR Option 3 cost; note it vs benefit")
print(f"  - If data loading/bev_model dominate → those are outside the diffuser")
 
print(f"\n{'='*70}")
print(f"  Done. Copy the output above and analyse it.")
print(f"{'='*70}\n")
 