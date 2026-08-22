"""
Smoke test — run from BEVFormer/projects/bevdiffuser/
  python smoke_test.py
Should print: SMOKE TEST PASSED
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
from layout_diffusion.layout_uvit import LayoutDiTModel, LayoutDiTModelIsotropic

# layout_encoder params — copied from layout_tiny.py
le_params = dict(
    used_condition_types=['obj_class', 'obj_bbox', 'is_valid_obj'],
    layout_length=300,
    num_classes_for_layout_object=12,
    mask_size_for_layout_object=0,
    hidden_dim=256,
    output_dim=1024,
    num_layers=6,
    num_heads=8,
    use_final_ln=True,
    use_positional_embedding=False,
    resolution_to_attention=[12, 25, 50],
    use_key_padding_mask=False,
    use_3d_bbox=True,
)

for ModelClass, name in [
    (LayoutDiTModel,          "U-ViT (long skips)"),
    (LayoutDiTModelIsotropic, "DiT  (no skips)   "),
]:
    print(f"\nTesting {name} ...")
    model = ModelClass(
        image_size=50,
        in_channels=256,
        out_channels=256,
        dim=512,
        depth=4,        # small for speed
        num_heads=8,
        patch=1,
        mlp_ratio=4.0,
        layout_encoder=dict(
            type='layout_diffusion.layout_encoder.LayoutTransformerEncoder',
            parameters=le_params,
        ),
    )
    model.eval()

    B = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x  = torch.randn(B, 256, 50, 50).to(device)
    t  = torch.randint(0, 1000, (B,)).to(device)
    cond = dict(
        obj_class    = torch.zeros(B, 300, dtype=torch.long).to(device),
        obj_bbox     = torch.zeros(B, 300, 9).to(device),
        is_valid_obj = torch.zeros(B, 300).to(device),
    )


    with torch.no_grad():
        out = model(x, t, **cond)

    assert isinstance(out, tuple),            "forward must return a tuple"
    assert out[0].shape == (B, 256, 50, 50),  f"wrong shape: {out[0].shape}"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  output shape : {out[0].shape}  ✓")
    print(f"  parameters   : {n_params:.1f} M")

print("\nSMOKE TEST PASSED")
