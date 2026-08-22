# ============================================================
# Config for isotropic DiT backbone (exp02 — comparison arm)
# Same as layout_tiny_uvit.py but type='LayoutDiTModelIsotropic'
# ============================================================

unet = dict(
    type='LayoutDiTModelIsotropic', # DiT — NO long skip connections
    image_size=bev_h_,
    in_channels=_dim_,
    out_channels=_dim_,
    dim=512,
    depth=12,
    num_heads=8,
    patch=1,
    mlp_ratio=4.0,
    layout_encoder=dict(
        type='layout_diffusion.layout_encoder.LayoutTransformerEncoder',
        parameters=dict(
            used_condition_types=['obj_class', 'obj_bbox', 'is_valid_obj'],
            layout_length=num_bboxes,
            num_classes_for_layout_object=num_classes,
            mask_size_for_layout_object=0,
            hidden_dim=256,
            output_dim=1024,
            num_layers=6,
            num_heads=8,
            use_final_ln=True,
            use_positional_embedding=False,
            resolution_to_attention=[12, 25, 50],
            use_key_padding_mask=False,
            use_3d_bbox=use_3d_bbox,
        ),
    ),
)
