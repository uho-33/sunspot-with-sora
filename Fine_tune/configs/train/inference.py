
# Add this to your configuration:
batch_size = 1  
prompt_path = "/content/dataset/training/brightness/L16-S8/20211212_120000.npz"




# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero1"

# Model settings
model = dict(
    type="Sunspot_STDiT3-XL/2",
    from_pretrained="outputs/0003-Sunspot_STDiT3-XL-2/epoch119-global_step600",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
    training=False
)
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,
    normalization="video",
    temporal_overlap=True,
    force_huggingface=True,
)
text_encoder = dict(
    type="fourier",
    from_pretrained=None,
    mapping_size=1024,  
    model_max_length=64,
    shardformer=True,
)
scheduler = dict(
    type="rflow",
    sample_method="logit-normal",
    use_timestep_transform=True,
)

# Log settings
seed = 42
save_dir = "outputs/inference"
wandb = True
log_every = 10
