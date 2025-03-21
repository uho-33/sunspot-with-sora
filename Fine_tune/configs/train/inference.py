
# Add this to your configuration:
# Log settings
seed = 42
save_dir = "outputs/inference/test2"
image_size=(240,240)
num_frames=16
fps=8
batch_size = 1  
prompt_path = "/content/dataset/training/brightness/L16-S8/20211212_120000.npz"
text_encoder_mapping_size = 256
frame_interval = 1
prompt_filename_as_path = True


# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero1"

# Model settings
model = dict(
    type="Sunspot_STDiT3-XL/2",
    from_pretrained="outputs/0009-Sunspot_STDiT3-XL-2/epoch18-global_step200",
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
    micro_batch_size_2d=1,   
    micro_frame_size=1,      
    use_tiled_conv3d=False,     
    tile_size=2,
    normalization="video",
    temporal_overlap=False,
    force_huggingface=True,     
)
text_encoder = dict(
    type="fourier",
    from_pretrained=None,
    mapping_size=text_encoder_mapping_size,  
    model_max_length=64,
    shardformer=True,
)
scheduler = dict(
    type="rflow",
    sample_method="logit-normal",
    use_timestep_transform=True,
)


