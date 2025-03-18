# Dataset settings
dataset = dict(
    type="SunObservationDataset",
    transform_name="center",
    time_series_dir="/content/dataset/validation/figure/360p/L16-S8/",
    brightness_dir="/content/dataset/validation/brightness/L16-S8/",
)
bucket_config = {
    "360p": {
        16: (1.0, 3),
    },
}
grad_checkpoint = True

# Add this to your configuration:
batch_size = 8  

# Acceleration settings
num_workers = 6
num_bucket_build_workers = 12
dtype = "bf16"
plugin = "zero1"

# Model settings
model = dict(
    type="Sunspot_STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    ignore_mismatched_sizes=True,
    class_dropout_prob=0.1,
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
    freeze_other=False, 
    init_cross_attn=False,
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
outputs = "outputs"
wandb = True
epochs = 1000
log_every = 10
ckpt_every = 250

# optimization settings
lr = 5e-5
warmup_steps = 500
grad_clip = 1.0
adam_eps = 1e-15
ema_decay = 0.99
accumulation_steps = 2
