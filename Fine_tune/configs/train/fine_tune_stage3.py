# Dataset settings
dataset = dict(
    type="SunObservationDataset",
    transform_name="center",
    time_series_dir="/content/dataset/training/figure/360p/L16-S8/",
    brightness_dir="/content/dataset/training/brightness/L16-S8/",
)
bucket_config = {
    "360p": {
        16: (1.0, 3),
    },
}
grad_checkpoint = True

# Add this to your configuration:
batch_size = 32  

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
    ignore_mismatched_sizes=True,
    class_dropout_prob=0.2,
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
    freeze_other=False, 
    init_cross_attn=False,
    orig_mapping_size=256,
    new_mapping_size=1024,
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
    mapping_size=1024,  # Changed from 256 to 1024
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
epochs = 2000
log_every = 10
ckpt_every = 200

# optimization settings
lr = 2e-5
warmup_steps = 500
grad_clip = 1.0
adam_eps = 1e-15
ema_decay = 0.99
accumulation_steps = 2
use_cosine_scheduler = True
