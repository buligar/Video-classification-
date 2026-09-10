import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

SEED = 42

IMAGE_SIZE = 64

CONTEXT_FRAMES = 5
SEQUENCE_LENGTH = 15

TRAIN_SAMPLES = 3000
VAL_SAMPLES = 300

BATCH_SIZE = 32
EPOCHS = 30

LATENT_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3

LR = 3e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# SYNTHETIC VIDEO
# ============================================================

def generate_moving_square(
    num_frames=15,
    image_size=64,
    square_size=8,
):
    """
    Create one synthetic video:
    a square moves and bounces from image borders.
    """

    frames = []

    x = random.randint(
        0,
        image_size - square_size
    )

    y = random.randint(
        0,
        image_size - square_size
    )

    vx = random.choice(
        [-3, -2, 2, 3]
    )

    vy = random.choice(
        [-3, -2, 2, 3]
    )

    for _ in range(num_frames):

        frame = np.zeros(
            (image_size, image_size),
            dtype=np.float32
        )

        frame[
            y:y + square_size,
            x:x + square_size
        ] = 1.0

        frames.append(frame)

        x += vx
        y += vy

        # bounce

        if x <= 0:
            x = 0
            vx *= -1

        if x >= image_size - square_size:
            x = image_size - square_size
            vx *= -1

        if y <= 0:
            y = 0
            vy *= -1

        if y >= image_size - square_size:
            y = image_size - square_size
            vy *= -1

    frames = np.stack(frames)

    # T H W
    # ->
    # T C H W

    frames = torch.tensor(
        frames
    ).unsqueeze(1)

    return frames


# ============================================================
# DATASET
# ============================================================

class SyntheticVideoDataset(Dataset):

    def __init__(
        self,
        num_samples,
        sequence_length,
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        video = generate_moving_square(
            num_frames=self.sequence_length,
            image_size=IMAGE_SIZE,
        )

        # Input:
        #
        # frame 0 ... frame T-2
        #
        # Target:
        #
        # frame 1 ... frame T-1

        x = video[:-1]
        y = video[1:]

        return x, y


train_dataset = SyntheticVideoDataset(
    TRAIN_SAMPLES,
    SEQUENCE_LENGTH,
)

val_dataset = SyntheticVideoDataset(
    VAL_SAMPLES,
    SEQUENCE_LENGTH,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# FRAME ENCODER
# ============================================================

class FrameEncoder(nn.Module):

    def __init__(self, latent_dim=128):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(),

            # 64 -> 32

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(),

            # 32 -> 16

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(),

            # 16 -> 8
        )

        self.fc = nn.Linear(
            128 * 8 * 8,
            latent_dim
        )

    def forward(self, x):

        x = self.conv(x)

        x = x.flatten(1)

        x = self.fc(x)

        return x


# ============================================================
# FRAME DECODER
# ============================================================

class FrameDecoder(nn.Module):

    def __init__(self, latent_dim=128):

        super().__init__()

        self.fc = nn.Linear(
            latent_dim,
            128 * 8 * 8
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(),

            # 8 -> 16

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(),

            # 16 -> 32

            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.Sigmoid(),

            # 32 -> 64
        )

    def forward(self, z):

        x = self.fc(z)

        x = x.view(
            -1,
            128,
            8,
            8
        )

        x = self.decoder(x)

        return x


# ============================================================
# VIDEO TRANSFORMER
# ============================================================

class VideoGeneratorTransformer(nn.Module):

    def __init__(
        self,
        latent_dim=128,
        num_heads=4,
        num_layers=3,
        max_frames=100,
    ):

        super().__init__()

        self.encoder = FrameEncoder(
            latent_dim
        )

        self.decoder = FrameDecoder(
            latent_dim
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                max_frames,
                latent_dim
            ) * 0.02
        )

        transformer_layer = (
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=latent_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                transformer_layer,
                num_layers=num_layers,
            )
        )

        self.latent_prediction = nn.Linear(
            latent_dim,
            latent_dim
        )

    def create_causal_mask(
        self,
        length,
        device,
    ):

        mask = torch.triu(
            torch.ones(
                length,
                length,
                device=device
            ),
            diagonal=1
        )

        mask = mask.bool()

        return mask

    def forward(self, video):

        # video:
        #
        # B T C H W

        B, T, C, H, W = video.shape

        frames = video.reshape(
            B * T,
            C,
            H,
            W
        )

        # --------------------------------
        # CNN encoder
        # --------------------------------

        latent = self.encoder(
            frames
        )

        latent = latent.view(
            B,
            T,
            -1
        )

        latent = (
            latent
            +
            self.pos_embedding[:, :T]
        )

        # --------------------------------
        # causal Transformer
        # --------------------------------

        mask = self.create_causal_mask(
            T,
            video.device
        )

        latent = self.transformer(
            latent,
            mask=mask
        )

        latent = self.latent_prediction(
            latent
        )

        # --------------------------------
        # Decode predicted frames
        # --------------------------------

        latent = latent.reshape(
            B * T,
            -1
        )

        frames = self.decoder(
            latent
        )

        frames = frames.view(
            B,
            T,
            1,
            H,
            W
        )

        return frames


# ============================================================
# MODEL
# ============================================================

model = VideoGeneratorTransformer(
    latent_dim=LATENT_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
).to(DEVICE)


num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: "
    f"{num_parameters / 1e6:.2f} M"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

criterion = nn.MSELoss()


# ============================================================
# TRAIN
# ============================================================

def train_epoch():

    model.train()

    total_loss = 0

    progress = tqdm(
        train_loader,
        desc="Train"
    )

    for x, target in progress:

        x = x.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(x)

        loss = criterion(
            prediction,
            target
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.5f}"
        )

    return (
        total_loss
        /
        len(train_loader)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validation_epoch():

    model.eval()

    total_loss = 0

    for x, target in val_loader:

        x = x.to(DEVICE)
        target = target.to(DEVICE)

        prediction = model(x)

        loss = criterion(
            prediction,
            target
        )

        total_loss += loss.item()

    return (
        total_loss
        /
        len(val_loader)
    )


# ============================================================
# TRAINING LOOP
# ============================================================

train_losses = []
val_losses = []

best_loss = float("inf")


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss = train_epoch()

    val_loss = validation_epoch()

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    print(
        f"Epoch {epoch:02d} | "
        f"train={train_loss:.6f} | "
        f"val={val_loss:.6f}"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            "video_generator.pt"
        )

        print("Best model saved")


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    train_losses,
    label="Train"
)

plt.plot(
    val_losses,
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "video_generation_loss.png",
    dpi=200
)

plt.show()


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        "video_generator.pt",
        map_location=DEVICE,
        weights_only=True,
    )
)

model.eval()


# ============================================================
# AUTOREGRESSIVE VIDEO GENERATION
# ============================================================

@torch.no_grad()
def generate_video(
    model,
    initial_frames,
    total_frames=40,
):

    """
    initial_frames:
        T C H W
    """

    generated = (
        initial_frames
        .clone()
        .to(DEVICE)
    )

    while generated.shape[0] < total_frames:

        # model input
        #
        # 1 T C H W

        inp = generated.unsqueeze(0)

        predictions = model(inp)

        # prediction corresponding
        # to last token

        next_frame = predictions[
            0,
            -1
        ]

        generated = torch.cat(
            [
                generated,
                next_frame.unsqueeze(0)
            ],
            dim=0
        )

    return generated.cpu()


# ============================================================
# INITIAL VIDEO
# ============================================================

true_video = generate_moving_square(
    num_frames=50,
    image_size=IMAGE_SIZE,
)

initial_frames = true_video[
    :CONTEXT_FRAMES
]


# ============================================================
# GENERATE
# ============================================================

generated = generate_video(
    model,
    initial_frames,
    total_frames=50,
)

print(
    "Generated:",
    generated.shape
)


# ============================================================
# SAVE MP4
# ============================================================

def save_video(
    video,
    filename,
    fps=10,
):

    # video:
    #
    # T 1 H W

    video = (
        video
        .squeeze(1)
        .numpy()
    )

    video = np.clip(
        video * 255,
        0,
        255
    ).astype(
        np.uint8
    )

    height, width = (
        video.shape[1:]
    )

    writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    for frame in video:

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_GRAY2BGR
        )

        writer.write(
            frame_rgb
        )

    writer.release()


save_video(
    generated,
    "generated_video.mp4",
)

save_video(
    true_video,
    "ground_truth.mp4",
)

print(
    "Saved generated_video.mp4"
)