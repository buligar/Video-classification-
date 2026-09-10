# ============================================================
# VideoMAE for HMDB51-v2
# 5 classes:
#   clap
#   jump
#   run
#   walk
#   wave
#
# Dataset:
#   Sina272/hmdb51-v2
#
# Model:
#   MCG-NJU/videomae-base-finetuned-kinetics
#
# IMPORTANT FIXES:
#   - metadata_errors="ignore"
#   - NO thread_type="AUTO"
#   - dataloader_num_workers=0
#   - no preliminary full dataset AVI validation
# ============================================================

import os
import gc
import random
from pathlib import Path

import av
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from huggingface_hub import snapshot_download

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

DATASET_ID = "Sina272/hmdb51-v2"

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"

OUTPUT_DIR = "./videomae_hmdb51_5classes"



CLASSES = [
    "clap",
    "jump",
    "run",
    "walk",
    "wave",
]

NUM_CLASSES = len(CLASSES)

NUM_FRAMES = 16

FRAME_SAMPLE_RATE = 4

VAL_RATIO = 0.20

NUM_EPOCHS = 30

LEARNING_RATE = 5e-5

WEIGHT_DECAY = 0.05

WARMUP_RATIO = 0.10

TRAIN_BATCH_SIZE = 2

EVAL_BATCH_SIZE = 2

GRAD_ACCUMULATION_STEPS = 2

EARLY_STOPPING_PATIENCE = 7

# IMPORTANT:
# PyAV + multiprocessing can hang.
NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

set_seed(SEED)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# DEVICE
# ============================================================

print("=" * 80)
print("DEVICE")
print("=" * 80)


if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("CUDA: YES")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print(
        f"GPU memory: {gpu_memory:.2f} GB"
    )

else:

    DEVICE = torch.device("cpu")

    print("CUDA: NO")
    print("Using CPU")


# ============================================================
# LABELS
# ============================================================

label2id = {

    label: idx

    for idx, label in enumerate(CLASSES)
}


id2label = {

    idx: label

    for label, idx in label2id.items()
}


print("\nLabels:")

print(label2id)


# ============================================================
# DOWNLOAD HMDB51 SUBSET
# ============================================================

print("\n" + "=" * 80)
print("DOWNLOADING HMDB51 SUBSET")
print("=" * 80)


allow_patterns = [

    "metadata_train.csv",

    "metadata_test.csv",
]


for class_name in CLASSES:

    allow_patterns.append(

        f"videos/{class_name}/*"

    )


dataset_root = snapshot_download(

    repo_id=DATASET_ID,

    repo_type="dataset",

    allow_patterns=allow_patterns,
)


dataset_root = Path(
    dataset_root
)


print(
    "\nDataset root:"
)

print(
    dataset_root
)


# ============================================================
# LOAD METADATA
# ============================================================

train_csv = (
    dataset_root
    / "metadata_train.csv"
)

test_csv = (
    dataset_root
    / "metadata_test.csv"
)


train_df_full = pd.read_csv(
    train_csv
)

test_df = pd.read_csv(
    test_csv
)


print(
    "\nOriginal train:",
    train_df_full.shape
)

print(
    "Original test:",
    test_df.shape
)


# ============================================================
# FILTER 5 CLASSES
# ============================================================

train_df_full = train_df_full[
    train_df_full["label"].isin(
        CLASSES
    )
].copy()


test_df = test_df[
    test_df["label"].isin(
        CLASSES
    )
].copy()


train_df_full = (
    train_df_full
    .reset_index(drop=True)
)

test_df = (
    test_df
    .reset_index(drop=True)
)


print("\n" + "=" * 80)
print("FILTERED DATASET")
print("=" * 80)


print("\nTrain:")

print(

    train_df_full[
        "label"
    ]

    .value_counts()

    .sort_index()
)


print("\nTest:")

print(

    test_df[
        "label"
    ]

    .value_counts()

    .sort_index()
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_df, val_df = train_test_split(

    train_df_full,

    test_size=VAL_RATIO,

    random_state=SEED,

    stratify=train_df_full["label"],
)


train_df = train_df.reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)


print("\n" + "=" * 80)
print("FINAL SPLITS")
print("=" * 80)


print(
    f"Train: {len(train_df)}"
)

print(
    f"Val:   {len(val_df)}"
)

print(
    f"Test:  {len(test_df)}"
)


print(
    "\nTrain distribution:"
)

print(

    train_df[
        "label"
    ]

    .value_counts()

    .sort_index()
)


print(
    "\nValidation distribution:"
)

print(

    val_df[
        "label"
    ]

    .value_counts()

    .sort_index()
)


print(
    "\nTest distribution:"
)

print(

    test_df[
        "label"
    ]

    .value_counts()

    .sort_index()
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE SPLITS
# ============================================================

train_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "train_split.csv"
    ),

    index=False
)


val_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "val_split.csv"
    ),

    index=False
)


test_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "test_split.csv"
    ),

    index=False
)


# ============================================================
# IMAGE PROCESSOR
# ============================================================

print("\n" + "=" * 80)
print("IMAGE PROCESSOR")
print("=" * 80)


image_processor = (
    VideoMAEImageProcessor
    .from_pretrained(
        MODEL_NAME
    )
)


print(
    image_processor
)


# ============================================================
# ROBUST VIDEO DECODER
# ============================================================

def decode_video(video_path):

    """
    Robust video decoding for HMDB51.

    Important:
    - metadata_errors="ignore"
    - NO thread_type="AUTO"

    Returns:
        list of RGB numpy frames
    """

    frames = []

    container = None

    try:

        container = av.open(

            str(video_path),

            mode="r",

            metadata_errors="ignore",
        )


        if len(
            container.streams.video
        ) == 0:

            raise RuntimeError(
                "No video stream"
            )


        # IMPORTANT:
        # Do NOT set:
        #
        # stream.thread_type = "AUTO"
        #
        # It can hang on some AVI files.


        for frame in container.decode(
            video=0
        ):

            rgb = frame.to_ndarray(
                format="rgb24"
            )

            frames.append(
                rgb
            )


    except Exception as e:

        raise RuntimeError(

            f"\nCannot decode video:\n"
            f"{video_path}\n"
            f"{type(e).__name__}: {e}"

        )


    finally:

        if container is not None:

            try:

                container.close()

            except Exception:

                pass


    if len(frames) == 0:

        raise RuntimeError(

            f"No frames decoded:\n"
            f"{video_path}"

        )


    return frames


# ============================================================
# TEMPORAL SAMPLING
# ============================================================

def sample_frame_indices(

    num_available_frames,

    num_frames=16,

    sample_rate=4,

    train=True,
):

    temporal_span = (

        (num_frames - 1)

        * sample_rate

        + 1
    )


    # ========================================================
    # LONG VIDEO
    # ========================================================

    if num_available_frames >= temporal_span:

        max_start = (

            num_available_frames

            - temporal_span
        )


        if train:

            start = random.randint(

                0,

                max_start
            )

        else:

            start = (
                max_start // 2
            )


        indices = (

            start

            + np.arange(
                num_frames
            )

            * sample_rate
        )


    # ========================================================
    # SHORT VIDEO
    # ========================================================

    else:

        indices = np.linspace(

            0,

            num_available_frames - 1,

            num_frames
        )


        indices = np.round(
            indices
        ).astype(int)


    indices = np.clip(

        indices,

        0,

        num_available_frames - 1
    )


    return indices.astype(
        int
    )


# ============================================================
# AUGMENTATION
# ============================================================

def horizontal_flip_frames(

    frames,

    probability=0.5,
):

    if random.random() < probability:

        frames = [

            np.ascontiguousarray(
                frame[:, ::-1, :]
            )

            for frame in frames
        ]


    return frames


# ============================================================
# DATASET
# ============================================================

class HMDB51VideoDataset(
    Dataset
):

    def __init__(

        self,

        dataframe,

        root_dir,

        processor,

        label2id,

        train=False,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.root_dir = Path(
            root_dir
        )

        self.processor = processor

        self.label2id = label2id

        self.train = train


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]


        relative_path = (
            row["file_name"]
        )


        label_name = (
            row["label"]
        )


        video_path = (

            self.root_dir

            / relative_path
        )


        # ====================================================
        # DECODE
        # ====================================================

        try:

            frames = decode_video(
                video_path
            )

        except Exception as e:

            print(
                "\nVIDEO ERROR:"
            )

            print(
                video_path
            )

            print(
                e
            )

            raise


        # ====================================================
        # TEMPORAL SAMPLE
        # ====================================================

        indices = sample_frame_indices(

            num_available_frames=len(
                frames
            ),

            num_frames=NUM_FRAMES,

            sample_rate=FRAME_SAMPLE_RATE,

            train=self.train,
        )


        sampled_frames = [

            frames[i]

            for i in indices
        ]


        del frames


        # ====================================================
        # AUGMENTATION
        # ====================================================

        if self.train:

            sampled_frames = (
                horizontal_flip_frames(

                    sampled_frames,

                    probability=0.5,
                )
            )


        # ====================================================
        # PROCESSOR
        # ====================================================

        encoded = self.processor(

            sampled_frames,

            return_tensors="pt",
        )


        pixel_values = (
            encoded[
                "pixel_values"
            ][0]
        )


        label_id = (
            self.label2id[
                label_name
            ]
        )


        return {

            "pixel_values":
                pixel_values,

            "labels":
                label_id,
        }


# ============================================================
# DATASETS
# ============================================================

train_dataset = HMDB51VideoDataset(

    dataframe=train_df,

    root_dir=dataset_root,

    processor=image_processor,

    label2id=label2id,

    train=True,
)


val_dataset = HMDB51VideoDataset(

    dataframe=val_df,

    root_dir=dataset_root,

    processor=image_processor,

    label2id=label2id,

    train=False,
)


test_dataset = HMDB51VideoDataset(

    dataframe=test_df,

    root_dir=dataset_root,

    processor=image_processor,

    label2id=label2id,

    train=False,
)


# ============================================================
# TEST FIRST SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("CHECK SAMPLE")
print("=" * 80)


sample = train_dataset[0]


print(
    "pixel_values:",
    sample["pixel_values"].shape
)


print(
    "label:",
    sample["labels"],
    id2label[
        sample["labels"]
    ]
)


# Expected:
# torch.Size([16, 3, 224, 224])


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 80)
print("LOADING VIDEOMAE")
print("=" * 80)


model = (
    VideoMAEForVideoClassification
    .from_pretrained(

        MODEL_NAME,

        num_labels=NUM_CLASSES,

        label2id=label2id,

        id2label=id2label,

        ignore_mismatched_sizes=True,
    )
)


print(
    "Number of labels:",
    model.config.num_labels
)


print(
    "Frames:",
    model.config.num_frames
)


# ============================================================
# PARAMETERS
# ============================================================

total_params = sum(

    p.numel()

    for p in model.parameters()
)


trainable_params = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad
)


print(
    f"Total parameters: "
    f"{total_params:,}"
)


print(
    f"Trainable parameters: "
    f"{trainable_params:,}"
)


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(
    examples
):

    pixel_values = torch.stack(

        [

            example[
                "pixel_values"
            ]

            for example in examples
        ]
    )


    labels = torch.tensor(

        [

            example[
                "labels"
            ]

            for example in examples
        ],

        dtype=torch.long
    )


    return {

        "pixel_values":
            pixel_values,

        "labels":
            labels
    }


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    eval_pred
):

    logits = (
        eval_pred.predictions
    )


    labels = (
        eval_pred.label_ids
    )


    predictions = np.argmax(

        logits,

        axis=-1
    )


    accuracy = accuracy_score(

        labels,

        predictions
    )


    macro_f1 = f1_score(

        labels,

        predictions,

        average="macro",

        zero_division=0
    )


    return {

        "accuracy":
            float(accuracy),

        "f1_macro":
            float(macro_f1),
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    overwrite_output_dir=True,


    # ========================================================
    # TRAIN
    # ========================================================

    num_train_epochs=NUM_EPOCHS,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    warmup_ratio=WARMUP_RATIO,

    lr_scheduler_type="cosine",

    optim="adamw_torch",


    # ========================================================
    # BATCH
    # ========================================================

    per_device_train_batch_size=(
        TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRAD_ACCUMULATION_STEPS
    ),


    # ========================================================
    # EVALUATION
    # ========================================================

    eval_strategy="epoch",

    save_strategy="epoch",


    # ========================================================
    # LOGGING
    # ========================================================

    logging_strategy="steps",

    logging_steps=10,


    # ========================================================
    # BEST MODEL
    # ========================================================

    load_best_model_at_end=True,

    metric_for_best_model=(
        "f1_macro"
    ),

    greater_is_better=True,

    save_total_limit=2,


    # ========================================================
    # CUDA
    # ========================================================

    fp16=torch.cuda.is_available(),


    # ========================================================
    # CRITICAL PYAV FIX
    # ========================================================

    dataloader_num_workers=0,

    dataloader_pin_memory=(
        torch.cuda.is_available()
    ),

    dataloader_persistent_workers=False,


    # ========================================================
    # IMPORTANT
    # ========================================================

    remove_unused_columns=False,


    # ========================================================
    # RANDOM
    # ========================================================

    seed=SEED,

    data_seed=SEED,


    # ========================================================
    # LOGGING
    # ========================================================

    report_to="none",

    push_to_hub=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=image_processor,

    data_collator=collate_fn,

    compute_metrics=compute_metrics,

    callbacks=[

        EarlyStoppingCallback(

            early_stopping_patience=(
                EARLY_STOPPING_PATIENCE
            )
        )
    ]
)


# ============================================================
# CLEAR MEMORY
# ============================================================

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)


train_result = trainer.train()


print("\n" + "=" * 80)
print("TRAINING FINISHED")
print("=" * 80)


print(
    train_result
)


print(
    "\nBest checkpoint:",
    trainer.state.best_model_checkpoint
)


print(
    "Best validation macro F1:",
    trainer.state.best_metric
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

BEST_MODEL_DIR = (

    Path(
        OUTPUT_DIR
    )

    / "best_model"
)


trainer.save_model(
    str(BEST_MODEL_DIR)
)


image_processor.save_pretrained(
    str(BEST_MODEL_DIR)
)


print(
    "\nBest model saved:"
)

print(
    BEST_MODEL_DIR
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)


val_metrics = trainer.evaluate(
    val_dataset
)


for key, value in val_metrics.items():

    print(
        key,
        value
    )


# ============================================================
# CLEAR MEMORY BEFORE TEST
# ============================================================

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST")
print("=" * 80)


print(
    f"Test videos: "
    f"{len(test_dataset)}"
)


# ============================================================
# IMPORTANT:
# Trainer.predict now runs with:
# dataloader_num_workers = 0
# ============================================================

test_output = trainer.predict(
    test_dataset
)


print(
    "\nTest prediction finished."
)


# ============================================================
# TEST OUTPUT
# ============================================================

test_logits = (
    test_output.predictions
)


test_labels = (
    test_output.label_ids
)


test_predictions = np.argmax(

    test_logits,

    axis=-1
)


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(

    test_labels,

    test_predictions
)


test_macro_f1 = f1_score(

    test_labels,

    test_predictions,

    average="macro",

    zero_division=0
)


print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)


print(
    f"Accuracy: "
    f"{test_accuracy:.4f}"
)


print(
    f"Macro F1: "
    f"{test_macro_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    test_labels,

    test_predictions,

    labels=list(
        range(NUM_CLASSES)
    ),

    target_names=CLASSES,

    digits=4,

    zero_division=0
)


print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)


print(
    report
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(

    os.path.join(

        OUTPUT_DIR,

        "classification_report.txt"
    ),

    "w",

    encoding="utf-8"

) as f:

    f.write(

        f"Accuracy: "
        f"{test_accuracy:.6f}\n"

    )

    f.write(

        f"Macro F1: "
        f"{test_macro_f1:.6f}\n\n"

    )

    f.write(
        report
    )


# ============================================================
# PROBABILITIES
# ============================================================

probabilities = torch.softmax(

    torch.tensor(
        test_logits
    ),

    dim=1

).numpy()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

pred_df = test_df.copy()


pred_df[
    "true_id"
] = test_labels


pred_df[
    "predicted_id"
] = test_predictions


pred_df[
    "predicted_label"
] = [

    id2label[
        int(pred)
    ]

    for pred in test_predictions
]


pred_df[
    "correct"
] = (

    pred_df[
        "true_id"
    ]

    ==

    pred_df[
        "predicted_id"
    ]
)


for class_id, class_name in id2label.items():

    pred_df[
        f"prob_{class_name}"
    ] = probabilities[
        :,
        class_id
    ]


pred_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "test_predictions.csv"
    ),

    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    test_labels,

    test_predictions,

    labels=list(
        range(NUM_CLASSES)
    )
)


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=CLASSES
)


fig, ax = plt.subplots(

    figsize=(8, 7)
)


disp.plot(

    ax=ax,

    values_format="d"
)


plt.title(

    "VideoMAE — HMDB51 5 classes"

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "confusion_matrix.png"
    ),

    dpi=250,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_norm = confusion_matrix(

    test_labels,

    test_predictions,

    labels=list(
        range(NUM_CLASSES)
    ),

    normalize="true"
)


disp_norm = ConfusionMatrixDisplay(

    confusion_matrix=cm_norm,

    display_labels=CLASSES
)


fig, ax = plt.subplots(

    figsize=(8, 7)
)


disp_norm.plot(

    ax=ax,

    values_format=".2f"
)


plt.title(

    "VideoMAE — Normalized confusion matrix"

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "confusion_matrix_normalized.png"
    ),

    dpi=250,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# TRAINING HISTORY
# ============================================================

history = trainer.state.log_history


history_df = pd.DataFrame(
    history
)


history_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "training_history.csv"
    ),

    index=False
)


# ============================================================
# EXTRACT HISTORY
# ============================================================

train_epochs = []

train_losses = []

eval_epochs = []

eval_losses = []

eval_accuracy = []

eval_f1 = []


for record in history:

    # TRAIN LOSS

    if (

        "loss" in record

        and

        "eval_loss" not in record

        and

        "epoch" in record
    ):

        train_epochs.append(
            record["epoch"]
        )

        train_losses.append(
            record["loss"]
        )


    # VALIDATION

    if (

        "eval_loss" in record

        and

        "epoch" in record
    ):

        eval_epochs.append(
            record["epoch"]
        )

        eval_losses.append(
            record["eval_loss"]
        )

        eval_accuracy.append(

            record.get(

                "eval_accuracy",

                np.nan
            )
        )

        eval_f1.append(

            record.get(

                "eval_f1_macro",

                np.nan
            )
        )


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(
    figsize=(9, 5)
)


if len(train_epochs) > 0:

    plt.plot(

        train_epochs,

        train_losses,

        label="Train loss"
    )


if len(eval_epochs) > 0:

    plt.plot(

        eval_epochs,

        eval_losses,

        marker="o",

        label="Validation loss"
    )


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "VideoMAE — Loss"
)


plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "loss_curve.png"
    ),

    dpi=250,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# VALIDATION ACCURACY
# ============================================================

if len(eval_epochs) > 0:

    plt.figure(
        figsize=(9, 5)
    )


    plt.plot(

        eval_epochs,

        eval_accuracy,

        marker="o"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Validation accuracy"
    )


    plt.ylim(
        0,
        1
    )


    plt.title(
        "VideoMAE — Validation accuracy"
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            "accuracy_curve.png"
        ),

        dpi=250,

        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# VALIDATION MACRO F1
# ============================================================

if len(eval_epochs) > 0:

    plt.figure(
        figsize=(9, 5)
    )


    plt.plot(

        eval_epochs,

        eval_f1,

        marker="o"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Validation macro F1"
    )


    plt.ylim(
        0,
        1
    )


    plt.title(
        "VideoMAE — Validation Macro F1"
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            "f1_curve.png"
        ),

        dpi=250,

        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# ACCURACY + F1
# ============================================================

if len(eval_epochs) > 0:

    plt.figure(
        figsize=(9, 5)
    )


    plt.plot(

        eval_epochs,

        eval_accuracy,

        marker="o",

        label="Accuracy"
    )


    plt.plot(

        eval_epochs,

        eval_f1,

        marker="o",

        label="Macro F1"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Score"
    )


    plt.ylim(
        0,
        1
    )


    plt.title(
        "VideoMAE — Validation metrics"
    )


    plt.grid(
        alpha=0.3
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            "validation_metrics.png"
        ),

        dpi=250,

        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# ACCURACY BY CLASS
# ============================================================

print("\n" + "=" * 80)
print("ACCURACY BY CLASS")
print("=" * 80)


class_accuracies = {}


for class_id in range(
    NUM_CLASSES
):

    mask = (
        test_labels == class_id
    )


    class_acc = accuracy_score(

        test_labels[
            mask
        ],

        test_predictions[
            mask
        ]
    )


    class_name = id2label[
        class_id
    ]


    class_accuracies[
        class_name
    ] = class_acc


    print(

        f"{class_name:10s}: "
        f"{class_acc:.4f}"
    )


# ============================================================
# CLASS ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.bar(

    list(
        class_accuracies.keys()
    ),

    list(
        class_accuracies.values()
    )
)


plt.xlabel(
    "Class"
)


plt.ylabel(
    "Accuracy"
)


plt.ylim(
    0,
    1
)


plt.title(
    "VideoMAE — Test accuracy by class"
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "class_accuracy.png"
    ),

    dpi=250,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# SINGLE TEST VIDEO
# ============================================================

print("\n" + "=" * 80)
print("SINGLE VIDEO EXAMPLE")
print("=" * 80)


example_index = 0


example = test_dataset[
    example_index
]


pixel_values = (

    example[
        "pixel_values"
    ]

    .unsqueeze(0)

    .to(
        trainer.model.device
    )
)


true_id = int(
    example[
        "labels"
    ]
)


trainer.model.eval()


with torch.no_grad():

    outputs = trainer.model(

        pixel_values=pixel_values
    )


    probs = torch.softmax(

        outputs.logits,

        dim=-1

    )[0]


predicted_id = int(

    probs.argmax().item()
)


print(
    "\nVideo:"
)

print(

    test_df.iloc[
        example_index
    ]["file_name"]
)


print(

    "\nTrue:",
    id2label[
        true_id
    ]
)


print(

    "Predicted:",
    id2label[
        predicted_id
    ]
)


print(
    "\nProbabilities:"
)


sorted_indices = torch.argsort(

    probs,

    descending=True
)


for idx in sorted_indices:

    idx = int(
        idx.item()
    )

    print(

        f"{id2label[idx]:10s}: "
        f"{probs[idx].item():.4f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)


print(
    f"Train videos:      "
    f"{len(train_dataset)}"
)


print(
    f"Validation videos: "
    f"{len(val_dataset)}"
)


print(
    f"Test videos:       "
    f"{len(test_dataset)}"
)


print(
    f"\nTest accuracy: "
    f"{test_accuracy:.4f}"
)


print(
    f"Test macro F1: "
    f"{test_macro_f1:.4f}"
)


print(
    "\nBest checkpoint:"
)

print(
    trainer.state.best_model_checkpoint
)


print(
    "\nOutput:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)