from dataclasses import dataclass
from pathlib import Path


# =========================
# Data Ingestion Config
# =========================
@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


# =========================
# Prepare Base Model Config
# =========================
@dataclass
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


# =========================
# Training Config
# =========================
@dataclass
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    training_csv: Path
    params_epochs: int
    params_batch_size: int
    params_image_size: list


# =========================
# Evaluation Config
# =========================
@dataclass
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    training_csv: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int