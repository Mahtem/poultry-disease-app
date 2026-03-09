from pathlib import Path
from poultry_disease.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig
)
from poultry_disease.utils.common import read_yaml, create_directories


class ConfigurationManager:

    def __init__(
        self,
        config_filepath=Path("config/config.yaml"),
        params_filepath=Path("params.yaml")
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    # ========================
    # Data Ingestion
    # ========================
    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

        return data_ingestion_config


    # ========================
    # Prepare Base Model
    # ========================
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:

        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config


    # ========================
    # Training
    # ========================
    def get_training_config(self) -> TrainingConfig:

        training = self.config.training

        create_directories([training.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(training.updated_base_model_path),
            training_data=Path(training.training_data),
            training_csv=Path(self.config.training.training_csv),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_image_size=self.params.IMAGE_SIZE,
        )

        return training_config