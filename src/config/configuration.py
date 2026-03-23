import yaml
from poultry_disease.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from pathlib import Path


class ConfigurationManager:

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):

        with open(config_filepath) as f:
            self.config = yaml.safe_load(f)

        with open(params_filepath) as f:
            self.params = yaml.safe_load(f)


    def get_prepare_base_model_config(self):

        config = self.config['prepare_base_model']

        return {
            "root_dir": Path(config['root_dir']),
            "base_model_path": Path(config['base_model_path']),
            "updated_base_model_path": Path(config['updated_base_model_path']),
            "params_image_size": self.params['IMAGE_SIZE'],
            "params_learning_rate": self.params['LEARNING_RATE'],
            "params_classes": self.params['CLASSES']
        }


    def get_training_config(self):

        config = self.config['training']

        return {
            "root_dir": Path(config['root_dir']),
            "trained_model_path": Path(config['trained_model_path']),
            "params_epochs": self.params['EPOCHS'],
            "params_batch_size": self.params['BATCH_SIZE']
        }