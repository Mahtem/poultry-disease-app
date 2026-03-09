from poultry_disease.config.configuration import ConfigurationManager
from poultry_disease.components.model_trainer import ModelTrainer


class ModelTrainingPipeline:

    def main(self):

        config = ConfigurationManager()

        training_config = config.get_training_config()

        trainer = ModelTrainer(config=training_config)

        trainer.train()


if __name__ == "__main__":

    obj = ModelTrainingPipeline()

    obj.main()