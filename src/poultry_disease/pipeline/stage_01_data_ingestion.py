from poultry_disease.config.configuration import ConfigurationManager
from poultry_disease.components.data_ingestion import DataIngestion


class DataIngestionTrainingPipeline:

    def main(self):

        config = ConfigurationManager()

        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(data_ingestion_config)

        data_ingestion.download_file()

        data_ingestion.extract_zip_file()


if __name__ == "__main__":

    obj = DataIngestionTrainingPipeline()

    obj.main()