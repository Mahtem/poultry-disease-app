import os
import urllib.request as request
import zipfile
from poultry_disease import logger

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info("Downloading dataset...")

            url = self.config.source_URL
            filename = self.config.local_data_file

            opener = request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            request.install_opener(opener)

            request.urlretrieve(url, filename)

            logger.info("Download completed.")
        else:
            logger.info("File already exists")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info("Extraction completed.")