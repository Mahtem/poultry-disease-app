from setuptools import setup, find_packages

setup(
    name="poultry_disease",
    version="0.0.0",
    author="YourName",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)