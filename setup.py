from setuptools import setup, find_packages

setup(
    name="drinks-quality-prediction",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11,<3.13",
    install_requires=[
        "numpy>=1.26,<3",
        "pandas>=2.2,<3",
        "scikit-learn==1.9.1",
        "joblib>=1.4,<2",
    ],
)
