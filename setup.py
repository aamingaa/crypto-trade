from setuptools import setup, find_packages

setup(
    name="crypto-trade",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy", 
        "ta",
        "statsmodels"
    ],
    author="Your Name",
    description="Crypto trading project with technical analysis utilities"
)
