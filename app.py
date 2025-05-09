from setuptools import setup, find_packages

setup(
    name="covid-dashboard",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.20.0,<1.25.0",
        "pandas>=1.3.0,<1.6.0",
        "numpy>=1.20.0,<1.25.0",
        "plotly>=5.10.0,<5.16.0",
        "requests>=2.25.0,<2.32.0",
    ],
)