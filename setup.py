from setuptools import setup, find_packages

setup(
    name="specification-benchmark",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gdown",    
        "numpy>=1.24.0",
        "datasets",
        "llm-sandbox==0.3.18",
        "lotus-ai>=1.1",
        "langchain==0.3",       
        "langchain-anthropic==0.3",
        "langchain-openai==0.3",
        "openai",
        "anthropic",
        "tenacity",
        "streamlit>=1.49",
        "inflect",
        "beautifulsoup4",
    ],
    python_requires=">=3.8",
)
