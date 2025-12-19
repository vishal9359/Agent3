"""
Setup script for Agent5 V4
"""

from setuptools import setup, find_packages

setup(
    name="agent5-v4",
    version="4.0.0",
    description="C++ Flowchart Generator with DocAgent-Inspired Bottom-Up Semantic Aggregation",
    author="Agent5 Team",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.2.62,<0.3.0",
        "langchain-core>=0.3.45,<0.4.0",
        "langchain-community>=0.3.13,<0.4.0",
        "langchain-chroma>=0.1.0",
        "langchain-text-splitters>=0.3.7,<0.4.0",
        "langchain-ollama>=0.2.0,<0.3.0",
        "chroma-hnswlib==0.7.6",
        "chromadb==0.5.23",
        "tree-sitter==0.25.2",
        "tree-sitter-cpp==0.23.4",
        "libclang==18.1.1",
        "pydantic==2.10.4",
        "python-dotenv==1.0.1",
        "tqdm==4.67.1",
        "rich==13.9.4",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "agent5-v4=agent5.cli_v4:cli_v4",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Documentation",
        "Topic :: Software Development :: Code Generators",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)



