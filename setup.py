import os

import setuptools
from setuptools import setup

install_requires = [
    "numpy",
    "keras",
    "tokenizer_tools",
    "flask",
    "flask-cors",
    "ioflow",
    "tf-crf-layer",
    "tf-attention-layer",
    "tensorflow==1.15.0",
    "deliverable_model==0.4.1",
    "gunicorn",
    "micro_toolkit",
    "seq2annotation"
]


setup(
    # _PKG_NAME will be used in Makefile for dev release
    name=os.getenv("_PKG_NAME", "mtnlpmodel"),
    version="0.0.2",
    packages=setuptools.find_packages(),
    include_package_data=True,
    url="https://github.com/xiaomihao/mtnlp_model",
    license="Apache 2.0",
    author="Xiao Mi",
    author_email="1922188869@qq.com",
    description="mtnlpmodel",
    install_requires=install_requires,
)
