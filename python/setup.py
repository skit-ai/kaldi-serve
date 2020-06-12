import os
import re
import sys

from setuptools import setup


def find_library(lib_regex, path="kaldiserve", required=True):
    pattern = re.compile(lib_regex)

    files = os.listdir(path)
    for f in files:
        if pattern.match(f):
            return f
    if required:
        raise FileNotFoundError(lib_regex)
    return None


setup(
    name='kaldiserve',
    version='1.0.0',
    author='Vernacular.ai team',
    author_email='hello@vernacular.ai',
    description='A plug-and-play abstraction over Kaldi ASR toolkit.',
    long_description='',
    packages=["kaldiserve"],
    package_dir={"kaldiserve": "kaldiserve"},
    include_package_data=True,
    package_data={
        "kaldiserve": [find_library(r"kaldiserve_pybind.*\.so")]
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="asr speech-recognition kaldi grpc-server",
    license="Apache",
    url="https://github.com/Vernacular-ai/kaldi-serve",
    project_urls={
        'Documentation': 'https://github.com/Vernacular-ai/kaldi-serve',
        'Source code': 'https://github.com/Vernacular-ai/kaldi-serve',
        'Issues': 'https://github.com/Vernacular-ai/kaldi-serve/issues',
    },
    zip_safe=False,
)