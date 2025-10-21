from setuptools import setup, find_packages

setup(
    name="dl_client",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "click>=8.0.0",
        "tqdm>=4.60.0",
        "pandas>=1.0.0",
    ],
    entry_points={
        'console_scripts': [
            'dl-client=dl_client.cli:main',
        ],
    },
    author="IFAB",
    author_email="raimondo.reggio@ifabfoundation.org",
    description="Client library for the AIND Datalake API",
    keywords="datalake, api, client",
    python_requires=">=3.6",
)
