from setuptools import find_packages, setup

setup(
    name="vdf_io",
    version="0.0.16",
    description="This library uses a universal format for vector datasets to easily export and import data from all vector databases.",
    long_description=open("README.rst").read(),
    license="Apache 2.0",
    author="Dhruv Anand",
    author_email="dhruv.anand@ainorthstartech.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "export_vdf=vdf_io.export_vdf_cli:main",
            "import_vdf=vdf_io.import_vdf_cli:main",
            "reembed=vdf_io.reembed:main",
            "consolidate_parquet=vdf_io.consolidate_parquet:main",
            "get_id_list=vdf_io.get_id_list:main",
            "push_to_hub_vdf=vdf_io.push_to_hub_vdf:main",
        ],
    },
    install_requires=open("requirements.txt").read().splitlines(),
)
