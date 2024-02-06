from setuptools import find_packages, setup

setup(
    name="vdf_io",
    version="0.0.32",
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
            "reembed_vdf=vdf_io.scripts.reembed:main",
            "consolidate_parquet_vdf=vdf_io.scripts.consolidate_parquet:main",
            "get_id_list_vdf=vdf_io.scripts.get_id_list:main",
            "push_to_hub_vdf=vdf_io.scripts.push_to_hub_vdf:main",
            "bump_version_vdf=vdf_io.scripts.bump_version:main",
        ],
    },
    install_requires=open("requirements.txt").read().splitlines(),
)
