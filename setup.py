from setuptools import find_packages, setup

setup(
    name="vdf_io",
    version="0.1.241",
    description="This library uses a universal format for vector datasets to easily export and import data from all vector databases.",
    long_description="Check out the README for more information: https://github.com/AI-Northstar-Tech/vector-io/blob/main/README.md",
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
            "check_for_updates_vdf=vdf_io.scripts.check_for_updates:main",
            "count_vdf=vdf_io.scripts.count_rows:main",
        ],
    },
    install_requires=open("requirements.txt").read().splitlines(),
)
