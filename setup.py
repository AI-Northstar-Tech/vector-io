from setuptools import find_packages, setup

setup(
    name="vdf_io",
    version=open("VERSION.txt").read().strip(),
    description="A brief description of your package",
    long_description=open("README.rst").read(),
    license="Apache 2.0",
    author="Dhruv Anand",
    author_email="dhruv.anand@ainorthstartech.com",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "export_vdf = vdf_io.export_vdf:main",
            "import_vdf = vdf_io.import_vdf:main",
        ],
    },
    install_requires=open("requirements.txt").read().splitlines(),
)
