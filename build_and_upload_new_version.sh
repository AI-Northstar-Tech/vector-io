#!/bin/bash

version=$(python src/vdf_io/scripts/bump_version.py)
# extract the version number from version
version=$(echo $version | grep -o -E '[0-9]+\.[0-9]+\.[0-9]+')
echo "New version: $version"
python setup.py sdist bdist_wheel --verbose
twine upload dist/* --skip-existing
# find the wheel file in dist/ and install it
pip install dist/vdf_io-$version-py3-none-any.whl