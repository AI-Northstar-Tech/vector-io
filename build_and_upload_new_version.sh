#!/bin/bash

version=$(python src/vdf_io/scripts/bump_version.py)
# extract the version number from version
version_number=$(echo "$version" | grep -o -E '[0-9]+\.[0-9]+\.[0-9]+')

# # Use the extracted version number in the sed command.
# # Adjust the sed command according to your operating system. This is the macOS version.
# # For GNU/Linux or other Unix-like, you might not need the '' after -i.
# sed -i '' "s/__version__ = .*/__version__ = '$version_number'/" src/vdf_io/__init__.py
echo "New version: $version_number"
python setup.py sdist bdist_wheel --verbose
twine upload dist/* --skip-existing
# find the wheel file in dist/ and install it
pip install dist/vdf_io-$version_number-py3-none-any.whl