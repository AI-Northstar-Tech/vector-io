#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys


def bump_version():
    # Execute the version bump script and return the new version number
    version_output = (
        subprocess.check_output(["python", "src/vdf_io/scripts/bump_version.py"])
        .decode("utf-8")
        .strip()
    )
    version_number = re.search(r"[0-9]+\.[0-9]+\.[0-9]+", version_output).group()
    print(f"New version: {version_number}")
    return version_number


def update_version_in_init(version_number):
    # Update the version number in __init__.py file
    with open("src/vdf_io/__init__.py", "r+") as f:
        content = f.read()
        content_new = re.sub(
            r"__version__ = .+", f'__version__ = "{version_number}"', content
        )
        f.seek(0)
        f.write(content_new)
        f.truncate()


def build():
    # move all files in dist/* to data/dist/* first. do not move the dist folder itself
    subprocess.run(["mkdir", "-p", "data/dist"], check=True)
    subprocess.run("mv dist/* data/dist/", shell=True, check=False)
    # delete build folder
    subprocess.run(["rm", "-rf", "build"], check=True)
    # Build the package
    subprocess.run(
        ["python", "setup.py", "sdist", "bdist_wheel", "--verbose"], check=True
    )


def upload():
    # Upload the package to PyPI
    subprocess.run(["twine", "upload", "dist/*", "--skip-existing"], check=True)


def install(version_number):
    # Install the wheel package
    wheel_file = f"dist/vdf_io-{version_number}-py3-none-any.whl"
    subprocess.run(["pip", "install", wheel_file], check=True)


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--bump",
        action=argparse.BooleanOptionalAction,
        help="Bump version number",
        default=True,
    )
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        help="Build package",
        default=True,
    )
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        help="Upload package to PyPI",
        default=True,
    )
    parser.add_argument(
        "--install",
        action=argparse.BooleanOptionalAction,
        help="Install the package",
        default=True,
    )
    print("Parsing args")
    args = parser.parse_args()
    print("Parsed args", args)

    version_number = None

    if args.bump:
        version_number = bump_version()

    if args.build:
        if not version_number:
            print(
                "Version number is required for building. Use --bump or manually specify the version."
            )
            sys.exit(1)
        update_version_in_init(version_number)
        build()

    if args.upload:
        upload()

    if args.install:
        if not version_number:
            print(
                "Version number is required for installation. Use --bump or manually specify the version."
            )
            sys.exit(1)
        install(version_number)


if __name__ == "__main__":
    print("Running build_and_upload_new_version.py")
    main()
