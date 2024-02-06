#!/usr/bin/env python3

import argparse


def bump_version(version_file, part="patch"):
    # Read the current version
    with open(version_file, "r") as file:
        version = file.read().strip()

    # Split version into components
    major, minor, patch = map(int, version.split("."))

    # Increment the appropriate part of the version
    if part == "major":
        major += 1
        minor = 0  # Reset minor and patch numbers when major is incremented
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0  # Reset patch number when minor is incremented
    elif part == "patch":
        patch += 1
    else:
        raise ValueError("Invalid part specified. Use 'major', 'minor', or 'patch'.")

    # Combine the parts back into a version string
    new_version = f"{major}.{minor}.{patch}"

    # Write the new version back to the file
    with open(version_file, "w") as file:
        file.write(new_version)

    return new_version


def main():
    parser = argparse.ArgumentParser(description="Bump version number in setup.py.")
    parser.add_argument(
        "version_file",
        nargs="?",
        default="setup.py",
        help="Path to the version file (default: setup.py)",
    )
    parser.add_argument(
        "--part",
        choices=["major", "minor", "patch"],
        default="patch",
        help="Part of version to bump (default: patch)",
    )
    args = parser.parse_args()

    new_version = bump_version(args.version_file, args.part)
    print(f"Updated version: {new_version}")


if __name__ == "__main__":
    main()
