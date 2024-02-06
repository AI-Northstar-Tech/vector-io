#!/usr/bin/env python3

import argparse
import re
def bump_version(version_file, part="patch"):
    # Read the current version
    with open(version_file, "r") as file:
        content = file.read()

    # Find the version number using regex
    version_match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
    if version_match:
        version = version_match.group(1)
    else:
        raise ValueError("Version number not found in setup.py.")

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

    # Replace the old version with the new version in the content
    new_content = re.sub(r"(version\s*=\s*['\"])[^'\"]+(['\"])", fr"\g<1>{new_version}\g<2>", content)

    # Write the new content back to the file
    with open(version_file, "w") as file:
        file.write(new_content)

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
