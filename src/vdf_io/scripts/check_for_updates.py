import requests
import pkg_resources as pkg

import vdf_io


def check_for_updates():
    package_name = "vdf-io"
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    if response.status_code == 200:
        latest_version = response.json()["info"]["version"]
        current_version = pkg.parse_version(vdf_io.__version__)
        latest_version = pkg.parse_version(latest_version)
        if current_version.release[:2] < latest_version.release[:2]:
            print(
                f"Current version: {vdf_io.__version__}. Update available: {latest_version}. Run `pip install --upgrade {package_name}` to update."
            )


def main():
    check_for_updates()


if __name__ == "__main__":
    main()
