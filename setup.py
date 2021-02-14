import os
import glob
from setuptools import find_packages, setup


def get_file_path(name):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), name))


def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


with open(get_file_path('zabanshenas/version.py')) as f:
    exec(f.read())


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


requirements = parse_requirements(get_file_path('requirements.txt'))


if __name__ == "__main__":
    setup(
        name="zabanshenas",
        version=__version__,
        description="zabanshenas - language detection",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Mehrdad Farahani",
        author_email="m3hrdadphi@gmail.com",
        url="https://github.com/m3hrdadfi/zabanshenas",
        license="Apache License",
        packages=find_packages(),
        package_data={
            "": glob.glob(get_file_path("zabanshenas/models/**/*"), recursive=True)
        },
        include_package_data=True,
        install_requires=requirements,
        platforms=["linux", "unix"],
        python_requires=">3.7.0",
    )
