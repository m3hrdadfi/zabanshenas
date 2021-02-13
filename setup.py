from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


install_requires = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = [p.strip() for p in f.read().splitlines()]


if __name__ == "__main__":
    setup(
        name="zabanshenas",
        version="0.1.0",
        description="zabanshenas - a transformer language detector",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Mehrdad Farahani",
        author_email="m3hrdadphi@gmail.com",
        url="https://github.com/m3hrdadfi/zabanshenas",
        license="Apache License",
        packages=find_packages(),
        include_package_data=True,
        install_requires=install_requires,
        platforms=["linux", "unix"],
        python_requires=">3.8.0",
    )
