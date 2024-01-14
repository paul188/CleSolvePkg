from setuptools import find_packages,setup

with open("app/README.md") as f:
    long_description = f.read()

setup(name="clesolvepkg",
      version = "0.0.10",
      description= "Simulation package for the Chemical Lagrange Equation",
      package_dir={"":"app"},
      packages=find_packages(where="app"),
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/paul188/CleSolvePkg/",
      author="Paul Johannssen",
      author_email="pauljoh@gmx.de",
      license="MIT",
      classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
      ],
      install_requires= ["sdeint >= 0.3.0","numpy >= 1.24.2", "scipy >= 1.11.1"],
      extras_require={
          "dev": ["pytest>=7.0", "twine >=4.0.2"]
      },
      python_requires=">=3.10"
      )