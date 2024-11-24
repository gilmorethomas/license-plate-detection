from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="License Plate Detection with YOLO V8",
    version="0.1.0",
    author="Tommy Gilmore, Joey Wysocki",
    author_email="gilmoret@vt.edu, joeywysocki@vt.edu",
    maintainer="Tommy Gilmore, Joey Wysocki",
    maintainer_email="gilmoret@vt.edu, joeywysocki@vt.edu",
    description="License Plate Detection with YOLO V8",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gilmorethomas/license-plate-detection/documentation/index.html",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11.10',
)