from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file):
    requirements = []

    with open(file) as f:
        requirements = f.readlines()
        requirements = [x.replace("\n", "") for x in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name = "Breast Cancer Prediction",
    version = "0.0.1",
    author="Abhijit Chakraborty",
    author_email="abhijityachak@gmail.com",
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)