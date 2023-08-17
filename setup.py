
from setuptools import setup, find_packages

setup(
    name="stochastiqML",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'torch',
        'torchsde',
        # ...
    ],
    extras_require={
        'dev': [
            # Development dependencies
            'pytest',
            # ...
        ]
    },
    # other metadata like author, license, description, etc.
)
