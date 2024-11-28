from setuptools import setup, find_packages

setup(
    name='anarcii',
    version='0.2',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    entry_points={
        "console_scripts": [
            "anarcii=anarcii.cli:main",
        ],
    },
    description='A sequence numbering tool called ANARCII that ',
    author='AlexGW',
    author_email='GreenshieldsWatsonAL@gmail.com',
    package_data={
        'anarcii': ['models/**/*.pt', 'models/**/*.json'],
    },
    install_requires=[
        'numpy',
        'torch'
    ],
)
