from setuptools import setup, find_packages

setup(
    name='modamix',
    version='0.0.2',
    description='A Unified Framework for Correcting Batch Effects and Integrating Multi-Omics Data',
    author='miniymay',
    author_email='joungmin@vt.edu',
    url='https://github.com/cbi-bioinfo/MoDAmix',
    install_requires=['torch', 'pandas', 'numpy',],
    packages=find_packages(exclude=[]),
    keywords=['modamix'],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)