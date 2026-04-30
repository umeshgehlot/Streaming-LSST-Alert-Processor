"""Setup configuration for Streaming LSST Alert Processor."""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(
    name='streaming-lsst',
    version='0.1.0',
    description='Lightweight streaming-first architecture for LSST alert processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='LSST Streaming Team',
    author_email='example@example.com',
    url='https://github.com/example/streaming-lsst',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'cuda': ['torch-cuda>=2.0.0'],
        'dev': ['pytest>=6.0', 'black>=21.0', 'flake8>=3.9'],
    },
    entry_points={
        'console_scripts': [
            'lsst-benchmark=streaming_lsst.benchmarks.run_benchmarks:main',
            'lsst-examples=streaming_lsst.examples:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='lsst astronomy streaming transformers gnn anomaly-detection',
)
