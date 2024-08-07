from deepepochs import __version__
import setuptools

with open("README.md", 'r', encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepepochs",
    version=__version__,
    author="hitlic",
    author_email="liuchen.lic@gmail.com",
    license='MIT',
    description="An easy-to-use tool for training Pytorch deep learning models",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/hitlic/deepepochs",
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='',
    packages=setuptools.find_packages(exclude=['__pycache__', '__pycache__/*']),
    py_modules=[],  # any single-file Python modules that aren’t part of a package
    install_requires=['torch>=2.0.0', 'pyyaml>=5.4', 'tqdm>=4.62', 'numpy>1.21', 'matplotlib>=3.6', 'tensorboard>=2.12', 'scikit-learn>=0.22', 'pandas>=1.3', 'accelerate>=0.24'],
    python_requires='>=3.8'
)

