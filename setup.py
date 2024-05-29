from setuptools import setup, find_packages

setup(
    name='icpReconstructor',  
    version='{{VERSION_PLACEHOLDER}}', 
    author='Matthias K. Hoffmann', 
    author_email='matthias.hoffmann@uni-saarland.de', 
    description='A package providing functionality to estimate the shape of continuum robots using the Iterative Closest Point algorithm.',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',  
    url='https://github.com/MKHoffmann/icpReconstructor', 
    packages=find_packages(),  # Automatically find all packages and subpackages
    classifiers=[
        'Development Status :: 5 - Production/Stable',  
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', 
        'Programming Language :: Python :: 3',  
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    license='GNU General Public License v3 (GPLv3)', 
    install_requires=[
        'casadi >= 3.6.1',
        'numpy >= 1.11.0',
        'torchdiffeq >= 0.2.2',
        'torch >= 2.0.0',      
        'scikit-learn >= 1.2.0',
        'scikit-image >= 0.20.0',
        'tqdm >= 4.64.1'
    ],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
    python_requires='>=3.8',
)
