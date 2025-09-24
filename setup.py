from setuptools import setup, find_packages

setup(
    name='stlenc',  
    version='0.1',  
    description='STL Encoder (Signal Temporal Logic)', 
    author='Sara Candussio',  
    author_email='sara.candussio@phd.units.it',  
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  
    install_requires=[
        'torch>=1.8.0',  
        'numpy>=1.20.0',
        'pandas>=1.1.0',
        'matplotlib>=3.0',  
        'scikit-learn>=0.24.0',  
        'networkx>=2.5',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',  
            'flake8>=3.8', 
        ],
    },
    include_package_data=True,  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',  
)
