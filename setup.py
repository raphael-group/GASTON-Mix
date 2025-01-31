"""
"""
import setuptools

setuptools.setup(
    name='gastonmix',
    version='v0.0.1',
    python_requires='>=3.8',
    packages=['gaston'],
    package_dir={'': 'src'},
    author='Uthsav Chitra',
    author_email='uchitra@broadinstitute.edu',
    description='GASTON-Mix: a unified model of spatial gradients and domains using spatial mixture-of-experts',
    url='https://github.com/raphael-group/GASTON-Mix',
    install_requires=[
        'torch',
        'matplotlib',
        'numpy',
        'pandas',
        'seaborn',
        'scikit-learn',
        'tqdm',
        'scipy',
        'jupyterlab',
        'glmpca',
        'scanpy'
    ],
    include_package_data = True,
    package_data = {
        '' : ['*.txt']
        },
    license='BSD',
    platforms=["Linux", "MacOs", "Windows"],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        'spatial transcriptomics',
        'neural field',
        'MoE',
        'spatial gradients'],
    entry_points={'console_scripts': 'gastonmix=gastonmix.__main__:main'}
)
