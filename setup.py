from setuptools import setup, find_packages

setup(
    name='ae',                              # Choose a unique name for your package
    version='0.1.0',                        # Start with a small version number
    packages=find_packages(where='src'),    # This automatically finds packages in the src directory
    package_dir={'': 'src'},                # Tells setuptools that packages are under src directory
)
