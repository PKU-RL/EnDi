import setuptools

setuptools.setup(
    name="endi_messenger",
    version="0.0.1",
    description="Implements EnDi and Multi-Agent Messenger environments.",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'gym==0.23.1',
        'numpy',
        'pygame'
    ],
)