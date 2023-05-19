from setuptools import setup, find_packages

setup(
    name='transparentmodel',
    version='1.0.0',
    description='A library for transparent models',
    author='Your Name',
    author_email='your@email.com',
    packages=find_packages(exclude=['tests']),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='transparentmodel wrapper library',
    install_requires=[
        'transformers>=4.0.0',
        # Add any other dependencies here
    ],
)
