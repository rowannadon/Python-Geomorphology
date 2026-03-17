"""Setup script for terrain generator package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="terrain-generator",
    version="1.0.0",
    author="Your Name",
    description="3D terrain generator with river networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
        "PyQt5>=5.15.0",
        "PyOpenGL>=3.1.0",
        "scikit-image>=0.18.0",
    ],
    extras_require={
        "dark_theme": ["pyqtdarktheme>=2.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-qt>=4.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ]
    },
    entry_points={
        "console_scripts": [
            "terrain-generator=terrain_generator.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)