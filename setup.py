from setuptools import setup, find_packages

with open('README.md', 'r', encoding="utf-8") as f:
    readme = f.read()

version = '1.0.0'
setup(
    name='panna',
    packages=find_packages(exclude=["hf_space", "tests"]),
    version=version,
    license='MIT',
    description='PANNA',
    url='https://github.com/asahi417',
    keywords=['machine-learning', 'computational-art'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "numpy<2.0.0",
        "torch==2.5.0",
        "datasets",
        "transformers<4.49.0",
        "diffusers>=0.19.0",
        "invisible_watermark",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "accelerate",
        "bitsandbytes",
        "DeepCache",
        "opencv-python"
    ],
    python_requires='>=3.8,<=3.12',
    entry_points={
        'console_scripts': [],
    }
)
