# PANNA
PANNA is a collection of image/video models with a unified interface.

## Installation
Panna support python<=3.12. Install from beta version 
```shell
pip install panna
```
or from the source.
```shell
pip install -U git+https://github.com/asahi417/panna.git@main
```


## Update PyPI

```shell
pip install twine
pip install setuptools wheel
python setup.py bdist_wheel 
twine upload dist/* 
```