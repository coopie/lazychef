# lazychef
Create lazily evaluated data sources (sauces) for scalable machine learning experiments. Currently only python 3 compatible.

## Why

Running machine learning experiments can become very annoying when you don't design them properly. Changing anything can require re-running minutes (if not hours) of scripts to re-process data for training. This can stifle experimenting, and stop research being fun.

We want to be able to change data representation as easily as we model design.

Lazychef tries to (one more sentence)


## Example


TODOs:
examples working with scipy, tensorflow and keras


### Distribution
make sure to remove older versions of the package
```
python setup.py sdist bdist_wheel
twine upload dist/*
```
