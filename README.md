# Identification-of-modulators-in-cancer-tumor-progression

**Author**: Dànae Canillas Sánchez

*Universitat Politècnica de Catalunya, UPC*


## Table of Contents

1. [Description](#description)
2. [Usage](#usage)
   1. [Virtual environment setup](#Virtual-environment-setup)
   2. [Project files](#Project-files)
   3. [Execution steps](#Execution-steps)
3. [License](#License)

## Description

## Usage

### Virtual environment setup

**1. Create a virtual environment on your local repository**

Check you have Python 3 version:

```py
$ python3 --version
Python 3.6.2
```

**2. Activate it to install packages and to execute scripts (ALWAYS!)**

```
$ source ./venv/bin/activate
(venv) @Username/:
```

You can deactivate it with:

```
$ deactivate
```



**3. Install `requirements.txt`**

With the environment activated:

```
(venv) $ pip install -r requirements.txt 
```



**4. Add new packages to the project, if needed**

```
(venv) $ pip install new_package_name
(venv) $ pip freeze -l > requirements.txt 
```



#### Install a Jupyter kernel

This will install a kernel inside the environment, to use to run in the Jupyter notebook there:

```
(venv) $ ipython kernel install --user --name=venv
```



### Project files
