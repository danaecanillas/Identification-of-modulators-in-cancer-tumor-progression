# Identification of modulators in cancer tumor progression

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

The main objective of this research is to provide information on the relevance of different pathways or genetic expressions for the development of breast cancer, for the relapse of patients suffering from it and the incidence of death. 

- Identify correlations between the most relevant pathways or genetic expressions when influencing the progression of breast cancer tumor.
- Provide a predictor model of relapse time and / or death time count from the first diagnosis.
- Identify the most relevant pathways in each breast cancer subtypes. 

## Usage

### Virtual environment setup

Check you have Python 3 version:

```py
$ python3 --version
Python 3.6.2
```

**1. Activate it to install packages and to execute scripts (ALWAYS!)**

```
$ source ./venv/bin/activate
(venv) @Username/:
```

You can deactivate it with:

```
$ deactivate
```



**2. Install `requirements.txt`**

With the environment activated:

```
(venv) $ pip install -r requirements.txt 
```



**3. Add new packages to the project, if needed**

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
