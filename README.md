# DeepBook: a French book recommendation engine

## Install


- Instructions for **pyenv** version manager
```bash
# Ensure correct Python version is installed and install it via pyenv if missing
pyenv versions | grep -q 3.10.6 || pyenv install 3.10.6
# Create and define project virtual environment
pyenv virtualenv 3.10.6 DeepBook && pyenv local DeepBook
# Install project dependencies
pip install -e .
```

This may take a few minutes as Deep Learning specific packages take some time to install.


## Contribute

1. Create a branch from master for each new feature
```bash
git checkout -b <branch-name>
```
2. Make sure to modify the `requirements.txt` accordingly for each new package
3. Make a pull request to master explaining your changes and wait for fellow contributors to accept it

---

`*` Update to the latest package versions:

```bash
pip install -e .
```
