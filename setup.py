from pathlib import Path
from setuptools import find_packages
from setuptools import setup

content = Path("requirements.txt").read_text().splitlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='deepbook',
      version="0.0.1",
      description="DeepBook project",
      license="MIT",
      author="Roxane Laigle, Adrien Aixala, Niklas Friese, Frédéric Saudemont",
      author_email="roxanelaigle.contact@gmail.com, adrien.aixala@gmail.com, niklasole.friese@gmail.com, fs1510@gmail.com",
      url="https://github.com/Roxanelaigle/DeepBook",
      install_requires=requirements,
      packages=find_packages()
      )
