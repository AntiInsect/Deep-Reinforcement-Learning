# week-one-report

* what I have done this is basically setting the environment and learning the basic knowledge for deep RL

## Environment Setting (On Mac)

* set up Conda
  * Conda is another great package manager for Python besides pip
  * Using Conda can help me better handle scientific usage of Python
  * In the openAI site, there is some unclear statements about Conda
  * `git clone https://github.com/openai/spinningup.git` `cd spinningup` `pip install -e .`
    * there is a problem in the last line
    * as we know, when we start a Python project it is always recommended that we use a virtual environment : `conda create -n spinningup python=python_version`
    * But we should know that in the Conda environment, pip will still install packages globally. So, we should modified the last command above : `conda install pip` `/anaconda3/envs/spinningup/bin/pip install -e .`
    * this line will install pip package into Conda virtual environment  
* set up Pycharm
  * Pycharm is a very powerful IDE for Python
  * with Conda we have just talked about, Pycharm can manage your Python project very well
* set up Tensorflow and Keras
  * these two libraries are basic for machine learning
  * on my MacPro, the setting of Tensorflow is quite tricky
    * if I just install the package through pip or Conda, it will not work well due to poor supporting for SSE and AVX (openMP things also)
    * So I have to build the package from the source on my computer which leads to another question — we must use another package called Bazel a package containing bugs for now
    * at last, I found a repo on GitHub which contains already compiled Mac-specific Tensorflow
    * Problem solved

## openAI Learning

* Learning using openAI, tensorflow and Keras through a CarPole agent
  * see ::carPole.py:: for more info

* deep RL Basic
  * Basically going through the openAI website
  * The hand written notes in the ::appendix::  
