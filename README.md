# intro_to_LMMs

## Introduction

This repository has some teaching resources and code for a course on introduction to large language models (LMMs). 


## Resources

* very good visual explanation of LLMs and transformers

  https://ig.ft.com/generative-ai/

* Introduction to LLMs theory

  https://docs.science.ai.cam.ac.uk/large-language-models/Introduction/Introduction/

* Andrej Karpathy build GPT-2 ground up

https://www.youtube.com/watch?v=kCc8FmEb1nY

* Vizuara videos

  https://www.youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu

  https://www.vizuaranewsletter.com/p/9e1

* 3blue1brown videos

VERY GOOD playlist on deep learning, LLMs and transformers

https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

Embedding video

Attention video

https://www.youtube.com/watch?v=wjZofJX0v4M

  Transformer video

https://www.youtube.com/watch?v=eMlx5fFNoYc&vl=en

https://www.3blue1brown.com/lessons/gpt


* Attention	

Introduction to attention mechanism (VERY GOOD)

https://www.youtube.com/watch?v=XN7sevVxyUM

Animation

https://jalammar.github.io/illustrated-transformer/

https://en.wikipedia.org/wiki/Attention_(machine_learning)#/media/File:Attention-qkv.png

https://en.wikipedia.org/wiki/Attention_(machine_learning)

* Next token prediction

  https://medium.com/@akash.kesrwani99/understanding-next-token-prediction-concept-to-code-1st-part-7054dabda347

* LangChain and huggingface open source model example

  https://python.langchain.com/docs/integrations/chat/huggingface/


## Installation

```R
pip install -r requirements-handson.txt

mkdir -p /home/codespace/.local/lib/python3.12/site-packages/google/colab

```

Add a new file called `.env` and type in the following:

```
OPENAI_API_KEY = "<yourapikeywhichisprivate>"
```

Create a .gitignore and add the following

```
.env
```


You can execute the following notebooks in Github codespaces or Google Colab.


## Code

https://github.com/neelsoumya/hands-on-llms

https://github.com/acceleratescience/large-language-models

https://github.com/acceleratescience/hands-on-llms

https://docs.science.ai.cam.ac.uk/hands-on-llms/setting-up/codespaces/

`L2_NLP_transformers.ipynb`: Simple code to call a facebook open-source model

https://github.com/neelsoumya/intro_to_LMMs/blob/main/L2_NLP_transformers.ipynb

Transformers from scratch

`[1_1]_Transformer_from_Scratch_(exercises).ipynb`

https://github.com/neelsoumya/intro_to_LMMs/blob/main/%5B1_1%5D_Transformer_from_Scratch_(exercises).ipynb

`text_classification_with_transformer.ipynb`: Multi-head attention transformers in Keras

https://github.com/neelsoumya/intro_to_LMMs/blob/main/text_classification_with_transformer.ipynb

`tiktoken_demo.ipynb`: Code to explain tokenizer

https://github.com/neelsoumya/intro_to_LMMs/blob/main/tiktoken_demo.ipynb

`softmax_practical.ipynb`: Practical to explain the softmax function.

https://github.com/neelsoumya/intro_to_LMMs/blob/main/softmax_practical.ipynb

`Lesson_3-selfattention.ipynb`: Coding self-attention in PyTorch

https://github.com/neelsoumya/intro_to_LMMs/blob/main/Lesson_3-selfattention.ipynb

`Situational_Awareness_LLMs_LLaMA.ipynb`: Open source model to test if LLM has situational awareness

https://github.com/neelsoumya/intro_to_LMMs/blob/main/Situational_Awareness_LLMs_LLaMA.ipynb

`Situational_Awareness_LLMs.ipynb`: Closed source model to test if LLM has situational awareness

https://github.com/neelsoumya/intro_to_LMMs/blob/main/Situational_Awareness_LLMs.ipynb


`arc_solver.ipynb`: IPython notebook to solve ARC puzzles and H.Dudeney puzzles

https://github.com/neelsoumya/intro_to_LMMs/blob/main/arc_solver.ipynb


`fine_tune_llm.ipynb`: IPython notebook that shows how to finetune a LLM

https://github.com/neelsoumya/intro_to_LMMs/blob/main/fine_tune_llm.ipynb


`text_translation_summarization.ipynb`: IPython notebook to translate text and summarize text using open-source models.

https://github.com/neelsoumya/intro_to_LMMs/blob/main/text_translation_summarization.ipynb

`L2_NLP_transformers.ipynb`: Open source LLM for probing superintelligence (model organism of misalignment)

https://github.com/neelsoumya/intro_to_LMMs/blob/main/L2_NLP_transformers.ipynb

`02_open_ai.ipynb`: Code to call OpenAI API

https://github.com/neelsoumya/hands-on-llms/blob/main/Notebooks/02_open_ai.ipynb

https://github.com/neelsoumya/intro_to_LMMs/blob/main/02_open_ai.ipynb

`agentic_workflow_llm_opensource.ipynb`: Open source model for agentic workflow

https://github.com/neelsoumya/intro_to_LMMs/blob/main/agentic_workflow_llm_opensource.ipynb


`introduction_to_stablediffusion.ipynb`: Stable diffusion text to image generation using open-source HuggingFace model

https://github.com/neelsoumya/intro_to_LMMs/blob/main/introduction_to_stablediffusion.ipynb

`Lesson3.ipynb`: Coding agents using `smolagents`

https://github.com/neelsoumya/intro_to_LMMs/tree/main/agents/smolagents/L3

https://github.com/neelsoumya/intro_to_LMMs/blob/main/agents/smolagents/L3/Lesson3.ipynb

`smolagents_websearch.ipynb`: Websearch agent using `smolagents`

https://github.com/neelsoumya/intro_to_LMMs/blob/main/smolagents_websearch.ipynb


## Project and hackathon

Some other projects and hackathons using LLMs are here:

* Code to perform CFD and solve ARC tasks

https://github.com/neelsoumya/CFD_LLM_Accelerate24

https://github.com/neelsoumya/hands-on-llms/blob/main/Notebooks/arc_solver.ipynb


* Code to create a healthcare AI chatbot

https://github.com/neelsoumya/LLM-Handon

* Code to use science-fiction to re-envision AI using LLMs

https://github.com/neelsoumya/science_fiction_LLM

https://github.com/neelsoumya/science_fiction_AI_LLM

* Code for open source LLM for probing superintelligence (model organism of misalignment).
    
https://github.com/neelsoumya/intro_to_LMMs/blob/main/L2_NLP_transformers.ipynb

* Coding agents using `smolagents`

https://github.com/neelsoumya/intro_to_LMMs/tree/main/agents/smolagents/L3

https://github.com/neelsoumya/intro_to_LMMs/blob/main/agents/smolagents/L3/Lesson3.ipynb



## User interfaces

* Streamlit

* https://docs.science.ai.cam.ac.uk/large-language-models/streamlit/



## Acknowledgement

Accelerate Science and Ryan Daniels

https://science.ai.cam.ac.uk/

https://docs.science.ai.cam.ac.uk/training/#accelerate-workshops

https://github.com/acceleratescience/diffusion-models

https://docs.science.ai.cam.ac.uk/large-language-models/

https://docs.science.ai.cam.ac.uk/diffusion-models/Introduction/Introduction/

https://science.ai.cam.ac.uk/team/ryan-daniels

https://docs.science.ai.cam.ac.uk/large-language-models/Introduction/Introduction/


## Making this a package

This is also a package. So in your terminal you can type

```R
pip install -e .
```

and in a Python script you can type the following

```py

from intro_to_LMMs import greet

print(greet())

```

or run

`test_pythonpackage.py`

or run the following in the terminal

```py
python -m unittest discover -s tests
```

Steps to upload this to `PyPI`:

 ```py
 pip install build twine
 ```
 
```py
 python -m build
```

Create a `.pypirc` with your credentials. Please have this in your `.gitignore` file

The format of a `.pypirc` file is:

```py
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <your-pypi-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <your-testpypi-token>
```

Upload to TestPyPI:


```py
 twine upload --repository testpypi dist/* 
```

Upload to PyPI


```py
pip install --index-url https://test.pypi.org/simple/intro-to-LMMs
```

Verify Installation


```py
pip install intro-to-LMMs
```

## Running Docker

Create a `Dockerfile`

https://github.com/neelsoumya/intro_to_LMMs/blob/main/Dockerfile

Build the docker image

```R
docker build -t my-jupyter-image .
```

Run the docker container

```R
docker run -p 8888:8888 -v $(pwd):/app my-jupyter-image
```

Access Jupyter Notebook by visiting 

`http://localhost:8888`


## Contact

Soumya Banerjee

sb2333@cam.ac.uk

