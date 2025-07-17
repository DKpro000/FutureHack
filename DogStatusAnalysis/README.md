# Introduction
This project is about the dog status inferencer by analyse dog condition through various information from the video using AIs. It helps human to understand the dog's feeling and situation so that they're able to treat symptoms before occur to avoid further deterioration.

# Setup
**Download dependencies**
``` bash
pip download -r requirements.txt
```

**Download dataset**

Download training dataset from [https://www.kaggle.com/datasets/danielshanbalico/dog-emotion]. Put it inside the train folder.
Run the training code block inside ``resnet18Training.ipynb`` to get resnet18 model weight. Drag it outside to the same path with ``DogEmotion.py``.

**Edit path**

***website/app.py:*** line 11 & 12

***website/routes.py:*** line 11

# Execution
1. Open terminal inside website folder and execute command below.
``` bash
python app.py
```

2. Enter ``http://127.0.0.1:5000`` on search engine and the webpage should appear:
![Alt text](images/mainpage.png)






