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

# Example
**Example 1**

Video found from YouTube [https://www.youtube.com/watch?v=01le4Ln8da0].
https://github.com/user-attachments/assets/fe70cddc-97e0-4491-b755-7a005c8d35cc

Output
<img width="1627" height="366" alt="image" src="https://github.com/user-attachments/assets/c53eefe6-172a-473d-99c5-d0572a320529" />
<img width="1623" height="484" alt="image" src="https://github.com/user-attachments/assets/815a9f00-28d1-4a27-abf9-39cb7115c73d" />
<img width="1628" height="156" alt="image" src="https://github.com/user-attachments/assets/fefa1d7a-dd2a-4c0e-ab8b-be2a00c8610c" />
<img width="1622" height="436" alt="image" src="https://github.com/user-attachments/assets/4ab008d3-8980-4c96-9d61-f7052e614458" />

**Example 2**

The video was shot by casually taking photos of dogs encountered while eating with family on the roadside to simulate the real scene.
https://github.com/user-attachments/assets/b46aa029-67bc-49f5-969a-fc221b83f149

Output
<img width="1624" height="375" alt="image" src="https://github.com/user-attachments/assets/55acab0a-155a-44dc-bdfa-a542f3097aef" />
<img width="1626" height="482" alt="image" src="https://github.com/user-attachments/assets/ed9ac875-7297-4a44-85a2-6630fc06de5c" />
<img width="1623" height="147" alt="image" src="https://github.com/user-attachments/assets/a35e6ecb-69ab-4cad-83b6-e75936690eac" />
<img width="1623" height="437" alt="image" src="https://github.com/user-attachments/assets/59e56c84-ba4f-4cb9-b271-ba579b7e99d9" />

**Example 3**

The video was taken from a husky club I passed by while shopping. This is also used to simulate a real scene.

https://github.com/user-attachments/assets/d21c9511-0ae5-489a-b53f-b22628da7b88

Output
<img width="1626" height="370" alt="image" src="https://github.com/user-attachments/assets/f34fcc39-1e2a-405f-bb03-a43980c1e0d0" />
<img width="1624" height="483" alt="image" src="https://github.com/user-attachments/assets/092a4b63-58cb-4e89-bcf8-9c1635b87052" />
<img width="1623" height="153" alt="image" src="https://github.com/user-attachments/assets/21bbfbab-2bf8-4e45-96bf-f65062b5e6d8" />
<img width="1625" height="381" alt="image" src="https://github.com/user-attachments/assets/ad488a6a-f4fc-4783-8cda-8a54ff08a250" />






















