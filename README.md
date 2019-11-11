main_project
==============================

Repositar projektu rieseneho v ramci predmetu neuronove siete na FIIT STU

Project Organization
------------

<br/><br/>

```
├── .gitignore          <- You usually don't want to push your data, logs or models to your repo
├── README.md
│
├── data
│   ├── processed       <- The data prepared to be fed into your model
│   └── raw             <- The original data you got
│
├── docker
│   ├── Dockerfile      <- Dockerfile to build the image for your project
│   └── setup           <- Additional files needed for Dockerfile
│
├── logs                <- Saved evaluation results
│
├── models              <- Saved models
│
├── notebooks           <- Jupyter notebooks for data analysis and model interaction.
│
└── src
    │
    ├── data            <- Scripts that load your data, e.g. tf.data pipeline
    │   └── load_data.py
    │
    └── models         
        ├── model.py    <- Your model definition
        ├── predict.py  <- Makes prediction with trained model on new data
        └── train.py    <- Training loop

```
<br/><br/>


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
