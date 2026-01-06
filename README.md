# Clash Royale Match Outcome Prediction

This project investigates whether deck composition in the game *Clash Royale* contains a measurable signal that can be used to predict match outcomes using machine learning.

Rather than aiming for perfect prediction accuracy, the goal is to analyze the limits of predictability in competitive multiplayer games and to study how much information can be extracted from static pre-game features such as deck selection and player trophies.

The project was developed as part of the course *Machine Learning (Fall 2025)* at Comenius University in Bratislava.

---

## Project Overview

Clash Royale is a real-time strategy game in which players compete in one-versus-one matches using decks composed of exactly eight cards.  
Within the player community, it is commonly believed that certain decks systematically counter others.  
This project evaluates this assumption using supervised machine learning techniques on a large dataset of ranked matches.

The task is formulated as a binary classification problem:
- **Input:** Two player decks (card identifiers), optionally augmented with trophy difference
- **Output:** Probability that Player 1 wins the match

---

## Project Structure
```clash-royale-ml/
│
├── src/
│ ├── main.py # Runs all experiments
│ ├── data_loader.py # Data loading and feature construction
│
├── data/
│ ├── data_ord.csv # Match data
│ ├── cardlist.csv # Card ID to name mapping
│
├── requirements.txt
└── README.md
```


---

## Dataset

The project uses a publicly available dataset of ranked Clash Royale 1v1 matches obtained from Kaggle:

https://www.kaggle.com/datasets/nonrice/clash-royale-battles-upper-ladder-december-2021?select=data_ord.csv

The dataset contains approximately 750,000 matches and includes:
- Deck compositions for both players
- Player trophy counts
- Match outcomes

Card identifiers are mapped to card names using `cardlist.csv`.

---

## Methods and Experiments

The following experiments are implemented in this project:

- Deck composition only
- Trophy difference only
- Deck composition combined with trophy difference
- Symmetric deck representation (Player 1 deck minus Player 2 deck)
- Logistic regression with L1 and L2 regularization
- Analysis of feature sparsity and influential cards
- Confidence-based evaluation of predictions

All experiments are executed from `main.py`, and results are printed directly to the console.

---

## Installation

``` pip install -r requirements.txt ```

``` python src/main.py ```

