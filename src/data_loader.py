import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_card_list():
    """
    Loads cardlist.csv.
    """
    path = DATA_DIR / "cardlist.csv"
    return pd.read_csv(path)


def load_battles():
    """
    Loads data_ord.csv.
    """
    path = DATA_DIR / "data_ord.csv"
    return pd.read_csv(path)

def extract_decks(df, player_prefix):
    """
    Extracts deck card IDs for a given player.

    player_prefix: 'p1' or 'p2'
    Returns: numpy array of shape (n_samples, 8)
    """
    card_cols = [f"{player_prefix}card{i}" for i in range(1, 9)]
    return df[card_cols].values


def build_multi_hot(decks, num_cards):
    """
    Converts deck card IDs into multi-hot encoded vectors.

    decks: numpy array (n_samples, 8)
    num_cards: total number of unique cards
    Returns: numpy array (n_samples, num_cards)
    """
    n_samples = decks.shape[0]
    multi_hot = np.zeros((n_samples, num_cards), dtype=int)

    for i in range(n_samples):
        for card_id in decks[i]:
            multi_hot[i, card_id] = 1

    return multi_hot

def extract_trophy_diff(df):
    """
    Computes trophy difference (p1 - p2).
    Returns: numpy array of shape (n_samples, 1)
    """
    diff = df["p1trophies"].values - df["p2trophies"].values
    return diff.reshape(-1, 1)
