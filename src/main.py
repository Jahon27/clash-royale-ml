import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import (
    load_card_list,
    load_battles,
    extract_decks,
    build_multi_hot,
    extract_trophy_diff,
)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"{name:35s} | acc={acc:.4f} | auc={auc:.4f}")
    return model, acc, auc

def prepare_base_data():
    card_df = load_card_list()
    battles_df = load_battles()

    num_cards = card_df["id"].nunique()

    p1_decks = extract_decks(battles_df, "p1")
    p2_decks = extract_decks(battles_df, "p2")

    p1_features = build_multi_hot(p1_decks, num_cards)
    p2_features = build_multi_hot(p2_decks, num_cards)

    trophy_diff = extract_trophy_diff(battles_df)
    y = battles_df["outcome"].values

    return p1_features, p2_features, trophy_diff, y

def fixed_split(X, y, train_size=600_000, test_size=100_000):
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))

    X = X[idx]
    y = y[idx]

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:train_size + test_size]
    y_test = y[train_size:train_size + test_size]

    return X_train, X_test, y_train, y_test

def experiment_decks_only(p1, p2, y):
    print("\n[Experiment] Decks only")

    X = np.hstack([p1, p2])
    X_train, X_test, y_train, y_test = fixed_split(X, y)

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    train_and_evaluate(model, X_train, y_train, X_test, y_test, "Decks only (L2)")

def experiment_decks_only(p1, p2, y):
    print("\n[Experiment] Decks only")

    X = np.hstack([p1, p2])
    X_train, X_test, y_train, y_test = fixed_split(X, y)

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    train_and_evaluate(model, X_train, y_train, X_test, y_test, "Decks only (L2)")

def experiment_trophy_only(trophy_diff, y):
    print("\n[Experiment] Trophy difference only")

    X = trophy_diff
    X_train, X_test, y_train, y_test = fixed_split(X, y)

    model = LogisticRegression(max_iter=1000)
    train_and_evaluate(model, X_train, y_train, X_test, y_test, "Trophy diff only")

def experiment_decks_plus_trophy(p1, p2, trophy_diff, y):
    print("\n[Experiment] Decks + trophy difference")

    X = np.hstack([p1, p2, trophy_diff])
    X_train, X_test, y_train, y_test = fixed_split(X, y)

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    train_and_evaluate(model, X_train, y_train, X_test, y_test, "Decks + trophies")

def experiment_symmetric_decks(p1, p2, y):
    print("\n[Experiment] Symmetric deck representation")

    X = p1 - p2
    X_train, X_test, y_train, y_test = fixed_split(X, y)

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    train_and_evaluate(model, X_train, y_train, X_test, y_test, "Symmetric decks")

def print_top_cards_l1(model, card_df, num_cards, top_k=10):
    coef = model.coef_[0]

    coef_p1 = coef[:num_cards]
    coef_p2 = coef[num_cards:]

    card_names = card_df.set_index("id")["card"]

    print("\nTop cards favoring Player 1 (p1 weights):")
    top_p1 = np.argsort(coef_p1)[-top_k:][::-1]
    for idx in top_p1:
        if coef_p1[idx] != 0:
            print(f"{card_names[idx]:25s} | weight={coef_p1[idx]:.4f}")

    print("\nTop cards favoring Player 2 (p2 weights):")
    top_p2 = np.argsort(coef_p2)[-top_k:][::-1]
    for idx in top_p2:
        if coef_p2[idx] != 0:
            print(f"{card_names[idx]:25s} | weight={coef_p2[idx]:.4f}")

def experiment_l1_l2(p1, p2, y, card_df):
    print("\n[Experiment] L1 vs L2 regularization")

    X = np.hstack([p1, p2])
    X_train, X_test, y_train, y_test = fixed_split(X, y)

    model_l2 = LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000, n_jobs=-1
    )

    model_l1 = LogisticRegression(
        penalty="l1", solver="saga", C=1.0, max_iter=1000, n_jobs=-1
    )

    train_and_evaluate(
        model_l2, X_train, y_train, X_test, y_test, "L2 Logistic Regression"
    )

    model_l1, _, _ = train_and_evaluate(
        model_l1, X_train, y_train, X_test, y_test, "L1 Logistic Regression"
    )

    nonzero = np.count_nonzero(model_l1.coef_)
    total = model_l1.coef_.size

    print(f"\nL1 sparsity: {nonzero} / {total} non-zero weights")

    num_cards = p1.shape[1]
    print_top_cards_l1(model_l1, card_df, num_cards)


def experiment_confidence(p1, p2, y):
    print("\n[Experiment] High-confidence predictions")

    X = np.hstack([p1, p2])
    X_train, X_test, y_train, y_test = fixed_split(X, y)

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    mask = (probs > 0.7) | (probs < 0.3)

    acc = accuracy_score(y_test[mask], model.predict(X_test)[mask])
    coverage = mask.mean()

    print(f"High-confidence accuracy: {acc:.4f}")
    print(f"Coverage: {coverage * 100:.2f}% of test samples")

def main():
    p1, p2, trophy_diff, y = prepare_base_data()
    card_df = load_card_list()

    experiment_decks_only(p1, p2, y)
    experiment_trophy_only(trophy_diff, y)
    experiment_decks_plus_trophy(p1, p2, trophy_diff, y)
    experiment_symmetric_decks(p1, p2, y)
    experiment_l1_l2(p1, p2, y, card_df)
    experiment_confidence(p1, p2, y)

if __name__ == "__main__":
    main()
