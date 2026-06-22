"""ML agent — learns to predict next-day direction from features.

It follows the same Strategy interface as the moving-average agent, so the
backtester treats it identically. The model is fit ONCE on the training data
passed to the constructor; `generate_signals` then predicts on whatever prices
it is given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from chf.features import make_features, make_target
from chf.strategies.base import Strategy


class MLAgent(Strategy):
    def __init__(
        self,
        train_prices: pd.DataFrame,
        model: RandomForestClassifier | None = None,
        threshold: float = 0.5,
    ) -> None:
        # Shallow trees + large leaves -> resists overfitting on noisy markets.
        self.model = model or RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=50,
            random_state=0,
            n_jobs=-1,
        )
        self.threshold = threshold
        self.name = "ML (RandomForest)"
        self.feature_names: list[str] = []
        self._fit(train_prices)

    def _fit(self, prices: pd.DataFrame) -> None:
        features = make_features(prices)
        target = make_target(prices)
        data = features.join(target.rename("target")).dropna()
        self.feature_names = list(features.columns)
        self.model.fit(data[self.feature_names], data["target"])

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        features = make_features(prices)
        valid = features.dropna()
        proba = pd.Series(0.0, index=prices.index)
        if len(valid):
            classes = list(self.model.classes_)
            if 1 in classes:
                up_col = classes.index(1)
                p = self.model.predict_proba(valid[self.feature_names])[:, up_col]
                proba.loc[valid.index] = p
        # Long when the model is confident the next day is up.
        return (proba > self.threshold).astype(float)
