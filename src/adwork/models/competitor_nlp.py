"""
NLP analysis of competitor ad copy.

Pipeline: TF-IDF → K-Means → PCA → auto-label clusters.
Uses only scikit-learn (already a dependency). No new packages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Keyword buckets for auto-labelling clusters ─────────────────────────

_LABEL_KEYWORDS: dict[str, list[str]] = {
    "Price & Deals": [
        "save", "sale", "deal", "discount", "price", "off", "clearance",
        "lowest", "cheap", "value", "flash", "coupon", "free shipping",
    ],
    "Features & Specs": [
        "hz", "ghz", "ram", "ssd", "display", "resolution", "processor",
        "battery", "sensor", "usb", "calibrated", "dpi", "refresh",
    ],
    "Lifestyle & Brand": [
        "experience", "feel", "designed", "crafted", "world", "silence",
        "immersive", "premium", "comfort", "create", "life", "moments",
    ],
    "Urgency & Scarcity": [
        "limited", "only", "last", "hurry", "gone", "stock", "ending",
        "remaining", "sold out", "reserve", "pre-order", "few left",
    ],
    "Comparison & Authority": [
        "rated", "best", "winner", "award", "review", "beats", "vs",
        "outperforms", "benchmark", "editor", "top pick", "proven",
    ],
}


class CompetitorAnalyzer:
    """TF-IDF + K-Means clustering of competitor ad copy."""

    def __init__(self, n_clusters: int = 5, max_features: int = 500):
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.vectorizer: TfidfVectorizer | None = None
        self.km: KMeans | None = None
        self.pca: PCA | None = None

    def analyze(self, ads_df: pd.DataFrame) -> dict:
        """
        Run full pipeline on a DataFrame with at least an 'ad_copy' column.

        Returns
        -------
        dict with keys:
            ads_df        – input df with 'cluster', 'x', 'y' columns added
            clusters      – list of {cluster_id, label, n_ads, top_terms}
            strategy_matrix – DataFrame (advertiser × cluster label) with counts
            feature_names – list of TF-IDF feature names
            explained_var – PCA explained variance ratio (2 components)
        """
        if "ad_copy" not in ads_df.columns:
            raise ValueError("ads_df must contain 'ad_copy' column")

        docs = ads_df["ad_copy"].fillna("").tolist()
        n_docs = len(docs)
        if n_docs < 3:
            raise ValueError(f"Need at least 3 ads to cluster, got {n_docs}")

        k = min(self.n_clusters, n_docs)

        # ── TF-IDF ──────────────────────────────────────────────────
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2 if n_docs > 10 else 1,
            max_df=0.85,
            sublinear_tf=True,
        )
        tfidf = self.vectorizer.fit_transform(docs)
        feature_names = self.vectorizer.get_feature_names_out().tolist()

        # ── K-Means ─────────────────────────────────────────────────
        self.km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = self.km.fit_predict(tfidf)

        # ── PCA → 2D ───────────────────────────────────────────────
        n_components = min(2, tfidf.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42)
        coords = self.pca.fit_transform(tfidf.toarray())

        ads_df = ads_df.copy()
        ads_df["cluster"] = labels
        ads_df["x"] = coords[:, 0]
        ads_df["y"] = coords[:, 1] if n_components == 2 else 0.0

        # ── Per-cluster summary ─────────────────────────────────────
        clusters = []
        used_labels: set[str] = set()
        for cid in range(k):
            mask = labels == cid
            top_terms = self._top_terms(tfidf[mask], feature_names, n=10)
            label = self._auto_label(top_terms, used_labels)
            used_labels.add(label)
            clusters.append({
                "cluster_id": cid,
                "label": label,
                "n_ads": int(mask.sum()),
                "top_terms": top_terms,
            })

        # ── Strategy matrix (advertiser × cluster) ──────────────────
        strategy_matrix = pd.DataFrame()
        if "advertiser_name" in ads_df.columns:
            label_map = {c["cluster_id"]: c["label"] for c in clusters}
            ads_df["cluster_label"] = ads_df["cluster"].map(label_map)
            strategy_matrix = (
                ads_df.groupby(["advertiser_name", "cluster_label"])
                .size()
                .unstack(fill_value=0)
            )

        logger.info(
            f"Clustered {n_docs} ads into {k} groups: "
            + ", ".join(f"{c['label']} ({c['n_ads']})" for c in clusters)
        )

        return {
            "ads_df": ads_df,
            "clusters": clusters,
            "strategy_matrix": strategy_matrix,
            "feature_names": feature_names,
            "explained_var": self.pca.explained_variance_ratio_.tolist(),
        }

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _top_terms(cluster_tfidf, feature_names: list[str], n: int = 10) -> list[str]:
        """Mean TF-IDF per term across cluster docs → top n."""
        mean_scores = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
        top_idx = mean_scores.argsort()[::-1][:n]
        return [feature_names[i] for i in top_idx]

    @staticmethod
    def _auto_label(top_terms: list[str], used: set[str]) -> str:
        """Match top terms against keyword buckets to pick a label."""
        best_label = "Other"
        best_score = 0

        for label, keywords in _LABEL_KEYWORDS.items():
            score = 0
            for kw in keywords:
                for term in top_terms:
                    if kw in term or term in kw:
                        score += 1
                        break
            if score > best_score and label not in used:
                best_score = score
                best_label = label

        # Fallback: if all labels used, append a suffix
        if best_label in used:
            best_label = f"{best_label} (2)"

        return best_label