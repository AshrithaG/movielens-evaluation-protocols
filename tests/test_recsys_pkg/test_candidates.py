import numpy as np
import pytest

from recsys.candidates import ItemKNN, Popularity, TwoTower, top_k
from recsys.metrics import (
    catalog_coverage,
    expected_calibration_error,
    hit_at_k,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    recall_at_k,
)


def test_top_k_excludes_and_breaks_ties_by_id():
    assert top_k(np.array([1.0, 1.0, 0.5, 1.0]), 3, exclude=np.array([0])) == [1, 3, 2]


def test_top_k_is_a_prefix_across_k():
    s = np.array([0.3, 0.3, 0.9, 0.1, 0.3])
    assert top_k(s, 2) == top_k(s, 5)[:2]


def test_popularity_counts_distinct_users(toy):
    frame, _, n_users, n_items = toy
    doubled = np.concatenate([frame["user"], frame["user"]])
    items = np.concatenate([frame["item"], frame["item"]])
    pop = Popularity().fit(doubled, items, n_items)
    single = Popularity().fit(frame["user"].to_numpy(), frame["item"].to_numpy(), n_items)
    assert np.array_equal(pop.scores(np.array([])), single.scores(np.array([])))


def test_item_knn_stays_inside_the_taste_cluster(toy):
    frame, _, n_users, n_items = toy
    knn = ItemKNN(neighbours=5).fit(frame["user"].to_numpy(), frame["item"].to_numpy(), n_users, n_items)
    recs = top_k(knn.scores(np.array([0, 1, 2])), 5, exclude=np.array([0, 1, 2]))
    assert all(r < 10 for r in recs)


def test_item_knn_empty_history():
    knn = ItemKNN().fit(np.array([0, 1]), np.array([0, 1]), 2, 2)
    assert np.all(knn.scores(np.array([], dtype=int)) == 0)


def test_two_tower_learns_the_clusters(toy):
    frame, genres, _, n_items = toy
    sequences = {u: g.sort_values("timestamp")["item"].to_numpy() for u, g in frame.groupby("user")}
    pop = Popularity().fit(frame["user"].to_numpy(), frame["item"].to_numpy(), n_items)
    model = TwoTower(dim=16, epochs=40, batch_size=32, max_history=10).fit(
        sequences, genres, n_items, pop.scores(np.array([]))
    )
    recs = top_k(model.scores(np.array([10, 11, 12])), 5, exclude=np.array([10, 11, 12]))
    assert sum(r >= 10 for r in recs) >= 4


def test_unfitted_models_refuse_to_score():
    with pytest.raises(RuntimeError):
        ItemKNN().scores(np.array([1]))
    with pytest.raises(RuntimeError):
        TwoTower().scores(np.array([1]))


def test_ranking_metrics_on_a_known_list():
    ranked, relevant = [3, 1, 7], {1, 9}
    assert recall_at_k(ranked, relevant, 3) == 0.5
    assert hit_at_k(ranked, relevant, 1) == 0.0 and hit_at_k(ranked, relevant, 2) == 1.0
    assert 0.0 < ndcg_at_k(ranked, relevant, 3) < 1.0
    assert ndcg_at_k([1, 9], relevant, 2) == pytest.approx(1.0)


def test_beyond_accuracy_metrics():
    genres = [("A",), ("A",), ("B",)]
    assert intra_list_diversity([0, 1], genres, 2) == 0.0
    assert intra_list_diversity([0, 2], genres, 2) == 1.0
    assert catalog_coverage([[0, 1], [1, 2]], 4, 2) == 0.75
    pop = np.array([0.5, 0.01, 0.5])
    assert novelty([[1]], pop, 1) > novelty([[0]], pop, 1)


def test_calibration_error_is_zero_when_calibrated():
    probs = np.array([0.2] * 10 + [0.8] * 10)
    labels = np.array([1, 1] + [0] * 8 + [1] * 8 + [0, 0])
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(np.full(10, 0.9), np.zeros(10)) == pytest.approx(0.9)


def test_ease_stays_inside_the_taste_cluster(toy):
    from recsys.candidates import EASE

    frame, _, n_users, n_items = toy
    model = EASE(l2=1.0).fit(frame["user"].to_numpy(), frame["item"].to_numpy(), n_users, n_items)
    recs = top_k(model.scores(np.array([10, 11])), 4, exclude=np.array([10, 11]))
    assert all(r >= 10 for r in recs)


def test_ease_never_scores_an_item_by_itself(toy):
    from recsys.candidates import EASE

    frame, _, n_users, n_items = toy
    model = EASE(l2=1.0).fit(frame["user"].to_numpy(), frame["item"].to_numpy(), n_users, n_items)
    assert np.allclose(np.diag(model._weights), 0.0)


def test_recent_popularity_ignores_old_interactions():
    from recsys.candidates import RecentPopularity

    items = np.array([0, 0, 0, 1])
    stamps = np.array([0, 0, 0, 100 * 86_400])
    model = RecentPopularity(window_days=10).fit(items, stamps, 2)
    assert top_k(model.scores(np.array([])), 1) == [1]
