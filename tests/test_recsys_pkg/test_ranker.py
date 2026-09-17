import numpy as np
import pytest

from recsys.candidates import ItemKNN, Popularity
from recsys.ranker import FEATURES, CandidateMerger, InteractionModel, LambdaMARTRanker
from recsys.rerank import mmr


def _merger(toy):
    frame, genres, n_users, n_items = toy
    users, items = frame["user"].to_numpy(), frame["item"].to_numpy()
    pop = Popularity().fit(users, items, n_items)
    knn = ItemKNN(neighbours=5).fit(users, items, n_users, n_items)
    names = sorted({g for gs in genres for g in gs})
    mat = np.array([[1.0 if g in gs else 0.0 for g in names] for gs in genres])
    return CandidateMerger(
        generators=[pop, knn], depth=10,
        pop_all=pop.share, pop_recent=pop.share, item_genres=mat,
        user_meta={u: (25.0, 1.0) for u in range(n_users)},
    ), mat


def test_candidates_exclude_history_and_have_every_feature(toy):
    merger, _ = _merger(toy)
    cs = merger.build(0, np.array([0, 1, 2]))
    assert not set(cs.items) & {0, 1, 2}
    assert cs.features.shape == (cs.items.size, len(FEATURES))


def test_ranker_learns_cluster_membership(toy):
    merger, _ = _merger(toy)
    sets = [merger.build(u, np.array([u % 10 if u < 10 else 10 + u % 10])) for u in range(20)]
    relevant = {u: set(range(0, 10)) if u < 10 else set(range(10, 20)) for u in range(20)}
    ranker = LambdaMARTRanker(n_estimators=30, min_child_samples=2).fit(sets, relevant)
    items, scores = ranker.rank(merger.build(12, np.array([12])))
    assert items.size and np.all(np.diff(scores) <= 1e-12)


def test_ranker_refuses_without_positives(toy):
    merger, _ = _merger(toy)
    with pytest.raises(ValueError):
        LambdaMARTRanker().fit([merger.build(0, np.array([0]))], {0: {999}})


def test_interaction_model_returns_probabilities(toy):
    merger, _ = _merger(toy)
    sets = [merger.build(u, np.array([u % 10 if u < 10 else 10 + u % 10])) for u in range(20)]
    relevant = {u: set(range(0, 10)) if u < 10 else set(range(10, 20)) for u in range(20)}
    model = InteractionModel().fit(sets[::2], sets[1::2], relevant)
    raw, cal = model.predict(sets[3].features)
    assert np.all((raw >= 0) & (raw <= 1)) and np.all((cal >= 0) & (cal <= 1))


def test_mmr_at_one_is_the_relevance_order():
    genres = np.eye(4)
    assert mmr(np.array([0, 1, 2, 3]), np.array([0.1, 0.9, 0.5, 0.3]), genres, 4, 1.0) == [1, 2, 3, 0]


def test_mmr_trades_relevance_for_diversity():
    genres = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    items, rel = np.array([0, 1, 2]), np.array([1.0, 0.95, 0.2])
    assert mmr(items, rel, genres, 2, 1.0) == [0, 1]
    assert mmr(items, rel, genres, 2, 0.3) == [0, 2]


def test_mmr_rejects_bad_lambda():
    with pytest.raises(ValueError):
        mmr(np.array([0]), np.array([1.0]), np.eye(1), 1, 1.5)


def test_per_generator_layout_has_a_column_pair_per_generator(toy):
    merger, _ = _merger(toy)
    merger.layout = "per_generator"
    cs = merger.build(0, np.array([0, 1]))
    assert cs.features.shape[1] == len(merger.feature_names) == 2 * 2 + 1 + 6
    assert merger.feature_names[:2] == ("popularity_z", "popularity_rank")


def test_unknown_layout_is_rejected(toy):
    merger, _ = _merger(toy)
    with pytest.raises(ValueError):
        CandidateMerger(generators=merger.generators, layout="nope")


def test_missing_proposals_rank_below_every_real_proposal(toy):
    merger, _ = _merger(toy)
    merger.layout = "per_generator"
    cs = merger.build(0, np.array([0, 1]))
    rank_col = merger.feature_names.index("item_knn_rank")
    assert cs.features[:, rank_col].max() <= merger.depth + 1


def test_single_rank_feature_ranker_reproduces_that_generator(toy):
    from recsys.candidates import top_k

    merger, _ = _merger(toy)
    merger.layout = "per_generator"
    col = merger.feature_names.index("item_knn_rank")
    sets = [merger.build(u, np.array([u % 10 if u < 10 else 10 + u % 10])) for u in range(20)]
    relevant = {u: set(range(0, 10)) if u < 10 else set(range(10, 20)) for u in range(20)}
    ranker = LambdaMARTRanker(n_estimators=50, min_child_samples=1, columns=(col,)).fit(sets, relevant)
    knn = merger.generators[1]
    history = np.array([12])
    by_ranker = ranker.rank(merger.build(12, history))[0][:3].tolist()
    by_knn = top_k(knn.scores(history), 3, exclude=history)
    assert set(by_ranker) == set(by_knn)


def test_monotone_rank_ranker_preserves_the_generator_order(toy):
    from recsys.candidates import top_k

    merger, _ = _merger(toy)
    merger.layout = "per_generator"
    col = merger.feature_names.index("item_knn_rank")
    sets = [merger.build(u, np.array([u % 10 if u < 10 else 10 + u % 10])) for u in range(20)]
    relevant = {u: set(range(0, 10)) if u < 10 else set(range(10, 20)) for u in range(20)}
    ranker = LambdaMARTRanker(n_estimators=50, min_child_samples=1, columns=(col,), monotone=(-1,)).fit(sets, relevant)
    cs = merger.build(12, np.array([12]))
    items, scores = ranker.rank(cs)
    ranks = cs.features[:, col][np.searchsorted(cs.items, items)]
    assert np.all(np.diff(scores) <= 1e-12)
    # With a non-increasing mapping, a strictly better rank never scores lower.
    for i in range(len(items)):
        for j in range(len(items)):
            if ranks[i] < ranks[j]:
                assert scores[i] >= scores[j] - 1e-12
