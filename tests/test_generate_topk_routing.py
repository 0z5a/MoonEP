"""Routing-generator contract tests.

These exercise ``tests/generate_topk_routing.py`` alone: they import ``torch``
and the generator, never ``moonep``, and need no distributed environment.

Run the CPU lane with ``CUDA_VISIBLE_DEVICES=""`` and the CUDA lane once per
visible device. The two backends are checked against the same contract but
their random streams are never compared element-wise.
"""

import pytest
import torch

from tests.generate_topk_routing import generate_topk_routing


BALANCED = "balanced"
BIASED = "biased"
BIAS_RATIO = {BALANCED: 0.0, BIASED: 1.0}

# (S, K, E, R): E % R == 0 and K <= E hold throughout. The last two keep
# S * K from being a multiple of R, which exercises the round-robin remainder.
ROUTING_CASES = [
    (8, 2, 4, 1),
    (17, 3, 6, 3),
    (17, 3, 9, 3),
    (16, 4, 8, 2),
    (33, 2, 16, 4),
    (64, 2, 20, 5),
    (256, 8, 256, 8),
]

DEFAULT_CASE = (17, 3, 9, 3)
DEFAULT_SEED = 20260919


def _devices():
    if not torch.cuda.is_available():
        return ["cpu"]
    return ["cpu"] + [f"cuda:{index}" for index in range(torch.cuda.device_count())]


DEVICES = _devices()


def _routing(dev, mode, S, K, E, R, seed=DEFAULT_SEED, rank=0):
    return generate_topk_routing(S, K, E, R, BIAS_RATIO[mode], dev, seed, rank)


def _assert_metadata(topk, tpe, S, K, E):
    assert topk.shape == (S, K)
    assert topk.dtype == torch.int32
    assert tpe.shape == (E,)
    assert tpe.dtype == torch.int32
    assert int(tpe.sum()) == S * K
    assert int(topk.min()) >= 0
    assert int(topk.max()) < E
    counts = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    assert torch.equal(counts, tpe)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("mode", [BALANCED, BIASED])
@pytest.mark.parametrize("S,K,E,R", ROUTING_CASES)
def test_metadata_dtype_ids_and_histogram(dev, mode, S, K, E, R):
    for rank in range(R):
        topk, tpe = _routing(dev, mode, S, K, E, R, rank=rank)
        _assert_metadata(topk, tpe, S, K, E)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("mode", [BALANCED, BIASED])
def test_repeated_calls_reproduce_exactly(dev, mode):
    for rank in range(3):
        first_topk, first_tpe = _routing(dev, mode, *DEFAULT_CASE, rank=rank)
        again_topk, again_tpe = _routing(dev, mode, *DEFAULT_CASE, rank=rank)
        assert torch.equal(first_topk, again_topk)
        assert torch.equal(first_tpe, again_tpe)


@pytest.mark.parametrize("dev", DEVICES)
def test_balanced_branch_does_not_consume_seed(dev):
    """Balanced routing is rank-seeded, so ``seed`` must not reach its output."""
    base_topk, base_tpe = _routing(dev, BALANCED, *DEFAULT_CASE, seed=1, rank=2)
    for seed in (0, 7, 20260919, 2**31 - 1):
        topk, tpe = _routing(dev, BALANCED, *DEFAULT_CASE, seed=seed, rank=2)
        assert torch.equal(topk, base_topk)
        assert torch.equal(tpe, base_tpe)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("S,K,E,R", ROUTING_CASES)
def test_balanced_round_robin_share_of_every_rank(dev, S, K, E, R):
    """Rank r receives S*K/R entries when the split is exact, otherwise the
    shares differ by at most one, and never by more."""
    epn = E // R
    for rank in range(R):
        _topk, tpe = _routing(dev, BALANCED, S, K, E, R, rank=rank)
        own = int(tpe[rank * epn:(rank + 1) * epn].sum())
        assert abs(own * R - S * K) <= R - 1, (S, K, E, R, rank, own)


@pytest.mark.parametrize("dev", DEVICES)
def test_balanced_rank_histograms_match_when_the_split_is_exact(dev):
    """The rank-seeded permutation is a bijection on [0, epn), so an exact
    S*K/R split gives every rank the same multiset of per-expert counts, placed
    on different experts. With an inexact split the per-rank multisets may
    differ, so only the exact case is asserted."""
    S, K, E, R = DEFAULT_CASE
    epn = E // R
    assert (S * K) % R == 0
    histograms = []
    for rank in range(R):
        _topk, tpe = _routing(dev, BALANCED, S, K, E, R, rank=rank)
        histograms.append(torch.sort(tpe[rank * epn:(rank + 1) * epn]).values)
    for histogram in histograms[1:]:
        assert torch.equal(histogram, histograms[0])


@pytest.mark.parametrize("dev", DEVICES)
def test_biased_branch_matches_an_explicit_generator_oracle(dev):
    """Pin the draw order: one size-(E,) normal draw from the seed generator
    produces the shared expert logits, and a rank generator then drives the
    per-token multinomial samples."""
    S, K, E, R = DEFAULT_CASE
    bias_ratio = BIAS_RATIO[BIASED]
    for rank in range(R):
        logit_gen = torch.Generator(device=dev).manual_seed(DEFAULT_SEED)
        logits = torch.exp(torch.normal(
            0.0, bias_ratio, size=(E,), device=dev, generator=logit_gen
        ))
        probs = logits[None, :].expand(S, E)
        token_gen = torch.Generator(device=dev).manual_seed(rank)
        expected = torch.multinomial(
            probs, K, replacement=False, generator=token_gen
        ).to(torch.int32)
        topk, tpe = _routing(dev, BIASED, S, K, E, R, rank=rank)
        assert torch.equal(topk, expected)
        assert torch.equal(
            tpe, torch.bincount(expected.flatten(), minlength=E).to(torch.int32)
        )


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("mode", [BALANCED, BIASED])
def test_explicit_generators_leave_the_default_rng_untouched(dev, mode):
    if dev == "cpu":
        before = torch.random.get_rng_state()
        _routing(dev, mode, *DEFAULT_CASE)
        assert torch.equal(torch.random.get_rng_state(), before)
    else:
        index = torch.device(dev).index
        before = torch.cuda.get_rng_state(index)
        _routing(dev, mode, *DEFAULT_CASE)
        assert torch.equal(torch.cuda.get_rng_state(index), before)
