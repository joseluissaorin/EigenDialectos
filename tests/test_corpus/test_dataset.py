"""Tests for the DialectDataset class."""

from __future__ import annotations

import pytest

from eigendialectos.constants import DialectCode
from eigendialectos.corpus.dataset import DialectDataset
from eigendialectos.types import CorpusSlice, DialectSample


# ======================================================================
# Fixtures
# ======================================================================


def _make_sample(
    text: str,
    code: DialectCode,
    idx: int = 0,
) -> DialectSample:
    return DialectSample(
        text=text,
        dialect_code=code,
        source_id="test",
        confidence=1.0,
        metadata={"index": idx},
    )


@pytest.fixture
def large_corpus() -> dict[DialectCode, CorpusSlice]:
    """20 samples per dialect for split tests."""
    result: dict[DialectCode, CorpusSlice] = {}
    for code in DialectCode:
        samples = [
            _make_sample(f"Ejemplo de texto {code.value} número {i}.", code, i)
            for i in range(20)
        ]
        result[code] = CorpusSlice(samples=samples, dialect_code=code)
    return result


@pytest.fixture
def dataset(large_corpus) -> DialectDataset:
    return DialectDataset(large_corpus)


@pytest.fixture
def small_dataset(tiny_corpus) -> DialectDataset:
    return DialectDataset(tiny_corpus)


# ======================================================================
# Creation tests
# ======================================================================


class TestDatasetCreation:
    """Tests for DialectDataset construction."""

    def test_all_dialects_present(self, dataset):
        for code in DialectCode:
            assert code in dataset

    def test_len(self, dataset):
        assert len(dataset) == 20 * len(DialectCode)

    def test_getitem(self, dataset):
        sample = dataset[0]
        assert isinstance(sample, DialectSample)
        assert len(sample.text) > 0

    def test_getitem_range(self, dataset):
        total = len(dataset)
        for idx in [0, total // 2, total - 1]:
            sample = dataset[idx]
            assert isinstance(sample, DialectSample)

    def test_iter(self, dataset):
        count = sum(1 for _ in dataset)
        assert count == len(dataset)

    def test_contains(self, dataset):
        assert DialectCode.ES_PEN in dataset
        assert DialectCode.ES_RIO in dataset

    def test_repr(self, dataset):
        r = repr(dataset)
        assert "DialectDataset" in r

    def test_dialect_codes_property(self, dataset):
        codes = dataset.dialect_codes
        assert len(codes) == len(DialectCode)
        for code in DialectCode:
            assert code in codes


# ======================================================================
# Access and sampling tests
# ======================================================================


class TestDatasetAccess:
    """Tests for get_slice, all_samples, and sample."""

    def test_get_slice(self, dataset):
        sl = dataset.get_slice(DialectCode.ES_MEX)
        assert isinstance(sl, CorpusSlice)
        assert sl.dialect_code == DialectCode.ES_MEX
        assert len(sl.samples) == 20

    def test_get_slice_missing_raises(self):
        """Requesting a dialect not in the dataset should raise KeyError."""
        empty = DialectDataset({})
        with pytest.raises(KeyError):
            empty.get_slice(DialectCode.ES_PEN)

    def test_all_samples(self, dataset):
        samples = dataset.all_samples()
        assert len(samples) == len(dataset)
        assert isinstance(samples, list)

    def test_sample_all(self, dataset):
        sampled = dataset.sample(5, seed=42)
        assert len(sampled) == 5
        for s in sampled:
            assert isinstance(s, DialectSample)

    def test_sample_single_dialect(self, dataset):
        sampled = dataset.sample(3, dialect_code=DialectCode.ES_CHI, seed=42)
        assert len(sampled) == 3
        for s in sampled:
            assert s.dialect_code == DialectCode.ES_CHI

    def test_sample_reproducibility(self, dataset):
        s1 = dataset.sample(10, seed=99)
        s2 = dataset.sample(10, seed=99)
        for a, b in zip(s1, s2):
            assert a.text == b.text

    def test_sample_more_than_available(self, dataset):
        """Requesting more than available should return all."""
        sampled = dataset.sample(1000, dialect_code=DialectCode.ES_AND, seed=42)
        assert len(sampled) == 20  # all available


# ======================================================================
# Statistics tests
# ======================================================================


class TestDatasetStats:
    """Tests for the stats method."""

    def test_stats_structure(self, dataset):
        s = dataset.stats()
        assert "total_samples" in s
        assert "dialect_counts" in s
        assert s["total_samples"] == 20 * len(DialectCode)

    def test_stats_per_dialect(self, dataset):
        s = dataset.stats()
        counts = s["dialect_counts"]
        for code in DialectCode:
            assert counts[code.value] == 20

    def test_stats_empty_dataset(self):
        empty = DialectDataset({})
        s = empty.stats()
        assert s["total_samples"] == 0


# ======================================================================
# Split tests
# ======================================================================


class TestDatasetSplit:
    """Tests for stratified train/val/test splitting."""

    def test_split_default(self, dataset):
        train, val, test = dataset.split()
        total = len(train) + len(val) + len(test)
        assert total == len(dataset)

    def test_split_proportions(self, dataset):
        train, val, test = dataset.split(train=0.7, val=0.15, test=0.15)
        # Each dialect should have samples in each split
        for code in DialectCode:
            train_n = len(train.get_slice(code).samples)
            val_n = len(val.get_slice(code).samples)
            test_n = len(test.get_slice(code).samples)
            assert train_n + val_n + test_n == 20
            assert train_n >= val_n  # train should be larger

    def test_split_all_dialects_in_each(self, dataset):
        train, val, test = dataset.split()
        for code in DialectCode:
            assert code in train
            assert code in val
            assert code in test

    def test_split_reproducibility(self, dataset):
        t1, v1, te1 = dataset.split(seed=42)
        t2, v2, te2 = dataset.split(seed=42)
        assert len(t1) == len(t2)
        assert len(v1) == len(v2)
        assert len(te1) == len(te2)
        for a, b in zip(t1.all_samples(), t2.all_samples()):
            assert a.text == b.text

    def test_split_different_seeds(self, dataset):
        t1, _, _ = dataset.split(seed=1)
        t2, _, _ = dataset.split(seed=2)
        texts1 = [s.text for s in t1.all_samples()]
        texts2 = [s.text for s in t2.all_samples()]
        # Different seeds should produce different orderings
        assert texts1 != texts2

    def test_split_invalid_proportions(self, dataset):
        with pytest.raises(ValueError):
            dataset.split(train=0.5, val=0.5, test=0.5)

    def test_split_small_corpus(self, small_dataset):
        """Splitting a very small corpus (5 per dialect) should not crash."""
        train, val, test = small_dataset.split()
        total = len(train) + len(val) + len(test)
        assert total == len(small_dataset)

    def test_split_no_data_loss(self, dataset):
        """All original texts should appear in exactly one split."""
        train, val, test = dataset.split(seed=42)
        original = sorted(s.text for s in dataset.all_samples())
        split_texts = sorted(
            s.text
            for s in train.all_samples() + val.all_samples() + test.all_samples()
        )
        assert original == split_texts


# ======================================================================
# CorpusSlice stats tests
# ======================================================================


class TestCorpusSliceStats:
    """Tests for CorpusSlice.stats property."""

    def test_stats_nonempty(self, tiny_corpus):
        for code, cs in tiny_corpus.items():
            stats = cs.stats
            assert stats["count"] == 5
            assert stats["avg_length"] > 0
            assert stats["min_length"] > 0
            assert stats["max_length"] >= stats["min_length"]

    def test_stats_empty(self):
        cs = CorpusSlice(samples=[], dialect_code=DialectCode.ES_PEN)
        stats = cs.stats
        assert stats["count"] == 0
        assert stats["avg_length"] == 0.0
