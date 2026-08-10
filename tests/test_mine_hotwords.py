from src.lexicon.mine_hotwords import mine_candidates, normalize_word


def _by_term(candidates):
    return {c["term"]: c for c in candidates}


def test_normalize_word():
    assert normalize_word("Shiok!") == "shiok"
    assert normalize_word("don't") == "don't"


def test_substitution_errors_are_mined():
    pairs = [
        ("the food damn shiok", "the food damn shook"),
        ("so shiok can", "so shook can"),
    ]
    candidates = _by_term(mine_candidates(pairs, min_count=2))
    assert "shiok" in candidates
    assert candidates["shiok"]["sub_count"] == 2
    assert candidates["shiok"]["count"] == 2


def test_deletion_errors_are_mined():
    pairs = [
        ("go home lah okay", "go home okay"),
        ("come lah faster", "come faster"),
    ]
    candidates = _by_term(mine_candidates(pairs, min_count=2))
    assert "lah" in candidates
    assert candidates["lah"]["del_count"] == 2


def test_min_count_filters_rare_nonseed_errors():
    # "zebra" fails only once and is not a seed term -> filtered
    pairs = [("a zebra crossing", "a cobra crossing")]
    candidates = _by_term(mine_candidates(pairs, min_count=2))
    assert "zebra" not in candidates


def test_seed_terms_bypass_min_count():
    # seed term "shiok" appears once as an error but is kept anyway
    pairs = [("very shiok food", "very shook food")]
    candidates = _by_term(mine_candidates(pairs, min_count=2))
    assert "shiok" in candidates
    assert candidates["shiok"]["seed"] is True


def test_stopwords_are_filtered():
    pairs = [
        ("and then we go", "an then we going"),
        ("and then we go", "an then we going"),
        ("and then we go", "an then we going"),
    ]
    candidates = _by_term(mine_candidates(pairs, min_count=2))
    assert "and" not in candidates
    assert "go" not in candidates


def test_correctly_recognized_seed_terms_still_enter_pool():
    # "makan" recognized correctly -> included with count 0 so the lexicon covers it
    pairs = [("we go makan now", "we go makan now")] * 2
    candidates = _by_term(mine_candidates(pairs, min_count=2))
    assert "makan" in candidates
    assert candidates["makan"]["count"] == 0


def test_candidates_sorted_by_frequency():
    pairs = [
        ("shiok shiok shiok", "shook shook shook"),
        ("so paiseh", "so pie say"),
    ]
    candidates = mine_candidates(pairs, min_count=1)
    counts = [c["count"] for c in candidates]
    assert counts == sorted(counts, reverse=True)


def test_examples_are_captured():
    pairs = [("the food damn shiok", "the food damn shook")] * 2
    candidates = _by_term(mine_candidates(pairs, min_count=2))
    example = candidates["shiok"]["examples"][0]
    assert example["error"] == "substitute"
    assert "shiok" in example["ref"]
    assert example["hyp"] == "shook"
