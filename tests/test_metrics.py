from src.eval.metrics import compute_cer, compute_wer, normalize_text, term_recall


def test_normalize_text():
    assert normalize_text("  Wah, SHIOK lah!  ") == "wah shiok lah"
    assert normalize_text("don't stop") == "don't stop"


def test_wer_perfect_match():
    assert compute_wer(["go makan lah"], ["go makan lah"]) == 0.0


def test_wer_ignores_case_and_punctuation():
    assert compute_wer(["Wah, shiok lah!"], ["wah shiok lah"]) == 0.0


def test_wer_counts_substitution():
    # 1 substitution over 3 reference words
    assert compute_wer(["go makan lah"], ["go making lah"]) == 1 / 3


def test_cer_perfect_and_nonzero():
    assert compute_cer(["shiok"], ["shiok"]) == 0.0
    assert compute_cer(["shiok"], ["shook"]) > 0.0


def test_term_recall_basic():
    refs = ["wah the food damn shiok lah", "so sian today"]
    hyps = ["wah the food damn shook lah", "so sian today"]
    result = term_recall(refs, hyps, ["shiok", "lah", "sian"])
    # shiok missed (1 ref, 0 hits), lah hit, sian hit -> 2/3
    assert result["n_ref_occurrences"] == 3
    assert result["n_hits"] == 2
    assert result["recall"] == 2 / 3
    assert result["per_term"]["shiok"] == {"ref": 1, "hit": 0}


def test_term_recall_word_boundaries():
    # "lah" must not match inside "flah"
    result = term_recall(["say lah"], ["say flah"], ["lah"])
    assert result["n_hits"] == 0


def test_term_recall_multiword_term():
    result = term_recall(["we ate nasi lemak today"], ["we ate nasi lemak today"], ["nasi lemak"])
    assert result["recall"] == 1.0


def test_term_recall_caps_hallucinated_repeats():
    # hypothesis repeats the term; hits capped at reference count
    result = term_recall(["one lah only"], ["one lah lah lah"], ["lah"])
    assert result["n_ref_occurrences"] == 1
    assert result["n_hits"] == 1


def test_term_recall_no_occurrences_returns_none():
    result = term_recall(["hello world"], ["hello world"], ["shiok"])
    assert result["recall"] is None
    assert result["n_ref_occurrences"] == 0


def test_bootstrap_wer_and_cer_ci():
    from src.eval.metrics import bootstrap_cer_ci, bootstrap_wer_ci

    # Perfect match -> CI is exactly (0.0, 0.0)
    low, high = bootstrap_wer_ci(["hello world"], ["hello world"], n_bootstrap=50)
    assert low == 0.0
    assert high == 0.0

    low_c, high_c = bootstrap_cer_ci(["hello"], ["hello"], n_bootstrap=50)
    assert low_c == 0.0
    assert high_c == 0.0

    # Non-perfect match
    refs = ["the quick brown fox", "jumped over the lazy dog", "another test sentence"]
    hyps = ["the quick brown cat", "jumped the lazy dog", "another test sentence"]
    low_w, high_w = bootstrap_wer_ci(refs, hyps, n_bootstrap=100)
    assert 0.0 < low_w <= high_w <= 1.0


def test_stratify_by_duration():
    from src.eval.metrics import stratify_by_duration

    records = [
        {"duration_s": 5.0},
        {"duration_s": 15.0},
        {"duration_s": 25.0},
    ]
    refs = ["short clip", "medium length clip", "very long clip audio"]
    hyps = ["short clip", "medium length", "very long clip audio"]

    strat = stratify_by_duration(records, refs, hyps)
    assert "short (<10s)" in strat
    assert strat["short (<10s)"]["n_clips"] == 1
    assert strat["short (<10s)"]["wer"] == 0.0

    assert strat["medium (10-20s)"]["n_clips"] == 1
    assert strat["medium (10-20s)"]["wer"] > 0.0

    assert strat["long (>20s)"]["n_clips"] == 1
    assert strat["long (>20s)"]["wer"] == 0.0
