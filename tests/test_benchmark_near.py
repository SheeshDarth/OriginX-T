"""Tests for benchmark-near contamination generation."""

import difflib
import inspect
import random
import re
from collections import Counter

import pytest

from src.generation import make_near_duplicates, mix, perturb, similarity
from src.generation import benchmark_near
from src.generation.benchmark_near import _recase, _repunct, _respace
from src.ingestion.schema import Sample


def benchmark(n=10):
    return [
        Sample(
            sample_id=f"bench-{i}",
            prompt=f"What is {i} plus one?",
            response=f"The answer is {i + 1}, clearly.",
            label=str(i + 1),
            split="test",
        )
        for i in range(n)
    ]


def test_marks_provenance_and_new_ids():
    out = make_near_duplicates(benchmark())
    assert len(out) == 10
    assert all(s.source == "benchmark_near" for s in out)
    assert all(s.generation == 0 for s in out)  # copied, not model-generated
    originals = {s.sample_id for s in benchmark()}
    assert not originals & {s.sample_id for s in out}, "must not reuse benchmark ids"


def test_exact_fraction_produces_verbatim_copies():
    out = make_near_duplicates(benchmark(10), exact_fraction=0.3)
    verbatim = [s for s in out if s.benchmark_near_score == 1.0]
    assert len(verbatim) >= 3


def test_perturbed_items_differ_from_source_but_stay_similar():
    src = benchmark(10)
    out = make_near_duplicates(src, exact_fraction=0.0, seed=1)
    responses = {s.response for s in src}
    changed = [s for s in out if s.response not in responses]
    assert changed, "expected perturbed text to differ from the original"
    # still recognisably the same item -- that is what makes it *near*
    assert all(s.benchmark_near_score > 0.3 for s in changed)


def test_score_is_measured_not_assumed():
    out = make_near_duplicates(benchmark(20), exact_fraction=0.5, seed=3)
    scores = [s.benchmark_near_score for s in out]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert 1.0 in scores  # exact copies
    # measured, not hardcoded: changing content lowers the score...
    assert similarity("The answer is 5.", "The answer is 6.") < 1.0
    # ...while reformatting alone does not, because it is still a full leak
    assert similarity("The Answer Is 5.", "the  answer is 5.") == 1.0


def test_deterministic_for_a_given_seed():
    a = [s.response for s in make_near_duplicates(benchmark(), seed=5)]
    b = [s.response for s in make_near_duplicates(benchmark(), seed=5)]
    assert a == b


def test_count_can_exceed_the_benchmark_pool():
    out = make_near_duplicates(benchmark(4), count=10)
    assert len(out) == 10
    assert len({s.sample_id for s in out}) == 10  # ids stay unique


def test_perturbation_preserves_content_not_formatting():
    rng = random.Random(0)
    text = "The Quick Brown Fox. Jumps!"
    variants = {perturb(text, rng) for _ in range(20)}
    assert any(v != text for v in variants)
    assert all(v.strip() for v in variants)  # never produces empty text


def test_similarity_bounds():
    assert similarity("abc", "abc") == 1.0
    assert similarity("abc", "xyz") < 0.5


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        make_near_duplicates([])
    with pytest.raises(ValueError):
        make_near_duplicates(benchmark(), exact_fraction=1.5)


def test_composes_with_the_mixer():
    """The end-to-end shape: leak benchmark items into clean data at a ratio."""
    clean = [Sample(sample_id=f"c-{i}", response=f"clean text {i}") for i in range(100)]
    leaked = make_near_duplicates(benchmark(10), count=50)
    out = mix(clean, leaked, ratio=0.25)
    assert len(out) == 100
    assert sum(1 for s in out if s.source == "benchmark_near") == 25


# --- the perturbations must leave the text near-identical to the model, not just to difflib -------------
#
# PR #12: recasing whole texts made GPT-2 see different tokens, so training on "near-duplicates" raised
# benchmark perplexity (+4.7%) instead of lowering it. These tests pin what "near" has to mean.

# Passages in the style of the WikiText-2 benchmark items (punctuation already space-separated), and in
# ordinary prose. Written for these tests, not taken from a dataset.
WIKI = [
    "The railway reached the town in 1887 , and within a decade the harbour had doubled its trade . Merchants "
    "from Aberdeen opened warehouses along the quay , while the old market square , once the centre of local "
    "life , fell quiet . By 1901 the population had grown to nearly 4 @,@ 000 , a figure that would not be "
    "matched again until the 1960s . The council 's decision to build a second bridge was widely criticised ; "
    "however , it proved decisive for the town 's later growth .",
    "Robert Vance ( born 12 March 1962 ) is an English cricketer who played for Kent between 1983 and 1996 . "
    "A right @-@ handed batsman , he scored 11 @,@ 204 first @-@ class runs at an average of 38 @.@ 4 . His "
    "highest score , 187 not out against Essex in 1989 , remains a county record for a number six . After "
    "retiring he worked as a coach and later as a commentator for local radio . He was appointed an MBE in "
    "2004 for services to the sport .",
    "The album received generally favourable reviews from critics . Writing for The Guardian , Alexis Moore "
    "called it \" a confident , restless record \" , although she felt the closing tracks lacked focus . It "
    "reached number 14 on the UK Albums Chart and was certified silver in March 2012 . The lead single , \" "
    "Paper Lanterns \" , was released in January and spent six weeks in the top 40 . Two further singles "
    "followed before the band began a tour of North America .",
]
PROSE = [
    "The committee met on Tuesday to review the proposal. Several members, including the chair, raised "
    "concerns about the budget; others argued that the delay would cost more in the end. \"We can't keep "
    "postponing this,\" said one councillor. After two hours of debate, the vote was carried by a narrow "
    "margin! The plan will now go to the full council, where it is expected to face further scrutiny.",
    "It was a cold morning in November when the letter arrived. Maria read it twice, then folded it carefully "
    "and put it in her coat pocket. She didn't tell anyone, not even her sister, who had always known when "
    "something was wrong. By evening the snow had started, and the streets were empty; the whole town seemed "
    "to be holding its breath. Was it really possible? She wasn't sure.",
]
SEEDS = range(200)

# GPT-2 splits text into pre-tokens with this pattern before applying BPE merges inside each one, so two
# texts with the same pre-tokens have the same tokens. This is the standard-library approximation of that
# pattern (\p{L} ~ [^\W\d_], \p{N} ~ \d); on the passages above it matches the exact pattern.
GPT2_PRETOKEN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")


def old_recase(text, rng):
    """What _recase did before: the whole text, in ALL CAPS or all lowercase."""
    return text.lower() if rng.random() < 0.5 else text.upper()


def word_overlap(original, perturbed):
    """Share of the original's whitespace-separated words (case-sensitive, counted) still present."""
    a, b = Counter(original.split()), Counter(perturbed.split())
    return sum((a & b).values()) / sum(a.values())


def token_similarity(original, perturbed):
    """difflib ratio over GPT-2 pre-tokens: 1.0 means the model sees the same sequence."""
    a, b = GPT2_PRETOKEN.findall(original), GPT2_PRETOKEN.findall(perturbed)
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def is_mixed_case(text):
    return text not in (text.lower(), text.upper())


MIXED_INPUTS = WIKI + PROSE + [
    "Hello world",                     # one capital: flipping it would lower-case the whole text
    "The answer is 5.",
    "The cat sat. The dog ran.",       # only sentence starts are capitals
    "\"Why?\" asked Maria. \"Because.\"",
    "1887 was cold. The end came in May.",
    "iPhone sales rose. Apple said so.",
    "hELLO WORLD. wELL DONE.",         # inverted case: flipping the starts would make it ALL CAPS
]
EDITS = {"perturb": perturb, "_recase": _recase}


@pytest.mark.parametrize("edit", EDITS)
def test_no_perturbed_output_is_a_full_upper_or_lower_casing(edit):
    fn = EDITS[edit]
    for text in MIXED_INPUTS:
        assert is_mixed_case(text)
        for seed in SEEDS:
            out = fn(text, random.Random(seed))
            assert out != text.lower() and out != text.upper(), (text, seed, out)
            assert is_mixed_case(out), (text, seed, out)  # the output is not all one case either


def test_the_old_recase_fails_that_check():
    # the test above has teeth: the previous implementation produces a full recasing in about half of the runs
    text = WIKI[0]
    outs = [old_recase(text, random.Random(seed)) for seed in SEEDS]
    full = sum(out in (text.lower(), text.upper()) for out in outs)
    assert full > 0.4 * len(outs)


@pytest.mark.parametrize("text", WIKI)
def test_word_overlap_stays_high_on_multi_sentence_input(text):
    for seed in SEEDS:
        assert word_overlap(text, perturb(text, random.Random(seed))) >= 0.8, seed
        assert word_overlap(text, _recase(text, random.Random(seed))) >= 0.9, seed


def test_old_recase_fails_the_word_overlap_check():
    for text in WIKI:
        overlaps = [word_overlap(text, old_recase(text, random.Random(seed))) for seed in SEEDS]
        assert sum(o < 0.8 for o in overlaps) > 0.4 * len(overlaps)  # the ALL CAPS half
        assert min(overlaps) < 0.3


def test_prose_with_dense_punctuation_keeps_most_words_too():
    # punctuation swaps and a case flip stack up on commas, quotes and apostrophes: still most of the text
    for text in PROSE:
        for seed in SEEDS:
            assert word_overlap(text, perturb(text, random.Random(seed))) >= 0.7, seed
            assert word_overlap(text, _recase(text, random.Random(seed))) >= 0.9, seed
            assert word_overlap(text, _respace(text, random.Random(seed))) == 1.0, seed


@pytest.mark.parametrize("text", WIKI)
def test_the_model_sees_almost_the_same_tokens(text):
    for seed in SEEDS:
        assert token_similarity(text, perturb(text, random.Random(seed))) >= 0.8, seed
        assert token_similarity(text, _recase(text, random.Random(seed))) >= 0.9, seed
        assert token_similarity(text, _respace(text, random.Random(seed))) >= 0.95, seed
        assert token_similarity(text, _repunct(text, random.Random(seed))) >= 0.85, seed


def test_old_recase_changes_most_of_the_tokens():
    for text in WIKI:
        ratios = [token_similarity(text, old_recase(text, random.Random(seed))) for seed in SEEDS]
        assert min(ratios) < 0.4  # ALL CAPS: nearly every word is a different token


def test_old_respace_padded_every_space_so_half_the_tokens_were_filler():
    text = WIKI[0]
    padded = re.sub(r"\s+", " ", text).replace(" ", "  ")  # the old padding branch
    assert token_similarity(text, padded) < 0.75
    new = {_respace(text, random.Random(seed)) for seed in SEEDS}
    assert all(token_similarity(text, out) >= 0.95 for out in new)


def test_recase_flips_the_first_letter_or_every_sentence_start():
    text = "The cat sat on Mars. The dog ran off. Then it rained."
    outs = {_recase(text, random.Random(seed)) for seed in SEEDS}
    assert outs == {
        "the cat sat on Mars. The dog ran off. Then it rained.",   # the first letter only
        "the cat sat on Mars. the dog ran off. then it rained.",   # every sentence start
    }


def test_recase_flips_lowercase_starts_up_and_ignores_leading_non_letters():
    outs = {_recase("\"hello there. goodbye now.", random.Random(seed)) for seed in SEEDS}
    assert outs == {"\"Hello there. goodbye now.", "\"Hello there. Goodbye now."}
    assert {_recase("1887 was cold. the end.", random.Random(seed)) for seed in SEEDS} == {
        "1887 Was cold. the end.", "1887 Was cold. The end."
    }


def test_recase_sees_a_sentence_start_after_a_closing_quote_or_bracket():
    text = 'He said "no." Then he left. (Really.) Why not?'
    outs = {_recase(text, random.Random(seed)) for seed in SEEDS}
    assert outs == {
        'he said "no." Then he left. (Really.) Why not?',
        'he said "no." then he left. (Really.) why not?',  # the sentence after the bracket is "Why not"
    }


def test_recase_sees_sentence_starts_after_question_and_exclamation_marks():
    text = "Wow! Really? Yes, Paris."  # "Paris" keeps it mixed-case when every start is flipped
    assert {_recase(text, random.Random(seed)) for seed in SEEDS} == {
        "wow! Really? Yes, Paris.", "wow! really? yes, Paris."
    }


def test_recase_works_on_non_ascii_letters():
    text = "Élan vital in Paris. Ünder the sea. Привет Мир. Да."
    outs = {_recase(text, random.Random(seed)) for seed in SEEDS}
    assert outs == {
        "élan vital in Paris. Ünder the sea. Привет Мир. Да.",
        "élan vital in Paris. ünder the sea. привет Мир. да.",
    }


def test_repunct_is_the_unchanged_swap_of_each_present_punctuation_kind():
    # kept as it was: each kind present is swapped for all its occurrences, or not, independently
    outs = {_repunct("a, b. c!", random.Random(seed)) for seed in SEEDS}
    assert outs == {
        f"a{comma} b{stop} c{bang}"
        for comma in (",", " ,") for stop in (".", " .") for bang in ("!", ".")
    }
    assert {_repunct("no marks here", random.Random(seed)) for seed in SEEDS} == {"no marks here"}


def test_recase_only_treats_a_letter_after_whitespace_as_a_sentence_start():
    # "a.txt" and "U.S.A" are not sentence ends
    text = "The file a.txt is the U.S.A copy. Then b.py ran."
    outs = {_recase(text, random.Random(seed)) for seed in SEEDS}
    assert outs == {
        "the file a.txt is the U.S.A copy. Then b.py ran.",
        "the file a.txt is the U.S.A copy. then b.py ran.",
    }


def test_recase_does_not_turn_an_inverted_case_text_into_all_caps():
    text = "hELLO WORLD."  # flipping the first letter alone would give HELLO WORLD.
    assert {_recase(text, random.Random(seed)) for seed in SEEDS} == {text}


def test_recase_leaves_a_text_without_letters_alone_and_never_recases_a_single_capital():
    for text in ("", "1887 , 1901 .", "@-@ @,@"):
        assert _recase(text, random.Random(0)) == text
    # flipping the only capital would lower-case the whole text, so nothing happens
    assert {_recase("Hello world", random.Random(seed)) for seed in SEEDS} == {"Hello world"}


def test_respace_pads_after_sentence_ends_not_every_word():
    text = "One two . Three four . Five six ."
    outs = {_respace(text, random.Random(seed)) for seed in SEEDS}
    assert outs == {text, "One two .  Three four .  Five six ."}
    assert all(out.split() == text.split() for out in outs)  # only whitespace changes


def test_respace_doubles_one_space_when_there_is_no_sentence_end():
    text = "no full stops in this line"
    outs = {_respace(text, random.Random(seed)) for seed in SEEDS}
    assert text in outs
    padded = outs - {text}
    assert padded and all(out.count("  ") == 1 and out.split() == text.split() for out in padded)
    assert _respace("single", random.Random(0)) == "single"


def test_respace_collapses_messy_whitespace():
    messy = "  one\ttwo\n\nthree   four  "
    assert "one two three four" in {_respace(messy, random.Random(seed)) for seed in SEEDS}


def test_perturb_never_returns_empty_text_and_keeps_the_ops_the_old_module_had():
    assert all(perturb(t, random.Random(seed)).strip() for t in ("x", ".", "The end") for seed in SEEDS)
    assert set(benchmark_near._PERTURBATIONS) == {_recase, _respace, _repunct}


def test_exact_copies_still_score_one_and_perturbed_ones_stay_in_bounds():
    items = [Sample(sample_id=f"w{i}", response=t, split="validation") for i, t in enumerate(WIKI + PROSE)]
    out = make_near_duplicates(items, count=100, exact_fraction=0.3, seed=7)
    n_exact = round(100 * 0.3)

    for i, s in enumerate(out):
        original = items[i % len(items)].response
        assert 0.0 <= s.benchmark_near_score <= 1.0
        if i < n_exact:
            assert s.response == original and s.benchmark_near_score == 1.0
        elif i % len(items) < len(WIKI):
            assert s.benchmark_near_score >= 0.95  # WikiText-style: the punctuation is already spaced out
        else:
            # dense prose loses more to the punctuation swaps (unchanged by this edit); still the same text
            assert s.benchmark_near_score >= 0.5
    assert all(s.source == "benchmark_near" and s.generation == 0 for s in out)


def test_exactly_exact_fraction_of_the_copies_are_verbatim(monkeypatch):
    # a perturbation can leave a text unchanged by chance, so mark the perturbed ones instead
    monkeypatch.setattr(benchmark_near, "perturb", lambda text, rng: text + " <perturbed>")
    items = benchmark(7)
    for count, fraction in ((10, 0.3), (10, 0.0), (10, 1.0), (7, 0.5)):
        out = make_near_duplicates(items, count=count, exact_fraction=fraction)
        n_exact = round(count * fraction)
        assert [s.response.endswith("<perturbed>") for s in out] == [i >= n_exact for i in range(count)]


def test_similarity_is_still_case_blind():
    assert similarity(WIKI[0], WIKI[0].upper()) == 1.0
    assert similarity(WIKI[0], WIKI[0].lower()) == 1.0
    assert similarity("The Answer Is 5.", "the  answer is 5.") == 1.0
    assert similarity("The answer is 5.", "The answer is 6.") < 1.0


def test_make_near_duplicates_signature_and_output_schema_are_unchanged():
    params = inspect.signature(make_near_duplicates).parameters
    assert list(params) == ["benchmark", "count", "exact_fraction", "seed"]
    assert [params[k].default for k in ("count", "exact_fraction", "seed")] == [None, 0.3, 0]
    assert all(params[k].kind is inspect.Parameter.KEYWORD_ONLY for k in ("count", "exact_fraction", "seed"))

    item = Sample(sample_id="b1", prompt="What is it ?", response=WIKI[0], label="x", split="test")
    (out,) = make_near_duplicates([item], count=1, exact_fraction=0.0, seed=3)
    assert set(out.to_dict()) == set(Sample.field_names())
    assert (out.sample_id, out.label, out.split, out.source, out.generation) == (
        "b1-near0", "x", "test", "benchmark_near", 0
    )
    assert out.prompt and out.response and 0.0 <= out.benchmark_near_score <= 1.0
