"""Tests for model-based contamination (synthetic / recursive / paraphrased), using a stub model."""

from src.generation import make_paraphrased, make_synthetic, mix
from src.ingestion.schema import Sample


def echo(text):
    return f"<gen:{text}>"


def human():
    return [
        Sample(sample_id="qa-0", prompt="Name a colour.", response="Blue."),
        Sample(sample_id="raw-0", response="one two three four five six"),
    ]


def test_synthetic_answers_prompt_or_continues_prefix():
    qa, raw = make_synthetic(human(), echo)
    assert qa.response == "<gen:Name a colour.>"
    assert raw.response == "one two three <gen:one two three>"
    assert {qa.source, raw.source} == {"synthetic"} and qa.generation == 1
    assert qa.sample_id == "qa-0-g1"


def test_resynthesizing_goes_recursive_and_deeper():
    gen2 = make_synthetic(make_synthetic(human(), echo), echo)
    gen3 = make_synthetic(gen2, echo)
    assert all(s.source == "recursive" for s in gen2 + gen3)
    assert [s.generation for s in gen3] == [3, 3]
    assert gen3[0].sample_id == "qa-0-g1-g2-g3"


def test_paraphrase_rewrites_response_keeps_prompt():
    out = make_paraphrased(human(), echo)
    assert out[0].prompt == "Name a colour."
    assert "Blue." in out[0].response and out[0].response != "Blue."
    assert all(s.source == "paraphrased" and s.generation == 1 for s in out)


def test_output_feeds_the_mixer():
    h = human()
    mixed = mix(h, make_paraphrased(h, echo), ratio=0.5, seed=0)
    assert sorted(s.source for s in mixed) == ["human", "paraphrased"]
