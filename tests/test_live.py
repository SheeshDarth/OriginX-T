"""Tests for src/evaluation/live.py: the logger with a stub measure, the HF callback, and one
real tiny run (skipped without torch) proving that logging does not change training."""

import json
import math
from pathlib import Path

import pytest

from src.evaluation import live

# the committed trainlog/1 fixture fixes the row shape the UI and X2 build against
FIXTURE = Path(__file__).parent / "fixtures" / "trainlog.jsonl"
FIELDS = set(json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0]))


def test_the_logger_writes_the_fixtures_fields():
    assert FIELDS == {"schema", "run_id", "step", "holdout_loss", "alarm", "alarm_signal",
                      *(f"sig_{k}" for k in live.SIGNALS)}


class FakeModel:
    def __init__(self):
        self.training, self.modes = True, []

    def eval(self):
        self.training = False
        self.modes.append("eval")

    def train(self):
        self.training = True
        self.modes.append("train")


def _stub(model, tok, logger):
    assert not model.training  # measured in eval mode
    return {"holdout_loss": 4.0 - len(logger.rows) * 0.1, **{k: 1.0 for k in live.SIGNALS}}


def test_rows_every_n_steps_plus_the_last_in_the_trainlog_1_shape(tmp_path):
    path = tmp_path / "logs/run.jsonl"
    seen = []
    logger = live.TrainLogger(path, "wikitext2_synthetic_r1_s0", ["probe text"], every=3,
                              measure=_stub, on_row=seen.append)
    model = FakeModel()
    for step in range(8):
        logger(step, model, None, last=step == 7)
    assert [r["step"] for r in logger.rows] == [0, 3, 6, 7]
    assert model.training and model.modes.count("eval") == 4  # train mode restored each time
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == logger.rows == seen
    for r in rows:
        assert set(r) == FIELDS and r["schema"] == "trainlog/1" and r["alarm"] is False
    assert rows[1]["holdout_loss"] == pytest.approx(3.9)


def test_a_last_step_on_the_grid_is_logged_once_and_a_new_logger_starts_a_new_file(tmp_path):
    path = tmp_path / "run.jsonl"
    logger = live.TrainLogger(path, "r", ["p"], every=2, measure=_stub)
    for step in range(5):
        logger(step, FakeModel(), None, last=step == 4)
    assert [r["step"] for r in logger.rows] == [0, 2, 4]
    live.TrainLogger(path, "r", ["p"], measure=_stub)
    assert path.read_text(encoding="utf-8") == ""


def test_bad_settings_are_refused(tmp_path):
    with pytest.raises(ValueError):
        live.TrainLogger(tmp_path / "x.jsonl", "r", ["p"], every=0)
    with pytest.raises(ValueError):
        live.TrainLogger(tmp_path / "x.jsonl", "r", [])


def test_the_trainer_callback_drives_the_logger(tmp_path):
    pytest.importorskip("transformers")
    from transformers import TrainerState

    calls = []
    cb = live.trainer_callback(lambda step, model, tok, last=False: calls.append((step, model, tok, last)))
    state = TrainerState(max_steps=3)
    cb.on_train_begin(None, state, None, model="m", processing_class="t")
    for step in (1, 2, 3):
        state.global_step = step
        cb.on_step_end(None, state, None, model="m", processing_class="t")
    assert calls == [(0, "m", "t", False), (1, "m", "t", False), (2, "m", "t", False), (3, "m", "t", True)]


def test_presets_name_real_cells_of_the_wikitext_grid():
    import yaml

    from src.finetune.grid import make_cells

    cfg = yaml.safe_load(open("configs/grid.yaml", encoding="utf-8"))
    ids = {c.id for c in make_cells(cfg["types"], cfg["ratios"], cfg["seeds"])}
    assert set(live.PRESETS["x3"]) <= ids and len(live.PRESETS["x3"]) == 18


def test_a_logged_run_trains_exactly_like_an_unlogged_one(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    from src.finetune.lora import train_lora

    texts = [f"The {w} sat on the mat and looked at the {w} by the door." for w in
             ("cat", "dog", "bird", "fox", "owl", "cow", "pig", "hen")]
    probe = ["A small house stood at the end of the quiet road.", "Rain fell all night on the hills."]
    kw = dict(base_model="distilgpt2", epochs=1, batch_size=2, max_len=32, lora_r=4, seed=7, device="cpu")
    try:
        plain = train_lora(texts, tmp_path / "plain", **kw)
    except OSError:
        pytest.skip("distilgpt2 not available offline")
    logger = live.TrainLogger(tmp_path / "log.jsonl", "tiny", probe, every=2, batch_size=2, max_len=32)
    logged = train_lora(texts, tmp_path / "logged", on_step=logger, **kw)

    a = AutoModelForCausalLM.from_pretrained(plain).state_dict()
    b = AutoModelForCausalLM.from_pretrained(logged).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)  # bit-identical

    rows = logger.rows
    assert [r["step"] for r in rows] == [0, 2, 4]
    assert rows[0]["sig_delta_rel_norm"] == 0.0 and rows[-1]["sig_delta_rel_norm"] > 0  # LoRA moved W
    assert all(math.isfinite(r["holdout_loss"]) and all(r[f"sig_{k}"] is not None for k in live.SIGNALS)
               for r in rows)
    assert rows[-1]["sig_stable_rank_attn"] != rows[0]["sig_stable_rank_attn"]  # sees the LoRA delta
