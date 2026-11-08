from src.evaluation.canary import verdict


def test_a_monotone_target_passes_and_a_reversed_one_fails():
    proxy = {("t", r, s): r + s * 0.01 for r in (0.5, 1.0) for s in range(6)}
    assert verdict(proxy, dict(proxy), n_boot=200)["status"] == "passed"
    assert verdict(proxy, {k: -v for k, v in proxy.items()}, n_boot=200)["status"] == "failed"
