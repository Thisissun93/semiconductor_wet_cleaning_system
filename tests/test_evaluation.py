from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    FACTORS,
    RESPONSES,
    build_face_centered_ccd,
    coded_to_actual,
    actual_to_coded,
    confirmation_plan,
    fit_quadratic,
    forty_nine_points,
    predict_grid,
    recommend_condition,
    simulate_responses,
    simulate_wafer_map,
    uniformity_summary,
)
from src.evaluation.analysis import model_matrix
from src.evaluation.simulate import true_responses


@pytest.fixture(scope="module")
def evaluation():
    design = build_face_centered_ccd(center_points=4, seed=7)
    results = simulate_responses(design, seed=11)
    fits = {r.key: fit_quadratic(results, r) for r in RESPONSES}
    grid = predict_grid(fits, steps=21)
    return design, results, fits, grid


def test_ccd_has_expected_structure():
    design = build_face_centered_ccd(center_points=4)

    assert len(design) == 8 + 6 + 4
    assert (design["point_type"] == "요인점").sum() == 8
    assert (design["point_type"] == "축점").sum() == 6
    assert (design["point_type"] == "중심점").sum() == 4
    assert sorted(design["run_order"]) == list(range(1, len(design) + 1))
    assert design["wafer_id"].is_unique

    coded = design[[f"{f.key}_coded" for f in FACTORS]].to_numpy()
    assert set(np.unique(coded)) <= {-1.0, 0.0, 1.0}


def test_ccd_rejects_zero_center_points():
    with pytest.raises(ValueError):
        build_face_centered_ccd(center_points=0)


def test_coded_actual_round_trip():
    coded = np.array([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.25]])
    actual = coded_to_actual(coded)

    assert actual.loc[0, "temperature_c"] == FACTORS[0].low
    assert actual.loc[0, "megasonic_w"] == FACTORS[2].high
    np.testing.assert_allclose(actual_to_coded(actual), coded)


def test_simulated_responses_are_physical(evaluation):
    _, results, _, _ = evaluation

    assert results["pre_pct"].between(0, 100).all()
    assert (results["oxide_loss_a"] >= 0).all()
    assert (results["pattern_damage"] >= 0).all()


def test_expected_trends_follow_assumptions():
    low_power = true_responses(np.array([[0.0, 0.0, -1.0]]))
    high_power = true_responses(np.array([[0.0, 0.0, 1.0]]))
    hot = true_responses(np.array([[1.0, 0.0, 0.0]]))
    cold = true_responses(np.array([[-1.0, 0.0, 0.0]]))

    assert high_power["pre_pct"][0] > low_power["pre_pct"][0]
    assert high_power["pattern_damage"][0] > low_power["pattern_damage"][0]
    assert hot["oxide_loss_a"][0] > cold["oxide_loss_a"][0]


def test_quadratic_fit_recovers_exact_model():
    rng = np.random.default_rng(0)
    coded = rng.uniform(-1, 1, size=(30, 3))
    beta = np.array([5.0, 1.0, -2.0, 0.5, 0.3, 0.0, -0.4, 0.2, 0.1, -0.6])
    y = model_matrix(coded) @ beta

    frame = pd.DataFrame(coded, columns=[f"{f.key}_coded" for f in FACTORS])
    frame["oxide_loss_a"] = y
    response = next(r for r in RESPONSES if r.key == "oxide_loss_a")
    fit = fit_quadratic(frame, response)

    np.testing.assert_allclose(fit.coefficients, beta, atol=1e-8)
    assert fit.r_squared == pytest.approx(1.0)


def test_transforms_round_trip():
    for response in RESPONSES:
        values = np.array([1.0, 2.0, 50.0, 90.0])
        np.testing.assert_allclose(
            response.inverse(response.forward(values)), values, rtol=1e-9
        )


def test_models_explain_most_variation(evaluation):
    _, _, fits, _ = evaluation

    assert fits["pre_pct"].adj_r_squared > 0.9
    assert fits["oxide_loss_a"].adj_r_squared > 0.9


def test_recommended_condition_meets_all_specs(evaluation):
    _, _, fits, grid = evaluation
    recommended = recommend_condition(grid)

    assert recommended is not None
    for response in RESPONSES:
        value = recommended[response.key]
        if response.goal == "maximize":
            assert value >= response.spec
        else:
            assert value <= response.spec

    plan = confirmation_plan(fits, recommended)
    assert len(plan) == len(RESPONSES)
    assert (plan["확인 웨이퍼 수"] >= 3).all()


def test_recommendation_is_none_when_nothing_is_feasible(evaluation):
    _, _, _, grid = evaluation
    infeasible = grid.assign(feasible=False)

    assert recommend_condition(infeasible) is None


def test_prediction_interval_stays_in_physical_range(evaluation):
    _, _, fits, _ = evaluation
    low, high = fits["pre_pct"].prediction_interval(np.array([[1.0, 1.0, 1.0]]))

    assert 0.0 <= low[0] <= high[0] <= 100.0


def test_forty_nine_point_layout():
    points = forty_nine_points()

    assert len(points) == 49
    assert points["site"].is_unique
    assert (np.hypot(points["x_mm"], points["y_mm"]) <= 150.0).all()


def test_hotter_condition_is_less_uniform():
    cool = uniformity_summary(simulate_wafer_map(5.0, temperature_coded=-1.0))
    hot = uniformity_summary(simulate_wafer_map(5.0, temperature_coded=1.0))

    assert hot["nu_1sigma_pct"] > cool["nu_1sigma_pct"]
    assert cool["mean"] == pytest.approx(5.0, rel=0.05)
