"""SC1 세정 조건 평가(DOE) 모듈.

평가용 웨이퍼 실험을 계획하고, 합성 평가 결과를 회귀 분석해
판정 기준을 만족하는 세정 조건과 확인 실험 계획을 제시합니다.

모든 결과값은 공개용 합성 데이터이며 특정 장비·고객 조건을 재현하지 않습니다.
"""

from src.evaluation.design import (
    FACTORS,
    Factor,
    build_face_centered_ccd,
    coded_to_actual,
    actual_to_coded,
)
from src.evaluation.simulate import (
    RESPONSES,
    Response,
    simulate_responses,
)
from src.evaluation.analysis import (
    QuadraticFit,
    fit_quadratic,
    predict_grid,
    recommend_condition,
    confirmation_plan,
)
from src.evaluation.wafer_map import (
    forty_nine_points,
    simulate_wafer_map,
    uniformity_summary,
)

__all__ = [
    "FACTORS",
    "Factor",
    "build_face_centered_ccd",
    "coded_to_actual",
    "actual_to_coded",
    "RESPONSES",
    "Response",
    "simulate_responses",
    "QuadraticFit",
    "fit_quadratic",
    "predict_grid",
    "recommend_condition",
    "confirmation_plan",
    "forty_nine_points",
    "simulate_wafer_map",
    "uniformity_summary",
]
