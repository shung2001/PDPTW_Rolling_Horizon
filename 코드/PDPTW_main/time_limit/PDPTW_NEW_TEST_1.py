# -*- coding: utf-8 -*-
"""PDPTW_NEW_TEST에 선택된 request별 직항 제약을 추가한 실행 파일.

NEW: 원본의 입출력, rolling-horizon, Pending penalty, 비용 및 상태 갱신
기능은 그대로 재사용하고, 선택된 Pickup의 바로 다음 노드가 그 request의
Delivery가 되도록 하는 제약만 추가한다.
"""

from __future__ import annotations

import PDPTW_NEW_TEST as _original
from PDPTW_NEW_TEST import *  # noqa: F401,F403 - NEW: 원본 공개 기능을 그대로 제공한다.


# NEW: wrapper가 호출되기 전에 보존하여 아래 확장 함수가 자기 자신을 다시
# 호출하는 재귀를 방지한다.
_ORIGINAL_BUILD_HORIZON_MODEL = _original.build_horizon_model


def _sync_configuration_to_original() -> None:
    """NEW: 이 파일에서 변경한 기존 전역 설정을 원본 실행 모듈에 반영한다."""
    for name, value in globals().items():
        if name.isupper() and hasattr(_original, name):
            setattr(_original, name, value)


def build_horizon_model(
    active: list[_original.RequestTask],
    batches: dict[str, _original.BatchState],
    vehicles: list[_original.VehicleState],
    distance: _original.pd.DataFrame,
    flight_time: _original.pd.DataFrame,
    fare_matrix: _original.pd.DataFrame,
    hs: int,
    he: int,
    maximum_max_wait: int,
    return_to_home: bool = False,
) -> _original.HorizonModel:
    """NEW: 원본 HorizonModel에 선택된 request의 직항 순서만 추가한다."""
    _sync_configuration_to_original()
    model = _ORIGINAL_BUILD_HORIZON_MODEL(
        active,
        batches,
        vehicles,
        distance,
        flight_time,
        fare_matrix,
        hs,
        he,
        maximum_max_wait,
        return_to_home,
    )
    solver = model.routing.solver()

    # NEW: Pickup이 선택(Active=1)된 경우에만 Next(Pickup)=Delivery를 강제한다.
    # Pickup이 drop(Active=0)된 경우에는 기존 AddDisjunction/Pending penalty의
    # 선택 가능성을 그대로 유지한다. 이에 따라 A1->A2->B1->B2 형태의
    # shared ride는 금지되고, 선택된 request는 A1->B1의 직항 쌍으로 처리된다.
    for task in active:
        pickup_index = model.manager.NodeToIndex(model.pickup_nodes[task.task_key])
        delivery_index = model.manager.NodeToIndex(model.delivery_nodes[task.task_key])
        direct_successor = solver.IsEqualCstVar(
            model.routing.NextVar(pickup_index),
            delivery_index,
        )
        solver.Add(model.routing.ActiveVar(pickup_index) <= direct_successor)

    return model


# NEW: 원본 main()과 원본 내부 호출도 위의 직항 제약 확장 함수를 사용하게 한다.
_original.build_horizon_model = build_horizon_model


def main() -> None:
    """NEW: 기존 설정 호환성을 유지한 채 직항 제약 버전을 실행한다."""
    _sync_configuration_to_original()
    _original.build_horizon_model = build_horizon_model
    _original.main()


if __name__ == "__main__":
    # NEW: 이 파일을 직접 실행할 때 수정된 직항 제약 모델을 사용한다.
    main()
