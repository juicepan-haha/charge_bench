"""共享测试夹具。"""

import json
import pathlib

import pytest

from chargebench.schemas import Scenario

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_scenario(name: str = "demo_scenario.json") -> Scenario:
    return Scenario(**json.loads((CONFIG_DIR / name).read_text(encoding="utf-8")))


@pytest.fixture
def demo_scenario() -> Scenario:
    """已标定到关键区间的演示场景：网络约束生效且算法可分化。"""
    return load_scenario()
