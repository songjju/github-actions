import pytest
from pathlib import Path
import sys

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_project_structure():
    """기본 프로젝트 구조 테스트"""
    assert Path("main.py").exists()
    assert Path("src").exists()
    assert True  # 기본 성공 테스트