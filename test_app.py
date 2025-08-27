import pytest
import json
from app import app, init_predictor

@pytest.fixture(scope="module")
def client():
    # 앱 테스트 클라이언트 초기화
    init_predictor()  # predictor 초기화
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index(client):
    """메인 페이지 렌더링 확인"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"<html" in response.data or b"<!DOCTYPE html" in response.data


def test_predict(client):
    """예측 API 확인"""
    response = client.post('/api/predict')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "number" in data
    assert "jo" in data
    assert "next_draw" in data
    assert "next_date" in data
    assert "confidence" in data
    assert "individual_predictions" in data


def test_recent(client):
    """최근 당첨번호 API"""
    response = client.get('/api/recent?count=5')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) <= 5
    if len(data) > 0:
        assert "draw_no" in data[0]
        assert "jo" in data[0]
        assert "number" in data[0]


def test_statistics(client):
    """통계 API"""
    response = client.get('/api/statistics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "digit_frequency" in data
    assert "jo_frequency" in data
    assert isinstance(data["digit_frequency"], dict)
    assert isinstance(data["jo_frequency"], dict)


def test_history(client):
    """히스토리 API"""
    # 먼저 예측 실행
    client.post('/api/predict')
    response = client.get('/api/history')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    if len(data) > 0:
        assert "number" in data[0]
        assert "jo" in data[0]
