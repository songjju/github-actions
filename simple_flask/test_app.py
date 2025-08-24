# test_app.py
import pytest
from app import app

@pytest.fixture
def client():
    """테스트 클라이언트 설정"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """홈페이지 테스트"""
    response = client.get('/')
    assert response.status_code == 200

def test_about_page(client):
    """소개 페이지 테스트"""
    response = client.get('/about')
    assert response.status_code == 200

def test_contact_page(client):
    """연락처 페이지 테스트"""
    response = client.get('/contact')
    assert response.status_code == 200

def test_404_page(client):
    """존재하지 않는 페이지 테스트"""
    response = client.get('/nonexistent')
    assert response.status_code == 404