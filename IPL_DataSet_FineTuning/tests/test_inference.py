from fastapi.testclient import TestClient
from service.app import app

def test_health():
    client = TestClient(app)
    r = client.get('/healthz')
    assert r.status_code == 200

def test_infer():
    client = TestClient(app)
    r = client.post('/infer', json={'prompt': 'How many runs in last 5 overs?'})
    assert r.status_code == 200
    assert 'answer' in r.json()
