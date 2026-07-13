from starlette.requests import Request

from backend.speech_backend import server


def _request(session_id: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"x-session-id", session_id.encode("ascii"))],
        }
    )


def test_realtime_bridges_are_isolated_per_session(monkeypatch):
    monkeypatch.setattr(server, "OPENAI_API_KEY", "test-key")
    server.realtime_sessions.clear()

    first = server._get_realtime_bridge("visitor-a")
    second = server._get_realtime_bridge("visitor-b")

    assert first is not second
    assert server._get_realtime_bridge("visitor-a") is first


def test_session_id_is_read_from_header():
    assert server._speech_session_id(_request("visitor-123")) == "visitor-123"


def test_invalid_session_id_is_rejected():
    try:
        server._speech_session_id(_request("invalid session"))
    except Exception as error:
        assert getattr(error, "status_code", None) == 400
    else:
        raise AssertionError("invalid session ID was accepted")
