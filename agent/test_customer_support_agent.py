import asyncio
import json
from types import SimpleNamespace

import httpx

from agent.customer_support_agent import AgentSettings, classify_text, run_demo


def test_classify_text_authenticates_and_calls_classify():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/auth/signin":
            return httpx.Response(
                200,
                json={"access_token": "jwt-de-teste", "token_type": "bearer"},
            )
        if request.url.path == "/classify":
            assert request.headers["authorization"] == "Bearer jwt-de-teste"
            return httpx.Response(
                200,
                json={"intent": "Billing inquiry", "probability": 0.61},
            )
        return httpx.Response(404)

    settings = AgentSettings(
        api_url="http://api.test",
        api_username="alice",
        api_password="alice123",
        api_token=None,
        gemini_api_key="chave-de-teste",
        gemini_model="modelo-de-teste",
    )
    client = httpx.AsyncClient(
        base_url=settings.api_url,
        transport=httpx.MockTransport(handler),
    )

    async def exercise():
        try:
            return await classify_text(
                "Fui cobrado duas vezes.", settings=settings, client=client
            )
        finally:
            await client.aclose()

    result = asyncio.run(exercise())

    assert result == {"intent": "Billing inquiry", "probability": 0.61}
    assert [request.url.path for request in requests] == [
        "/auth/signin",
        "/classify",
    ]


def test_classify_text_accepts_preconfigured_token():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/classify"
        assert request.headers["authorization"] == "Bearer token-pronto"
        return httpx.Response(
            200,
            json={"intent": "Technical issue", "probability": 0.55},
        )

    settings = AgentSettings(
        api_url="http://api.test",
        api_username="unused",
        api_password="unused",
        api_token="token-pronto",
        gemini_api_key="chave-de-teste",
        gemini_model="modelo-de-teste",
    )
    client = httpx.AsyncClient(
        base_url=settings.api_url,
        transport=httpx.MockTransport(handler),
    )

    async def exercise():
        try:
            return await classify_text("O aplicativo não abre.", settings=settings, client=client)
        finally:
            await client.aclose()

    result = asyncio.run(exercise())
    assert result["intent"] == "Technical issue"


def test_empty_text_is_rejected_before_http_call():
    try:
        asyncio.run(classify_text("   "))
    except ValueError as exc:
        assert "não pode ser vazio" in str(exc)
    else:
        raise AssertionError("Era esperado ValueError para texto vazio")


def test_demo_runs_three_turns_with_same_session(monkeypatch, tmp_path):
    calls = []

    async def fake_run(agent, message, *, session):
        calls.append((agent, message, session))
        return SimpleNamespace(final_output=f"Resposta para: {message}")

    monkeypatch.setattr(
        "agent.customer_support_agent.Runner.run",
        fake_run,
    )
    transcript_path = tmp_path / "conversation.json"
    messages = ["Turno 1", "Turno 2", "Turno 3"]
    settings = AgentSettings(
        api_url="http://api.test",
        api_username="unused",
        api_password="unused",
        api_token="token-pronto",
        gemini_api_key="chave-de-teste",
        gemini_model="modelo-de-teste",
    )

    transcript = asyncio.run(
        run_demo(messages, transcript_path=transcript_path, settings=settings)
    )

    assert len(calls) == 3
    assert calls[0][2] is calls[1][2] is calls[2][2]
    assert [item["user"] for item in transcript] == messages
    assert json.loads(transcript_path.read_text(encoding="utf-8")) == transcript
