import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from providers import EXTRA_CONTENT_KEY, ChatOpenAICompatible, build_model

FIRMA = {"google": {"thought_signature": "firma-opaca"}}


@pytest.fixture
def entorno(monkeypatch):
    for var in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "LLM_RPM"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_build_model_sin_key_falla_con_mensaje_claro(entorno):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_model()


def test_build_model_lee_el_entorno(entorno):
    entorno.setenv("OPENAI_API_KEY", "clave-de-prueba")
    entorno.setenv("OPENAI_MODEL", "modelo-x")
    entorno.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")
    model = build_model()
    assert isinstance(model, ChatOpenAICompatible)
    assert model.model_name == "modelo-x"
    assert model.openai_api_base == "http://localhost:9999/v1"
    assert model.temperature == 0
    assert model.rate_limiter is None


def test_llm_rpm_activa_el_limitador(entorno):
    entorno.setenv("OPENAI_API_KEY", "clave-de-prueba")
    entorno.setenv("LLM_RPM", "6")
    assert build_model().rate_limiter.requests_per_second == pytest.approx(0.1)


def _respuesta_con_firma() -> dict:
    return {
        "id": "r1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "buscar_cliente", "arguments": '{"nombre": "Ana"}'},
                            "extra_content": FIRMA,
                        }
                    ],
                },
            }
        ],
    }


def test_conserva_y_reenvia_la_thought_signature(entorno):
    model = ChatOpenAICompatible(model="m", api_key="clave-de-prueba")

    mensaje = model._create_chat_result(_respuesta_con_firma()).generations[0].message
    assert mensaje.tool_calls[0]["args"] == {"nombre": "Ana"}
    assert mensaje.additional_kwargs[EXTRA_CONTENT_KEY] == {"call_1": FIRMA}

    payload = model._get_request_payload(
        [HumanMessage("Buscá a Ana"), mensaje, ToolMessage("{}", tool_call_id="call_1")]
    )
    assert payload["messages"][1]["tool_calls"][0]["extra_content"] == FIRMA
    # el campo propio no se filtra como clave desconocida al pedido
    assert EXTRA_CONTENT_KEY not in payload["messages"][1]


def test_sin_firma_el_pedido_queda_igual(entorno):
    model = ChatOpenAICompatible(model="m", api_key="clave-de-prueba")
    ai = AIMessage("", tool_calls=[{"name": "buscar_cliente", "args": {}, "id": "c", "type": "tool_call"}])
    payload = model._get_request_payload([HumanMessage("hola"), ai])
    assert "extra_content" not in payload["messages"][1]["tool_calls"][0]
