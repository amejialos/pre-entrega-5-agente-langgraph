import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from traza import Turno, guardar_traza, serializar_mensaje


def test_serializar_mensajes_con_etiquetas_react():
    accion = AIMessage(
        "",
        tool_calls=[{"name": "buscar_cliente", "args": {"nombre": "Ana"}, "id": "c1", "type": "tool_call"}],
        additional_kwargs={"privado": "no va a la traza"},
        response_metadata={"headers": "tampoco"},
    )
    assert serializar_mensaje(accion) == {
        "react": "accion",
        "pensamiento": None,
        "tool_calls": [{"id": "c1", "nombre": "buscar_cliente", "args": {"nombre": "Ana"}}],
    }
    obs = serializar_mensaje(ToolMessage('{"cliente_id": 102}', tool_call_id="c1", name="buscar_cliente"))
    assert obs["react"] == "observacion" and obs["resultado"] == {"cliente_id": 102}
    assert serializar_mensaje(AIMessage("listo")) == {"react": "respuesta_final", "contenido": "listo"}
    assert serializar_mensaje(HumanMessage("hola"))["react"] == "usuario"


def test_guardar_traza_escribe_json_y_log(tmp_path):
    turno = Turno(thread_id="t", pregunta="¿Cuánto gastó Ana?", respuesta="$14.500", tool_calls=2)
    turno.pasos = [
        {"paso": 1, "nodo": "agente", "react": "accion", "pensamiento": None,
         "tool_calls": [{"id": "c1", "nombre": "buscar_cliente", "args": {"nombre": "Ana"}}]},
        {"paso": 2, "nodo": "tools", "react": "observacion", "herramienta": "buscar_cliente",
         "tool_call_id": "c1", "estado": "success", "resultado": {"cliente_id": 102}},
        {"paso": 3, "nodo": "agente", "react": "respuesta_final", "contenido": "$14.500"},
    ]
    ruta_json, ruta_log = tmp_path / "trazas" / "t.json", tmp_path / "trazas" / "t.log"

    guardar_traza([{"titulo": "Multi-paso", "notas": ["nota"], "turnos": [turno]}], {"modelo": "m"}, ruta_json, ruta_log)

    datos = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert datos["escenarios"][0]["turnos"][0]["tool_calls"] == 2
    log = ruta_log.read_text(encoding="utf-8")
    assert 'Acción: buscar_cliente({"nombre": "Ana"})' in log
    assert "Observación [buscar_cliente]" in log and "Respuesta final: $14.500" in log
