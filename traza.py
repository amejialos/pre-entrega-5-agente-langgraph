"""Ejecución de un turno con registro de la traza ReAct, y volcado a .json y .log.

`ejecutar_turno` usa `astream(stream_mode="updates")`: LangGraph emite lo que devuelve
cada nodo apenas termina, así la traza muestra el ciclo paso a paso:

    agente  -> Acción: buscar_cliente({"nombre": "Ana Gómez"})      (Thought/Action)
    tools   -> Observación: {"cliente_id": 102, ...}                (Observation)
    agente  -> Acción: buscar_pedidos({"cliente_id": 102})
    tools   -> Observación: {"cantidad_pedidos": 3, "total": 14500, ...}
    agente  -> Respuesta final: "Ana Gómez hizo 3 pedidos por $14.500."

De cada mensaje se guarda solo el contenido, las tool_calls y sus resultados: nunca
`response_metadata` ni `additional_kwargs`, que pueden traer datos del proveedor.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from agent import RECURSION_LIMIT, config_para


@dataclass
class Turno:
    """Una pregunta del usuario y todo lo que hizo el agente para responderla."""

    thread_id: str
    pregunta: str
    pasos: list[dict[str, Any]] = field(default_factory=list)
    respuesta: str = ""
    tool_calls: int = 0
    error: str | None = None
    duracion_s: float = 0.0


def _parsear(texto: str) -> Any:
    """Los resultados de las herramientas son JSON: se guardan como objeto para que la traza se lea mejor."""
    try:
        return json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        return texto


def serializar_mensaje(msg: BaseMessage) -> dict[str, Any]:
    """Mensaje -> dict con la etiqueta ReAct que le corresponde."""
    if isinstance(msg, AIMessage) and msg.tool_calls:
        return {
            "react": "accion",
            "pensamiento": msg.text or None,
            "tool_calls": [{"id": c.get("id"), "nombre": c["name"], "args": c["args"]} for c in msg.tool_calls],
        }
    if isinstance(msg, AIMessage):
        return {"react": "respuesta_final", "contenido": msg.text}
    if isinstance(msg, ToolMessage):
        return {
            "react": "observacion",
            "herramienta": msg.name,
            "tool_call_id": msg.tool_call_id,
            "estado": msg.status,
            "resultado": _parsear(msg.text),
        }
    if isinstance(msg, HumanMessage):
        return {"react": "usuario", "contenido": msg.text}
    return {"react": msg.type, "contenido": msg.text}


async def ejecutar_turno(
    graph: CompiledStateGraph,
    pregunta: str,
    thread_id: str,
    recursion_limit: int = RECURSION_LIMIT,
) -> Turno:
    """Manda una pregunta al agente en el thread dado y registra cada paso del ciclo."""
    turno = Turno(thread_id=thread_id, pregunta=pregunta)
    inicio = time.perf_counter()
    numero = 0
    try:
        async for update in graph.astream(
            {"messages": [HumanMessage(pregunta)]},
            config_para(thread_id, recursion_limit),
            stream_mode="updates",
        ):
            for nodo, cambios in update.items():
                for msg in (cambios or {}).get("messages", []):
                    numero += 1
                    paso = {"paso": numero, "nodo": nodo, **serializar_mensaje(msg)}
                    turno.pasos.append(paso)
                    turno.tool_calls += len(paso.get("tool_calls", []))
                    if paso["react"] == "respuesta_final":
                        turno.respuesta = paso["contenido"]
    except GraphRecursionError:
        turno.error = f"GraphRecursionError: el agente superó recursion_limit={recursion_limit} sin terminar."
        turno.respuesta = "No pude resolver la consulta en la cantidad de pasos permitida. ¿Podés reformularla?"
    turno.duracion_s = round(time.perf_counter() - inicio, 2)
    return turno


def turno_a_texto(turno: Turno) -> str:
    """Versión legible de un turno, para el .log y la consola."""
    lineas = [f"[thread_id={turno.thread_id}] Usuario: {turno.pregunta}"]
    for p in turno.pasos:
        prefijo = f"  {p['paso']:>2}. ({p['nodo']})"
        if p["react"] == "accion":
            if p["pensamiento"]:
                lineas.append(f"{prefijo} Pensamiento: {p['pensamiento']}")
            for c in p["tool_calls"]:
                lineas.append(f"{prefijo} Acción: {c['nombre']}({json.dumps(c['args'], ensure_ascii=False)})")
        elif p["react"] == "observacion":
            resultado = json.dumps(p["resultado"], ensure_ascii=False)
            lineas.append(f"{prefijo} Observación [{p['herramienta']}]: {resultado}")
        elif p["react"] == "respuesta_final":
            lineas.append(f"{prefijo} Respuesta final: {p['contenido']}")
    if turno.error:
        lineas.append(f"  ERROR: {turno.error}")
    lineas.append(f"  -> {turno.tool_calls} llamadas a herramientas en {turno.duracion_s} s")
    return "\n".join(lineas)


def guardar_traza(
    escenarios: list[dict[str, Any]],
    metadata: dict[str, Any],
    ruta_json: Path,
    ruta_log: Path,
) -> None:
    """Escribe la traza completa en JSON y en texto. Cada escenario tiene "titulo", "notas" y "turnos"."""
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    datos = {
        "metadata": metadata,
        "escenarios": [{**e, "turnos": [asdict(t) for t in e["turnos"]]} for e in escenarios],
    }
    ruta_json.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bloques = [f"Traza ReAct - {json.dumps(metadata, ensure_ascii=False)}"]
    for i, e in enumerate(escenarios, start=1):
        bloques.append(f"\n=== Escenario {i}: {e['titulo']} ===")
        for nota in e.get("notas", []):
            bloques.append(f"# {nota}")
        bloques.extend(turno_a_texto(t) for t in e["turnos"])
    ruta_log.write_text("\n".join(bloques) + "\n", encoding="utf-8")
