"""Corre los escenarios de prueba contra el modelo real y guarda la traza ReAct.

    uv run python demo.py

Escenarios:
1. Multi-paso: una pregunta por nombre obliga a encadenar buscar_cliente -> buscar_pedidos.
2. Memoria persistente: se CIERRA la conexión a SQLite, se abre otra sobre el mismo
   archivo y se arma un grafo nuevo; con el mismo thread_id, "¿y el último?" se
   resuelve con el contexto guardado.
3. Ciclo de retorno (reintento): un nombre que no existe tal cual -> la herramienta
   devuelve error con sugerencia -> el agente reintenta con otros argumentos.
4. Ciclo de retorno (aclaración): un apellido ambiguo -> el agente pregunta; la
   respuesta del usuario en el mismo thread completa la consulta.
5. Aislamiento: un thread_id nuevo no ve la conversación de los otros.

Salida: trazas/traza_ejecucion.json y trazas/traza_ejecucion.log.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import RECURSION_LIMIT, build_graph, config_para
from providers import DEFAULT_OPENAI_MODEL, build_model
from traza import Turno, ejecutar_turno, guardar_traza, turno_a_texto

BASE = Path(__file__).parent
DB_PATH = BASE / "checkpoints.sqlite"
TRAZA_JSON = BASE / "trazas" / "traza_ejecucion.json"
TRAZA_LOG = BASE / "trazas" / "traza_ejecucion.log"



def _nuevo_escenario(titulo: str) -> dict[str, Any]:
    return {"titulo": titulo, "notas": [], "turnos": []}


async def _turno(escenario: dict[str, Any], graph: Any, pregunta: str, thread_id: str) -> Turno:
    turno = await ejecutar_turno(graph, pregunta, thread_id)
    escenario["turnos"].append(turno)
    print(turno_a_texto(turno), end="\n\n", flush=True)
    return turno


async def correr_demo(model: BaseChatModel, db_path: Path, nota: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Ejecuta los cinco escenarios y devuelve sus turnos (sin escribir archivos)."""
    db_path.unlink(missing_ok=True)  # cada corrida arranca sin historia previa
    escenarios: list[dict[str, Any]] = []

    # --- Fase 1: primera conexión al archivo SQLite ---
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = build_graph(model, checkpointer)
        e1 = _nuevo_escenario("Multi-paso: el agente encadena dos herramientas")
        escenarios.append(e1)
        await _turno(e1, graph, "¿Cuántos pedidos hizo Ana Gómez y cuánto gastó en total?", "sesion-ana")

    # --- Fase 2: conexión NUEVA al mismo archivo y grafo nuevo (como reiniciar el proceso) ---
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = build_graph(model, checkpointer)

        e2 = _nuevo_escenario("Memoria persistente: mismo thread_id tras reabrir el SQLite")
        escenarios.append(e2)
        estado = await graph.aget_state(config_para("sesion-ana"))
        recuperados = len(estado.values.get("messages", []))
        e2["notas"].append(
            f"Checkpointer y grafo recreados desde {db_path.name}: el thread 'sesion-ana' "
            f"trae {recuperados} mensajes guardados en la fase anterior."
        )
        nota(e2["notas"][-1])
        await _turno(e2, graph, "¿Y cuál fue el último? ¿Qué compró en ese pedido?", "sesion-ana")

        e3 = _nuevo_escenario("Ciclo de retorno: error de la herramienta -> reintento")
        escenarios.append(e3)
        await _turno(e3, graph, "¿Cuánto gastó en total Carlos Ruiz Díaz?", "sesion-ruiz")

        e4 = _nuevo_escenario("Ciclo de retorno: nombre ambiguo -> pide aclaración")
        escenarios.append(e4)
        await _turno(e4, graph, "¿Cuánto gastó en total el cliente Martínez?", "sesion-martinez")
        await _turno(e4, graph, "Me refiero a Lucía.", "sesion-martinez")

        e5 = _nuevo_escenario("Aislamiento: otro thread_id no comparte memoria")
        escenarios.append(e5)
        await _turno(e5, graph, "¿Y cuál fue el último pedido?", "sesion-nueva")

    return escenarios


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    model = build_model()
    escenarios = await correr_demo(model, DB_PATH)
    metadata = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
        "recursion_limit": RECURSION_LIMIT,
        "checkpointer": "AsyncSqliteSaver",
        "total_tool_calls": sum(t.tool_calls for e in escenarios for t in e["turnos"]),
    }
    guardar_traza(escenarios, metadata, TRAZA_JSON, TRAZA_LOG)
    print(f"Traza guardada en {TRAZA_JSON.relative_to(BASE)} y {TRAZA_LOG.relative_to(BASE)}")


if __name__ == "__main__":
    asyncio.run(main())
