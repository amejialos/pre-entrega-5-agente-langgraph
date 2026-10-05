import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agent import RECURSION_LIMIT, AgentState, build_graph, config_para, recortar_historial
from tests.fakes import FakeToolModel, llamada
from traza import ejecutar_turno


def _multipaso() -> list[AIMessage]:
    return [
        llamada("buscar_cliente", {"nombre": "Ana Gómez"}, "c1"),
        llamada("buscar_pedidos", {"cliente_id": 102}, "c2"),
        AIMessage("Ana Gómez hizo 3 pedidos por un total de $14.500."),
    ]


def test_el_estado_hereda_de_messages_state():
    from langgraph.graph import MessagesState

    # issubclass no funciona con TypedDict: la herencia queda registrada en __orig_bases__
    assert MessagesState in AgentState.__orig_bases__
    assert {"messages", "herramientas_usadas"} <= set(AgentState.__annotations__)


def test_config_define_thread_y_recursion_limit():
    config = config_para("t1")
    assert config["configurable"]["thread_id"] == "t1"
    assert config["recursion_limit"] == RECURSION_LIMIT == 10


async def test_multipaso_llama_dos_herramientas_y_termina():
    model = FakeToolModel(respuestas=_multipaso())
    graph = build_graph(model, InMemorySaver())

    result = await graph.ainvoke({"messages": [HumanMessage("¿Cuánto gastó Ana Gómez?")]}, config_para("t"))

    assert model.herramientas_vinculadas == ["buscar_cliente", "buscar_pedidos", "detalle_pedido"]
    assert result["herramientas_usadas"] == ["buscar_cliente", "buscar_pedidos"]
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert json.loads(tool_msgs[0].content)["cliente_id"] == 102
    assert json.loads(tool_msgs[1].content)["total"] == 14500
    assert result["messages"][-1].content.endswith("$14.500.")
    # el modelo recibe el system prompt y, en la 3ra llamada, los dos resultados
    assert isinstance(model.recibidos[0][0], SystemMessage)
    assert sum(isinstance(m, ToolMessage) for m in model.recibidos[2]) == 2


async def test_ejecutar_turno_registra_la_traza_react():
    graph = build_graph(FakeToolModel(respuestas=_multipaso()), InMemorySaver())

    turno = await ejecutar_turno(graph, "¿Cuánto gastó Ana Gómez?", "t")

    assert [p["react"] for p in turno.pasos] == ["accion", "observacion", "accion", "observacion", "respuesta_final"]
    assert [p["nodo"] for p in turno.pasos] == ["agente", "tools", "agente", "tools", "agente"]
    assert turno.tool_calls == 2
    assert turno.pasos[3]["resultado"]["cantidad_pedidos"] == 3
    assert turno.respuesta.startswith("Ana Gómez hizo 3 pedidos")
    assert turno.error is None


async def test_error_de_herramienta_vuelve_al_modelo_y_reintenta():
    model = FakeToolModel(
        respuestas=[
            llamada("buscar_cliente", {"nombre": "Carlos Ruiz Díaz"}, "c1"),
            llamada("buscar_cliente", {"nombre": "Ruiz"}, "c2"),
            AIMessage("Encontré a Carlos Ruiz (id 105)."),
        ]
    )
    graph = build_graph(model, InMemorySaver())

    result = await graph.ainvoke({"messages": [HumanMessage("¿Quién es Carlos Ruiz Díaz?")]}, config_para("t"))

    error = json.loads(model.recibidos[1][-1].content)  # lo último que vio el modelo antes de reintentar
    assert error["error"].startswith("cliente no encontrado")
    assert json.loads(result["messages"][-2].content)["cliente_id"] == 105


async def test_excepcion_en_herramienta_no_rompe_el_grafo():
    model = FakeToolModel(
        respuestas=[
            llamada("buscar_pedidos", {"cliente_id": "no-es-un-numero"}, "c1"),
            AIMessage("Necesito un id numérico."),
        ]
    )
    graph = build_graph(model, InMemorySaver())

    result = await graph.ainvoke({"messages": [HumanMessage("pedidos del cliente x")]}, config_para("t"))

    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.status == "error"
    assert result["messages"][-1].content == "Necesito un id numérico."


async def test_recursion_limit_corta_un_bucle_infinito():
    model = FakeToolModel(respuestas=[llamada("buscar_cliente", {"nombre": "Ana"}, "c")], repetir_ultima=True)
    graph = build_graph(model, InMemorySaver())

    with pytest.raises(GraphRecursionError):
        await graph.ainvoke({"messages": [HumanMessage("bucle")]}, config_para("t"))
    assert len(model.recibidos) == RECURSION_LIMIT // 2


async def test_ejecutar_turno_reporta_el_corte_por_recursion_limit():
    model = FakeToolModel(respuestas=[llamada("buscar_cliente", {"nombre": "Ana"}, "c")], repetir_ultima=True)
    graph = build_graph(model, InMemorySaver())

    turno = await ejecutar_turno(graph, "bucle", "t", recursion_limit=4)

    assert turno.error is not None and "recursion_limit=4" in turno.error
    assert turno.respuesta


async def test_memoria_persistente_en_sqlite_entre_conexiones(tmp_path):
    db = str(tmp_path / "checkpoints.sqlite")
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(FakeToolModel(respuestas=_multipaso()), saver)
        await graph.ainvoke({"messages": [HumanMessage("¿Cuánto gastó Ana Gómez?")]}, config_para("sesion"))

    # conexión y grafo nuevos sobre el mismo archivo, como si se reiniciara el proceso
    model = FakeToolModel(respuestas=[AIMessage("El último fue el pedido 1007.")])
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(model, saver)
        result = await graph.ainvoke({"messages": [HumanMessage("¿Y el último?")]}, config_para("sesion"))

    vistos = model.recibidos[0]
    assert any(isinstance(m, HumanMessage) and "Ana Gómez" in m.content for m in vistos)
    assert any(isinstance(m, ToolMessage) and "14500" in m.content for m in vistos)
    assert len(result["messages"]) == 8  # 6 del primer turno + pregunta + respuesta
    # el reducer operator.add acumula las herramientas del thread a través de los turnos
    assert result["herramientas_usadas"] == ["buscar_cliente", "buscar_pedidos"]


async def test_threads_distintos_no_comparten_memoria(tmp_path):
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        model = FakeToolModel(respuestas=[*_multipaso(), AIMessage("¿De qué cliente me hablás?")])
        graph = build_graph(model, saver)
        await graph.ainvoke({"messages": [HumanMessage("¿Cuánto gastó Ana Gómez?")]}, config_para("a"))
        await graph.ainvoke({"messages": [HumanMessage("¿Y el último?")]}, config_para("b"))

        vistos_b = model.recibidos[-1]
        assert [m.content for m in vistos_b if isinstance(m, HumanMessage)] == ["¿Y el último?"]
        assert len((await graph.aget_state(config_para("a"))).values["messages"]) == 6


def test_recortar_historial_conserva_lo_ultimo_y_arranca_en_humano():
    historial = []
    for i in range(20):
        historial += [HumanMessage(f"pregunta {i} " + "x" * 200), AIMessage(f"respuesta {i} " + "y" * 200)]
    historial += [
        HumanMessage("¿Cuánto gastó Ana?"),
        llamada("buscar_cliente", {"nombre": "Ana"}, "c1"),
        ToolMessage('{"cliente_id": 102}', tool_call_id="c1", name="buscar_cliente"),
    ]

    recortado = recortar_historial(historial, max_tokens=400)

    assert len(recortado) < len(historial)
    assert isinstance(recortado[0], HumanMessage)
    assert recortado[-3:] == historial[-3:]


def test_recortar_historial_nunca_deja_afuera_el_turno_actual():
    turno = [HumanMessage("z" * 2000), llamada("buscar_cliente", {"nombre": "Ana"}, "c1")]
    assert recortar_historial([HumanMessage("vieja"), AIMessage("ok"), *turno], max_tokens=50) == turno


async def test_el_modelo_recibe_el_historial_recortado_pero_el_estado_guarda_todo():
    model = FakeToolModel(respuestas=[AIMessage("r" * 400)] * 6)
    graph = build_graph(model, InMemorySaver(), max_tokens_historial=300)

    for i in range(6):
        result = await graph.ainvoke({"messages": [HumanMessage(f"pregunta {i} " + "p" * 400)]}, config_para("t"))

    assert len(result["messages"]) == 12
    ultimo_envio = model.recibidos[-1]
    assert len(ultimo_envio) < 12
    assert ultimo_envio[-1].content.startswith("pregunta 5")


class _ErrorHttp(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _ModeloQueFallaUnaVez(FakeToolModel):
    """La primera llamada falla con un error HTTP; las siguientes responden normalmente."""

    status_code: int = 429
    fallo: bool = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not self.fallo:
            self.fallo = True
            raise _ErrorHttp(self.status_code)
        return super()._generate(messages, stop, run_manager, **kwargs)


async def test_el_nodo_agente_reintenta_un_429():
    from langgraph.types import RetryPolicy

    model = _ModeloQueFallaUnaVez(respuestas=[AIMessage("ok")])
    rapido = RetryPolicy(max_attempts=2, initial_interval=0.01, retry_on=lambda e: getattr(e, "status_code", None) == 429)
    graph = build_graph(model, InMemorySaver(), retry_modelo=rapido)

    result = await graph.ainvoke({"messages": [HumanMessage("hola")]}, config_para("t"))

    assert model.fallo and len(model.recibidos) == 1 and result["messages"][-1].content == "ok"


async def test_un_error_permanente_no_se_reintenta():
    model = _ModeloQueFallaUnaVez(respuestas=[AIMessage("ok")], status_code=401)
    graph = build_graph(model, InMemorySaver())

    with pytest.raises(_ErrorHttp):
        await graph.ainvoke({"messages": [HumanMessage("hola")]}, config_para("t"))
    assert model.recibidos == []  # falló una vez y no hubo segundo intento
