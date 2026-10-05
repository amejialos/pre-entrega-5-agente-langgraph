"""El agente: un StateGraph de LangGraph con el ciclo ReAct modelo <-> herramientas.

    START -> agente --tools_condition--> tools -> agente -> ... -> END

- `agente` llama al LLM (con `bind_tools`). El LLM decide solo si responde o si pide
  herramientas: no hay ningún if/else nuestro que elija la herramienta.
- `tools_condition` mira el último mensaje: si trae `tool_calls` va a `tools`, si no
  termina. `tools` (un `ToolNode`) ejecuta las llamadas y vuelve a `agente` con los
  resultados, que pueden ser datos o errores en JSON: el modelo decide si reintenta,
  encadena otra herramienta o le pide una aclaración al usuario.
- Con un checkpointer, el estado de cada `thread_id` se guarda después de cada paso,
  así la conversación continúa donde quedó (incluso en otro proceso, con SQLite).
- `recursion_limit` corta un ciclo que no converge.
"""

import operator
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS

# Cada vuelta modelo -> herramientas consume 2 pasos del grafo. Con 10 entran hasta 4
# rondas de herramientas más la respuesta final; un bucle infinito se corta ahí.
RECURSION_LIMIT = 10
# Presupuesto (aproximado) de tokens del historial que se manda al modelo en cada llamada.
MAX_TOKENS_HISTORIAL = 4000

SYSTEM_PROMPT = """Sos el asistente de atención al cliente de una tienda de tecnología.
Respondés preguntas sobre clientes y pedidos usando EXCLUSIVAMENTE las herramientas:
- buscar_cliente(nombre) -> cliente_id
- buscar_pedidos(cliente_id) -> cantidad, total y lista de pedidos
- detalle_pedido(pedido_id) -> productos de un pedido

Reglas:
1. Nunca inventes ids, montos ni productos: si no los tenés, consultá la herramienta que corresponda.
   Podés encadenar varias herramientas antes de responder.
2. Si una herramienta devuelve "error", leé la "sugerencia" y seguila: reintentá con otros
   argumentos o, si hace falta un dato que solo sabe el usuario (por ejemplo, un nombre ambiguo),
   preguntale en vez de adivinar.
3. Usá el contexto de la conversación: si el usuario dice "¿y el último?" se refiere al cliente
   del que venían hablando; reutilizá los ids que ya conocés.
4. Respondé en español rioplatense, breve y concreto. Los montos van en pesos con punto de
   miles, por ejemplo $14.500."""


class AgentState(MessagesState):
    """Estado del grafo: los mensajes (heredados, con el reducer `add_messages`) más
    un registro de las herramientas que pidió el modelo en este thread.

    `operator.add` es el reducer de `herramientas_usadas`: cada nodo devuelve solo la
    lista nueva y LangGraph la concatena a la acumulada (como hace `add_messages`).
    """

    herramientas_usadas: Annotated[list[str], operator.add]


def recortar_historial(messages: list[AnyMessage], max_tokens: int) -> list[AnyMessage]:
    """Últimos mensajes que entran en `max_tokens`, empezando siempre en un HumanMessage.

    Empezar en un mensaje del usuario evita mandarle al modelo un ToolMessage huérfano
    (sin el AIMessage que lo pidió), que los proveedores rechazan. Si ni siquiera el
    turno actual entra en el presupuesto, se manda el turno actual completo: recortarlo
    rompería el par tool_call / resultado.
    """
    recortados = trim_messages(
        messages,
        max_tokens=max_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
    if recortados:
        return recortados
    ultimo_humano = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    return messages[ultimo_humano:]


def build_graph(
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    max_tokens_historial: int = MAX_TOKENS_HISTORIAL,
) -> CompiledStateGraph:
    """Arma y compila el grafo ReAct. Sin checkpointer el agente no recuerda entre invocaciones."""
    llm_con_tools = model.bind_tools(TOOLS)

    async def agente(state: AgentState) -> dict[str, Any]:
        historial = recortar_historial(state["messages"], max_tokens_historial)
        respuesta = await llm_con_tools.ainvoke([SystemMessage(SYSTEM_PROMPT), *historial])
        return {
            "messages": [respuesta],
            "herramientas_usadas": [call["name"] for call in getattr(respuesta, "tool_calls", [])],
        }

    builder = StateGraph(AgentState)
    builder.add_node("agente", agente)
    # handle_tool_errors=True: si una herramienta igual levanta una excepción (por
    # ejemplo, el modelo manda un argumento con el tipo equivocado), el error vuelve
    # como ToolMessage y el modelo puede corregirse en vez de romper el grafo.
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_edge(START, "agente")
    builder.add_conditional_edges("agente", tools_condition)  # -> "tools" o END
    builder.add_edge("tools", "agente")
    return builder.compile(checkpointer=checkpointer)


def config_para(thread_id: str, recursion_limit: int = RECURSION_LIMIT) -> RunnableConfig:
    """Config de una invocación: el thread_id identifica la conversación en el checkpointer."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
