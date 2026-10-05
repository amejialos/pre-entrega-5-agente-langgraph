# Agente de razonamiento cíclico con memoria persistente

Agente ReAct en Python 3.12 + **LangGraph** que atiende consultas sobre clientes y
pedidos de una tienda de tecnología. El modelo decide solo qué herramienta usar y
cuántas veces (sin if/else nuestros), encadena llamadas hasta tener la respuesta,
reintenta o pide una aclaración cuando una herramienta devuelve un error, y recuerda la
conversación de cada `thread_id` en un archivo **SQLite**, incluso después de reiniciar
el proceso.

Pre-entrega 5 del curso AI Engineering: "Agente de razonamiento cíclico con memoria persistente".

## Estructura

```
database.py     # base de datos simulada: 6 clientes y 9 pedidos en memoria (dataclasses)
tools.py        # @tool buscar_cliente, buscar_pedidos, detalle_pedido (docstrings largos, errores en JSON)
providers.py    # build_model(): ChatOpenAI (OpenAI o Gemini compatible) + limitador de RPM opcional
agent.py        # AgentState(MessagesState), build_graph(), recortar_historial(), config_para()
traza.py        # ejecutar_turno() con astream: registra cada paso ReAct y lo vuelca a .json y .log
demo.py         # corre 5 escenarios contra el modelo real y genera la traza
trazas/         # traza_ejecucion.json y traza_ejecucion.log de una corrida REAL
tests/          # pytest sin red: modelo fake que emite tool_calls, checkpointer en tmp_path
.env.example    # variables de entorno
```

## El grafo

```
START -> agente --tools_condition--> tools -> agente -> ... -> END
                         \-- (sin tool_calls) --> END
```

| Pieza | Dónde | Qué hace |
|---|---|---|
| `AgentState(MessagesState)` | `agent.py` | Hereda `messages` (reducer `add_messages`) y agrega `herramientas_usadas: Annotated[list[str], operator.add]` |
| Nodo `agente` | `agent.py` | Recorta el historial con `trim_messages`, antepone el system prompt y llama a `llm.bind_tools(TOOLS)` |
| Nodo `tools` | `agent.py` | `ToolNode(TOOLS, handle_tool_errors=True)`: ejecuta las tool calls y devuelve `ToolMessage`s |
| `tools_condition` | `agent.py` | Arista condicional: si el último mensaje tiene `tool_calls` va a `tools`, si no termina |
| Checkpointer | `demo.py` | `AsyncSqliteSaver` sobre `checkpoints.sqlite`; la conversación se identifica con `thread_id` |
| `recursion_limit=10` | `agent.py` (`config_para`) | Cada invocación lo lleva; un ciclo que no converge termina en `GraphRecursionError` |
| `RetryPolicy` del nodo `agente` | `agent.py` (`RETRY_MODELO`) | Reintenta la llamada al modelo ante 429/5xx con esperas de 20, 40 y 80 s; 400/401 no se reintentan |

**Por qué `AsyncSqliteSaver`.** La consigna pide `SqliteSaver`. Todo el agente es
asíncrono (`ainvoke`/`astream`), y `SqliteSaver` es síncrono: no implementa los
métodos `aget_tuple`/`aput` que usa el grafo en modo async. `AsyncSqliteSaver`
(paquete `langgraph-checkpoint-sqlite`, sobre `aiosqlite`) es **la variante async de
`SqliteSaver`**: mismo formato de tablas, mismo archivo `.sqlite`, misma semántica de
`thread_id`.

### Los criterios de la consigna

- **Autonomía.** El grafo no tiene ninguna regla del tipo "si la pregunta dice X, llamá
  a Y". El modelo recibe las tres herramientas con `bind_tools` y sus docstrings, y
  elige. `tools_condition` solo mira si el modelo pidió herramientas o no.
- **Ciclo de retorno.** Las herramientas nunca levantan excepciones por datos
  inválidos: devuelven `{"error": ..., "sugerencia": ...}`. Ese JSON vuelve al modelo
  como observación y el modelo decide: reintentar con otros argumentos (cliente no
  encontrado -> probar solo con el apellido) o preguntarle al usuario (nombre ambiguo
  -> mostrar las opciones). Si igual hay una excepción (por ejemplo, un argumento con
  el tipo equivocado), `handle_tool_errors=True` la convierte en un `ToolMessage` de
  error en lugar de cortar el grafo.
- **Resiliencia de estado.** El checkpointer guarda el estado después de cada paso.
  Con el mismo `thread_id`, una pregunta como "¿y el último?" se resuelve con el
  contexto previo. La demo lo prueba cerrando la conexión a SQLite y abriendo otra
  (grafo nuevo) sobre el mismo archivo.
- **Estado sucio.** El estado guarda la conversación completa, pero antes de cada
  llamada al modelo `recortar_historial` aplica `trim_messages` (estrategia `last`,
  ~4000 tokens aproximados, empezando siempre en un mensaje del usuario para no dejar
  un `ToolMessage` huérfano). Si el turno actual solo ya supera el presupuesto, se manda
  completo: cortarlo rompería el par tool call / resultado.

## Las herramientas

| Herramienta | Entrada | Devuelve | Errores |
|---|---|---|---|
| `buscar_cliente` | `nombre` | `cliente_id`, nombre, ciudad | no encontrado (sugiere reintentar con el apellido), ambiguo (lista de coincidencias), vacío |
| `buscar_pedidos` | `cliente_id` | cantidad, total (sin cancelados), pedidos con fecha/estado/monto, `ultimo_pedido_id` | id inválido, cliente inexistente; sin pedidos devuelve una nota |
| `detalle_pedido` | `pedido_id` | productos, cantidades, precios, subtotales, total | pedido inexistente (avisa si parece un `cliente_id`) |

Como `buscar_pedidos` necesita un id y el usuario habla con nombres, una pregunta como
"¿cuánto gastó Ana Gómez?" obliga a dos llamadas: `buscar_cliente` -> `buscar_pedidos`.

Datos de ejemplo: Juan Pérez (101), Ana Gómez (102, 3 pedidos, $14.500), Lucía Martínez
(103), Pablo Martínez (104, un pedido cancelado), Carlos Ruiz (105), Sofía Herrera (106,
sin pedidos).

## Instalación

Requiere Python 3.12 o superior.

**Con `uv` (recomendado):**

```bash
uv sync
```

**Con `venv` clásico:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install langgraph langgraph-checkpoint-sqlite aiosqlite langchain-core langchain-openai python-dotenv
pip install pytest pytest-asyncio
```

## Variables de entorno

Copiá `.env.example` a `.env` y completalo. El `.env` está en `.gitignore`: nunca se sube.

| Variable | Obligatoria | Default | Para qué |
|---|---|---|---|
| `OPENAI_API_KEY` | Sí | | Key de OpenAI (o de Gemini, ver abajo) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Modelo con soporte de tool calling |
| `OPENAI_BASE_URL` | No | | Endpoint alternativo compatible con la API de OpenAI |
| `LLM_RPM` | No | sin límite | Máximo de pedidos por minuto al modelo (útil con el free tier de Gemini) |

### Probar gratis con Gemini

Google expone un endpoint compatible con la API de OpenAI y una key gratuita en
[Google AI Studio](https://aistudio.google.com/apikey):

```bash
OPENAI_API_KEY=<tu key de Gemini>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-3.7-flash
LLM_RPM=2
```

Dos detalles que aparecieron en la prueba real:

1. **Thought signatures.** Los Gemini con razonamiento devuelven en cada tool call un
   campo `extra_content.google.thought_signature` y lo exigen de vuelta en el turno
   siguiente; si falta responden `400 Function call is missing a thought_signature`.
   `ChatOpenAI` descarta ese campo, así que `providers.ChatOpenAICompatible` lo guarda en
   el `AIMessage` y lo reinyecta al armar el pedido. Con OpenAI el campo no existe y no
   cambia nada.
2. **Cuotas del free tier.** ~5 pedidos por minuto y ~20 por día **por modelo**. Un
   agente hace varios pedidos por pregunta (uno por cada vuelta del ciclo), así que la
   demo completa (~15 pedidos) necesita `LLM_RPM=2` o `3` y puede agotar la cuota diaria
   de un modelo; en ese caso cambiá `OPENAI_MODEL` por otro Gemini. Si igual llega un
   429, el `RetryPolicy` del nodo `agente` espera y reintenta. Los reintentos rápidos del
   SDK de OpenAI están apagados (`max_retries=0`): contra un 429 solo gastaban más cuota.

## Uso

```bash
uv run python demo.py
```

Corre cinco escenarios y escribe `trazas/traza_ejecucion.json` y `trazas/traza_ejecucion.log`:

1. **Multi-paso** (`sesion-ana`): "¿Cuántos pedidos hizo Ana Gómez y cuánto gastó en total?"
2. **Memoria persistente** (`sesion-ana`, tras cerrar y reabrir el SQLite): "¿Y cuál fue el último? ¿Qué compró en ese pedido?"
3. **Error -> reintento** (`sesion-ruiz`): "¿Cuánto gastó en total Carlos Ruiz Díaz?" (el cliente se llama "Carlos Ruiz")
4. **Ambigüedad -> aclaración** (`sesion-martinez`): "¿Cuánto gastó en total el cliente Martínez?" y luego "Me refiero a Lucía."
5. **Aislamiento** (`sesion-nueva`): "¿Y cuál fue el último pedido?" sin contexto previo

Para usar el agente desde código:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent import build_graph, config_para
from providers import build_model

async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    graph = build_graph(build_model(), checkpointer)
    result = await graph.ainvoke({"messages": [("user", "¿Cuánto gastó Ana Gómez?")]}, config_para("mi-sesion"))
    print(result["messages"][-1].content)
```

## Traza real

Corrida real del 2026-10-05 con `gemini-3.8-flash` por el endpoint compatible de
Gemini (`OPENAI_MODEL=gemini-3.8-flash LLM_RPM=3 uv run python demo.py`). Completa en
[`trazas/traza_ejecucion.json`](trazas/traza_ejecucion.json) (cada paso con nodo,
tool calls y argumentos, resultados y respuesta final) y en
[`trazas/traza_ejecucion.log`](trazas/traza_ejecucion.log). Extracto (resultados acortados):

```
=== Escenario 1: Multi-paso: el agente encadena dos herramientas ===
[thread_id=sesion-ana] Usuario: ¿Cuántos pedidos hizo Ana Gómez y cuánto gastó en total?
   1. (agente) Acción: buscar_cliente({"nombre": "Ana Gómez"})
   2. (tools) Observación [buscar_cliente]: {"cliente_id": 102, "nombre": "Ana Gómez", "ciudad": "Rosario"}
   3. (agente) Acción: buscar_pedidos({"cliente_id": 102})
   4. (tools) Observación [buscar_pedidos]: {"cliente_id": 102, "cantidad_pedidos": 3, "total": 14500, ..., "ultimo_pedido_id": 1007}
   5. (agente) Respuesta final: Ana Gómez hizo 3 pedidos y gastó en total $14.500.
  -> 2 llamadas a herramientas

=== Escenario 2: Memoria persistente: mismo thread_id tras reabrir el SQLite ===
# Checkpointer y grafo recreados desde checkpoints.sqlite: el thread 'sesion-ana' trae 6 mensajes guardados en la fase anterior.
[thread_id=sesion-ana] Usuario: ¿Y cuál fue el último? ¿Qué compró en ese pedido?
   1. (agente) Acción: detalle_pedido({"pedido_id": 1007})
   2. (tools) Observación [detalle_pedido]: {"pedido_id": 1007, "cliente": "Ana Gómez", "estado": "en camino", "items": [{"producto": "Auriculares con micrófono", ...}], "total": 6000}
   3. (agente) Respuesta final: El último pedido fue el #1007 (del 20/09/2026, actualmente en camino). Compró: 1x Auriculares con micrófono ($6.000) ...

=== Escenario 3: Ciclo de retorno: error de la herramienta -> reintento ===
[thread_id=sesion-ruiz] Usuario: ¿Cuánto gastó en total Carlos Ruiz Díaz?
   1. (agente) Acción: buscar_cliente({"nombre": "Carlos Ruiz Díaz"})
   2. (tools) Observación [buscar_cliente]: {"error": "cliente no encontrado: ninguno coincide con 'Carlos Ruiz Díaz'", "sugerencia": "Reintentá con una sola palabra ..."}
   3. (agente) Acción: buscar_cliente({"nombre": "Ruiz"})
   4. (tools) Observación [buscar_cliente]: {"cliente_id": 105, "nombre": "Carlos Ruiz", "ciudad": "La Plata"}
   5. (agente) Acción: buscar_pedidos({"cliente_id": 105})
   6. (tools) Observación [buscar_pedidos]: {"cliente_id": 105, "cantidad_pedidos": 1, "total": 15500, ...}
   7. (agente) Respuesta final: Carlos Ruiz gastó en total $15.500 (tiene un único pedido registrado por ese monto).

=== Escenario 4: Ciclo de retorno: nombre ambiguo -> pide aclaración ===
[thread_id=sesion-martinez] Usuario: ¿Cuánto gastó en total el cliente Martínez?
   1. (agente) Acción: buscar_cliente({"nombre": "Martínez"})
   2. (tools) Observación [buscar_cliente]: {"error": "nombre ambiguo: 2 clientes coinciden con 'Martínez'", "coincidencias": [...]}
   3. (agente) Respuesta final: Encontré dos clientes con el apellido Martínez: Lucía Martínez (Mendoza) y Pablo Martínez (Buenos Aires). ¿A cuál de los dos te referís?
[thread_id=sesion-martinez] Usuario: Me refiero a Lucía.
   1. (agente) Acción: buscar_pedidos({"cliente_id": 103})
   2. (tools) Observación [buscar_pedidos]: {"cliente_id": 103, "cantidad_pedidos": 2, "total": 18400, ...}
   3. (agente) Respuesta final: Lucía Martínez gastó en total $18.400 (en 2 pedidos).

=== Escenario 5: Aislamiento: otro thread_id no comparte memoria ===
[thread_id=sesion-nueva] Usuario: ¿Y cuál fue el último pedido?
   1. (agente) Respuesta final: ¿De qué cliente estamos hablando? Decime su nombre o apellido así te busco los pedidos.
```

Qué muestra:

- **Multi-paso:** el escenario 1 llama a dos herramientas distintas y el 3 a tres (una
  de ellas, reintentada) antes de responder.
- **Memoria real:** en el escenario 2 el proceso ya cerró la conexión de la fase 1; el
  grafo nuevo lee del `.sqlite` los 6 mensajes previos y el modelo va directo a
  `detalle_pedido(1007)` usando el `ultimo_pedido_id` que había visto antes, sin volver a
  buscar al cliente. En el escenario 4, "Me refiero a Lucía" se resuelve con el
  `cliente_id` 103 de la lista de coincidencias del turno anterior.
- **Aislamiento:** el escenario 5 hace la misma pregunta que el 2 en un thread nuevo y
  el agente pide el dato que le falta.

Las duraciones que figuran en la traza (~60 a 85 s por turno) son casi todas espera del
limitador de RPM y de los reintentos por 429 del free tier, no del agente.

## Tests

```bash
uv run pytest -q
```

Sin red ni keys: `tests/fakes.py` define un `BaseChatModel` fake que soporta
`bind_tools`, devuelve `AIMessage`s con `tool_calls` en el orden que se le indica y
guarda lo que recibió en cada llamada. Cubren:

- herramientas: casos ok, tildes/mayúsculas, no encontrado, ambiguo, ids inválidos, sin pedidos, cancelados, docstrings con descripción de cada argumento;
- grafo: dos herramientas encadenadas y fin por `tools_condition`; el error de una herramienta llega al modelo y reintenta; una excepción en una herramienta no rompe el grafo;
- `recursion_limit`: un modelo que pide herramientas para siempre termina en `GraphRecursionError`;
- reintentos: un 429 del modelo se reintenta en el nodo; un 401 no;
- memoria: `AsyncSqliteSaver` en `tmp_path`, cerrar y reabrir la conexión mantiene la conversación; dos `thread_id` no se mezclan;
- recorte: `trim_messages` conserva lo último, arranca en un mensaje del usuario y nunca deja afuera el turno actual; el estado guarda todo aunque el modelo reciba menos;
- proveedor: lectura del entorno, limitador de RPM, ida y vuelta de la thought signature;
- traza: etiquetas ReAct, sin `additional_kwargs` ni `response_metadata`, escritura de `.json` y `.log`.
