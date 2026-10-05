"""Herramientas del agente: consultas a la base simulada de clientes y pedidos.

El docstring de cada herramienta es lo que el modelo lee para decidir cuándo usarla
y con qué argumentos (`parse_docstring=True` además pasa la sección Args a la
descripción de cada parámetro). Por eso son largos y explícitos.

Contrato de salida: todas devuelven un JSON en texto. Si algo sale mal NO levantan
una excepción: devuelven `{"error": ..., "sugerencia": ...}`. Así el error vuelve al
modelo como una observación más (un ToolMessage) y el agente puede reintentar o pedir
una aclaración, en vez de que se corte el grafo. Ese es el "ciclo de retorno".
"""

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from database import CLIENTES, PEDIDOS, buscar_clientes_por_nombre, pedidos_de_cliente

ESTADO_CANCELADO = "cancelado"


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _error(mensaje: str, sugerencia: str, **extra: Any) -> str:
    return _json({"error": mensaje, "sugerencia": sugerencia, **extra})


@tool(parse_docstring=True)
def buscar_cliente(nombre: str) -> str:
    """Busca clientes de la tienda por nombre y devuelve su id (cliente_id).

    Usala SIEMPRE que el usuario mencione a un cliente por su nombre o apellido:
    las otras herramientas necesitan el cliente_id numérico y nunca hay que inventarlo.
    La búsqueda ignora tildes y mayúsculas, y exige que TODAS las palabras buscadas
    aparezcan en el nombre del cliente ("Gómez" encuentra a "Ana Gómez", pero
    "Ana María Gómez" no).
    Respuestas posibles (JSON):
    - una coincidencia: {"cliente_id", "nombre", "ciudad"}.
    - varias coincidencias: {"error": "nombre ambiguo", "coincidencias": [...]}.
      No elijas vos: preguntale al usuario a cuál se refiere, mostrándole las opciones.
    - ninguna: {"error": "cliente no encontrado", "sugerencia": ...}. Reintentá con
      menos palabras (solo el apellido o solo el nombre) antes de rendirte.

    Args:
        nombre: Nombre, apellido o nombre completo del cliente, tal como lo escribió el usuario. Ejemplo: "Ana Gómez".
    """
    if not nombre.strip():
        return _error("nombre vacío", "Pedile al usuario el nombre o el apellido del cliente.")
    encontrados = buscar_clientes_por_nombre(nombre)
    if not encontrados:
        return _error(
            f"cliente no encontrado: ninguno coincide con '{nombre}'",
            "Reintentá con una sola palabra (solo el apellido o solo el nombre). "
            "Si tampoco aparece, avisale al usuario que no existe y pedile que revise el nombre.",
        )
    if len(encontrados) > 1:
        return _error(
            f"nombre ambiguo: {len(encontrados)} clientes coinciden con '{nombre}'",
            "No elijas uno al azar: preguntale al usuario a cuál de estos clientes se refiere.",
            coincidencias=[{"cliente_id": c.id, "nombre": c.nombre, "ciudad": c.ciudad} for c in encontrados],
        )
    cliente = encontrados[0]
    return _json({"cliente_id": cliente.id, "nombre": cliente.nombre, "ciudad": cliente.ciudad})


@tool(parse_docstring=True)
def buscar_pedidos(cliente_id: int) -> str:
    """Lista los pedidos de un cliente: cantidad, total gastado, y fecha, estado y monto de cada pedido.

    Usala para preguntas como "¿cuántos pedidos hizo?", "¿cuánto gastó en total?" o
    "¿cuál fue su último pedido?". Necesita el cliente_id numérico: si el usuario dio
    un nombre, primero llamá a buscar_cliente para obtenerlo.
    Los pedidos vienen ordenados del más viejo al más nuevo; "ultimo_pedido_id" es el
    más reciente. "total" suma todos los pedidos que NO están cancelados (los
    cancelados se listan igual, con su estado, pero no cuentan en el total).
    Esta herramienta NO trae los productos de cada pedido: para eso usá detalle_pedido.
    Si el cliente no existe devuelve {"error": ..., "sugerencia": ...}.

    Args:
        cliente_id: Id numérico del cliente (por ejemplo 102). Se obtiene con buscar_cliente.
    """
    if cliente_id <= 0:
        return _error(
            f"cliente_id inválido: {cliente_id}",
            "El id es un entero positivo. Si tenés el nombre, usá buscar_cliente para obtenerlo.",
        )
    cliente = CLIENTES.get(cliente_id)
    if cliente is None:
        return _error(
            f"no existe un cliente con id {cliente_id}",
            "Verificá el id con buscar_cliente o pedile al usuario el nombre del cliente.",
        )
    pedidos = pedidos_de_cliente(cliente_id)
    resultado: dict[str, Any] = {
        "cliente_id": cliente.id,
        "nombre": cliente.nombre,
        "cantidad_pedidos": len(pedidos),
        "total": sum(p.total for p in pedidos if p.estado != ESTADO_CANCELADO),
        "pedidos": [{"pedido_id": p.id, "fecha": p.fecha, "estado": p.estado, "monto": p.total} for p in pedidos],
        "ultimo_pedido_id": pedidos[-1].id if pedidos else None,
    }
    if not pedidos:
        resultado["nota"] = "El cliente existe pero todavía no hizo pedidos."
    return _json(resultado)


@tool(parse_docstring=True)
def detalle_pedido(pedido_id: int) -> str:
    """Devuelve el detalle completo de UN pedido: productos, cantidades, precios unitarios, subtotales, total, fecha y estado.

    Usala cuando el usuario pregunte qué compró, qué productos tenía un pedido o
    cuánto costó cada cosa. Necesita el pedido_id (un número de 4 cifras, como 1007),
    que se obtiene con buscar_pedidos. No confundas pedido_id con cliente_id.
    Si el pedido no existe devuelve {"error": ..., "sugerencia": ...}.

    Args:
        pedido_id: Id numérico del pedido (por ejemplo 1007). Se obtiene con buscar_pedidos.
    """
    pedido = PEDIDOS.get(pedido_id)
    if pedido is None:
        return _error(
            f"no existe un pedido con id {pedido_id}",
            "Los pedido_id se obtienen con buscar_pedidos(cliente_id). Revisá que no hayas pasado un cliente_id.",
        )
    cliente = CLIENTES[pedido.cliente_id]
    return _json(
        {
            "pedido_id": pedido.id,
            "cliente_id": cliente.id,
            "cliente": cliente.nombre,
            "fecha": pedido.fecha,
            "estado": pedido.estado,
            "items": [
                {
                    "producto": i.producto,
                    "cantidad": i.cantidad,
                    "precio_unitario": i.precio_unitario,
                    "subtotal": i.subtotal,
                }
                for i in pedido.items
            ],
            "total": pedido.total,
        }
    )


TOOLS: list[BaseTool] = [buscar_cliente, buscar_pedidos, detalle_pedido]
