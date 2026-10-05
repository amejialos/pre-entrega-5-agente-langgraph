import json

from tools import TOOLS, buscar_cliente, buscar_pedidos, detalle_pedido


def _call(tool, **args):
    return json.loads(tool.invoke(args))


def test_las_herramientas_tienen_docstrings_descriptivos():
    for t in TOOLS:
        assert len(t.description) > 200, t.name
        for nombre, schema in t.args.items():
            assert schema.get("description"), f"{t.name}.{nombre} sin descripción"


def test_buscar_cliente_encuentra_ignorando_tildes_y_mayusculas():
    assert _call(buscar_cliente, nombre="ana gomez") == {"cliente_id": 102, "nombre": "Ana Gómez", "ciudad": "Rosario"}
    assert _call(buscar_cliente, nombre="Gómez")["cliente_id"] == 102


def test_buscar_cliente_no_encontrado_sugiere_reintentar():
    r = _call(buscar_cliente, nombre="Carlos Ruiz Díaz")
    assert r["error"].startswith("cliente no encontrado")
    assert "apellido" in r["sugerencia"]
    assert _call(buscar_cliente, nombre="Ruiz")["cliente_id"] == 105


def test_buscar_cliente_ambiguo_devuelve_las_opciones():
    r = _call(buscar_cliente, nombre="Martínez")
    assert r["error"].startswith("nombre ambiguo")
    assert {c["cliente_id"] for c in r["coincidencias"]} == {103, 104}
    assert "preguntale" in r["sugerencia"].lower()


def test_buscar_cliente_vacio():
    assert _call(buscar_cliente, nombre="   ")["error"] == "nombre vacío"


def test_buscar_pedidos_ok():
    r = _call(buscar_pedidos, cliente_id=102)
    assert r["cantidad_pedidos"] == 3
    assert r["total"] == 14500
    assert [p["pedido_id"] for p in r["pedidos"]] == [1001, 1004, 1007]
    assert r["ultimo_pedido_id"] == 1007


def test_buscar_pedidos_no_suma_cancelados():
    r = _call(buscar_pedidos, cliente_id=104)
    assert r["cantidad_pedidos"] == 1
    assert r["pedidos"][0]["estado"] == "cancelado"
    assert r["total"] == 0


def test_buscar_pedidos_cliente_sin_pedidos():
    r = _call(buscar_pedidos, cliente_id=106)
    assert r["cantidad_pedidos"] == 0 and r["ultimo_pedido_id"] is None
    assert "nota" in r


def test_buscar_pedidos_errores():
    assert _call(buscar_pedidos, cliente_id=999)["error"] == "no existe un cliente con id 999"
    assert _call(buscar_pedidos, cliente_id=0)["error"].startswith("cliente_id inválido")


def test_detalle_pedido_ok():
    r = _call(detalle_pedido, pedido_id=1009)
    assert r["cliente"] == "Carlos Ruiz"
    assert r["items"][1] == {"producto": "Memoria USB 64 GB", "cantidad": 3, "precio_unitario": 1500, "subtotal": 4500}
    assert r["total"] == 15500


def test_detalle_pedido_inexistente_avisa_posible_confusion_de_ids():
    r = _call(detalle_pedido, pedido_id=102)
    assert r["error"] == "no existe un pedido con id 102"
    assert "cliente_id" in r["sugerencia"]
