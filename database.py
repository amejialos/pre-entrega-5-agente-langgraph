"""Base de datos simulada de una tienda de tecnología: clientes y pedidos.

Son datos fijos en memoria (diccionarios), así las herramientas son deterministas y
los tests no dependen de nada externo. Las funciones de consulta imitan lo que haría
una capa de acceso a datos real: buscar por nombre, listar por cliente, traer detalle.
Los montos están en pesos enteros.
"""

import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    producto: str
    cantidad: int
    precio_unitario: int

    @property
    def subtotal(self) -> int:
        return self.cantidad * self.precio_unitario


@dataclass(frozen=True)
class Pedido:
    id: int
    cliente_id: int
    fecha: str  # ISO 8601 (AAAA-MM-DD): se ordena bien como texto
    estado: str
    items: tuple[Item, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return sum(item.subtotal for item in self.items)


@dataclass(frozen=True)
class Cliente:
    id: int
    nombre: str
    ciudad: str


CLIENTES: dict[int, Cliente] = {
    c.id: c
    for c in (
        Cliente(101, "Juan Pérez", "Córdoba"),
        Cliente(102, "Ana Gómez", "Rosario"),
        Cliente(103, "Lucía Martínez", "Mendoza"),
        Cliente(104, "Pablo Martínez", "Buenos Aires"),
        Cliente(105, "Carlos Ruiz", "La Plata"),
        Cliente(106, "Sofía Herrera", "Salta"),  # sin pedidos: caso de información vacía
    )
}

PEDIDOS: dict[int, Pedido] = {
    p.id: p
    for p in (
        Pedido(1001, 102, "2026-07-03", "entregado", (Item("Teclado mecánico", 1, 5000),)),
        Pedido(1002, 101, "2026-07-10", "entregado", (Item("Monitor 27 pulgadas", 1, 42000),)),
        Pedido(1003, 103, "2026-07-18", "entregado", (Item("Soporte para notebook", 1, 8500),)),
        Pedido(
            1004,
            102,
            "2026-08-15",
            "entregado",
            (Item("Mouse inalámbrico", 1, 2000), Item("Pad para mouse", 1, 1500)),
        ),
        Pedido(1005, 101, "2026-08-22", "entregado", (Item("Cable HDMI 2 m", 2, 1200),)),
        Pedido(1006, 104, "2026-09-02", "cancelado", (Item("Silla ergonómica", 1, 65000),)),
        Pedido(1007, 102, "2026-09-20", "en camino", (Item("Auriculares con micrófono", 1, 6000),)),
        Pedido(1008, 103, "2026-09-28", "en preparación", (Item("Webcam Full HD", 1, 9900),)),
        Pedido(
            1009,
            105,
            "2026-09-30",
            "en camino",
            (Item("Disco SSD 1 TB", 1, 11000), Item("Memoria USB 64 GB", 3, 1500)),
        ),
    )
}


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes: "Gómez" y "gomez" tienen que coincidir."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return " ".join(sin_tildes.lower().split())


def buscar_clientes_por_nombre(nombre: str) -> list[Cliente]:
    """Clientes cuyo nombre contiene TODAS las palabras buscadas (sin importar tildes ni mayúsculas)."""
    palabras = normalizar(nombre).split()
    if not palabras:
        return []
    return [c for c in CLIENTES.values() if all(p in normalizar(c.nombre).split() for p in palabras)]


def pedidos_de_cliente(cliente_id: int) -> list[Pedido]:
    """Pedidos del cliente ordenados del más viejo al más nuevo."""
    return sorted((p for p in PEDIDOS.values() if p.cliente_id == cliente_id), key=lambda p: p.fecha)
