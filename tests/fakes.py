"""Modelo de chat falso para probar el grafo sin red ni keys.

Devuelve, en orden, los AIMessage que se le pasan (pueden traer tool_calls) y guarda
los mensajes que recibió en cada llamada, para verificar qué vio el modelo (memoria,
errores de herramientas, recorte del historial).
"""

import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def llamada(nombre: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """AIMessage que pide una herramienta."""
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": args, "id": call_id, "type": "tool_call"}])


class FakeToolModel(BaseChatModel):
    """Responde con `respuestas` en orden. Con `repetir_ultima=True` repite la última para siempre."""

    respuestas: list[AIMessage]
    repetir_ultima: bool = False
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)
    herramientas_vinculadas: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolModel":
        self.herramientas_vinculadas = [t.name for t in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.recibidos.append(list(messages))
        indice = len(self.recibidos) - 1
        if indice >= len(self.respuestas):
            if not self.repetir_ultima:
                raise AssertionError(f"El modelo fake recibió una llamada de más (#{indice + 1}).")
            indice = len(self.respuestas) - 1
        # copia con id único: add_messages REEMPLAZA un mensaje si llega otro con el mismo id
        respuesta = self.respuestas[indice].model_copy(update={"id": f"fake-{uuid.uuid4()}"})
        return ChatResult(generations=[ChatGeneration(message=respuesta)])
