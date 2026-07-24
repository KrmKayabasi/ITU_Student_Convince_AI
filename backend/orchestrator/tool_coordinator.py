"""Per-session execution and cancellation for Gemini Live tool calls."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from professor_search import ItuProfessorSearch

logger = logging.getLogger("orchestrator.tools")


class ToolCoordinator:
    def __init__(
        self,
        *,
        bridge: Any,
        send_json: Callable[[dict[str, Any]], Awaitable[None]],
        professor_search: ItuProfessorSearch,
    ) -> None:
        self.bridge = bridge
        self.send_json = send_json
        self.professor_search = professor_search
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    def start(self, call: dict[str, Any]) -> None:
        call_id = str(call.get("id") or "")
        if self._closed or not call_id or call_id in self._tasks:
            return
        task = asyncio.create_task(self._run(call), name=f"tool:{call_id}")
        self._tasks[call_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(call_id, None))

    async def cancel(self, call_ids: list[str]) -> None:
        for call_id in call_ids:
            task = self._tasks.get(call_id)
            if task is not None and not task.done():
                task.cancel()
            await self.send_json(
                {"type": "tool_activity", "id": call_id, "status": "cancelled"}
            )

    async def _run(self, call: dict[str, Any]) -> None:
        call_id = str(call["id"])
        name = str(call.get("name") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        topic = str(args.get("topic") or "").strip()
        department = str(args.get("department") or "").strip()
        related_terms: list[str] = (
            [str(t).strip() for t in args["related_terms"] if str(t).strip()]
            if isinstance(args.get("related_terms"), list)
            else []
        )
        await self.send_json(
            {
                "type": "tool_activity",
                "id": call_id,
                "name": name,
                "status": "searching",
                "query": topic,
            }
        )
        try:
            if name != "search_itu_professors":
                raise ValueError(f"Unsupported tool: {name}")
            result = await self.professor_search.search(
                topic, department, related_terms
            )
            await self.send_json({"type": "tool_result", "id": call_id, **result})
            await self.bridge.send_tool_response(
                call_id=call_id, name=name, response=result
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            message = "İTÜ Akademi araması şu anda tamamlanamadı."
            await self.send_json(
                {
                    "type": "tool_activity",
                    "id": call_id,
                    "name": name,
                    "status": "error",
                    "query": topic,
                    "message": message,
                }
            )
            await self.bridge.send_tool_response(
                call_id=call_id,
                name=name,
                response={"error": message, "query": topic, "results": []},
            )

    async def close(self) -> None:
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
