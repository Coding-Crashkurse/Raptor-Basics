"""
RAPTOR summary demo server.

This FastAPI app exposes an A2A agent that serves the latest root summary generated
in raptor_pipeline.ipynb. The AgentCard description is pulled from
artifacts/root_summary.txt so remote clients always see the up-to-date synopsis.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import RequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    Artifact,
    MessageSendParams,
    Task,
    TaskArtifactUpdateEvent,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task


ROOT_SUMMARY_PATH = Path("artifacts/root_summary.txt")


def load_root_summary() -> str:
    """Load the persisted RAPTOR root summary for use in the AgentCard description."""
    if ROOT_SUMMARY_PATH.exists():
        text = ROOT_SUMMARY_PATH.read_text(encoding="utf-8").strip()
        if text:
            return text
    return (
        "RAPTOR root summary not found. Run raptor_pipeline.ipynb and execute the root-summary "
        "persistence cell to generate artifacts/root_summary.txt before starting the A2A server."
    )


class AgentExecutor:
    """Simulates a short-lived background task that emits generic status updates."""

    def __init__(self, task_store: InMemoryTaskStore):
        self.task_store = task_store

    async def run_summary_process(
        self, task_id: str, user_input: str, update_queue: Optional[asyncio.Queue] = None
    ):
        """Emit a couple of generic task updates and finish quickly.

        The server's focus is on demonstrating how the AgentCard picks up the persisted
        root summary, so this background worker just simulates a short-running task.
        """

        async def update_status(new_state: TaskState, message_text: str | None = None) -> TaskStatus:
            task = await self.task_store.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found while updating status.")
            task.status.state = new_state
            task.status.message = (
                new_agent_text_message(message_text) if message_text else None
            )
            await self.task_store.save(task)
            return task.status

        async def notify(event):
            if update_queue:
                await update_queue.put(event)

        try:
            status = await update_status(TaskState.working, "Processing your request…")
            await notify(
                TaskStatusUpdateEvent(
                    taskId=task_id,
                    contextId=(await self.task_store.get(task_id)).contextId,
                    status=status,
                    final=False,
                )
            )
            await asyncio.sleep(1)

            task = await self.task_store.get(task_id)
            task.artifacts = [
                Artifact(
                    artifactId=f"artifact-{uuid4().hex}",
                    parts=[TextPart(text="Dummy artifact produced for demo purposes.")],
                )
            ]
            await notify(
                TaskArtifactUpdateEvent(
                    taskId=task_id,
                    contextId=task.contextId,
                    artifact=task.artifacts[0],
                    lastChunk=True,
                )
            )

            status = await update_status(TaskState.completed, "Done.")
            await notify(
                TaskStatusUpdateEvent(
                    taskId=task_id,
                    contextId=(await self.task_store.get(task_id)).contextId,
                    status=status,
                    final=True,
                )
            )
        finally:
            if update_queue:
                await update_queue.put(None)


class RaptorAgentHandler(RequestHandler):
    """Implements the JSON-RPC handlers wired into A2AStarletteApplication."""

    def __init__(self, task_store: InMemoryTaskStore, agent_executor: AgentExecutor):
        self.task_store = task_store
        self.agent_executor = agent_executor

    async def on_message_send(self, params: MessageSendParams, context=None) -> Task:
        task = new_task(request=params.message)
        await self.task_store.save(task)
        print(f"[polling] Accepted task {task.id}. Starting background executor.")
        user_input = params.message.parts[0].root.text.lower() if params.message.parts else ""
        asyncio.create_task(
            self.agent_executor.run_summary_process(task_id=task.id, user_input=user_input)
        )
        return task

    async def on_get_task(self, params: TaskQueryParams, context=None) -> Task:
        print(f"📬 Polling Handler: Received get_task request for {params.id}")
        task = await self.task_store.get(params.id)
        if not task:
            raise ValueError(f"Task with ID {params.id} not found.")
        return task

    async def on_message_send_stream(self, params: MessageSendParams, context=None):
        task = new_task(request=params.message)
        await self.task_store.save(task)
        user_input = params.message.parts[0].root.text.lower() if params.message.parts else ""
        print(f"[streaming] Accepted task {task.id}. Starting background executor.")
        update_queue: asyncio.Queue = asyncio.Queue()
        asyncio.create_task(
            self.agent_executor.run_summary_process(
                task_id=task.id, user_input=user_input, update_queue=update_queue
            )
        )
        yield task
        while True:
            update = await update_queue.get()
            if update is None:
                break
            yield update

    async def on_cancel_task(self, params, context=None):
        raise NotImplementedError("Not implemented.")

    async def on_resubscribe_to_task(self, params, context=None):
        raise NotImplementedError("Not implemented.")

    async def on_set_task_push_notification_config(self, params, context=None):
        raise NotImplementedError("Not implemented.")

    async def on_get_task_push_notification_config(self, params, context=None):
        raise NotImplementedError("Not implemented.")

    async def on_list_task_push_notification_config(
        self, params, context=None
    ) -> list[TaskPushNotificationConfig]:
        raise NotImplementedError("Not implemented.")

    async def on_delete_task_push_notification_config(self, params, context=None) -> None:
        raise NotImplementedError("Not implemented.")


def build_app() -> FastAPI:
    task_store = InMemoryTaskStore()
    agent_executor = AgentExecutor(task_store)

    skill = AgentSkill(
        id="raptor-summary-service",
        name="RAPTOR Summary Service",
        description="Returns concise summaries derived from the latest RAPTOR/adRAP hierarchy and emits basic status events.",
        tags=["summary", "retrieval", "knowledge"],
        examples=[],
    )

    card = AgentCard(
        name="RAPTOR Summary Agent",
        description=load_root_summary(),
        url="http://localhost:8002/",
        version="4.0",
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[skill],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
    )

    http_handler = RaptorAgentHandler(task_store, agent_executor)

    a2a_app = A2AStarletteApplication(agent_card=card, http_handler=http_handler).build()
    api = FastAPI(title="RAPTOR Summary A2A Server")
    api.mount("/", a2a_app)
    return api


app = build_app()


if __name__ == "__main__":
    print("🧠 Starting RAPTOR Summary Server on http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
