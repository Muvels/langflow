"""Test OpenAI Responses Error Handling."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langflow.main import create_app


def parse_data_chunks(stream_text: str) -> list[dict]:
    """Extract JSON payloads from SSE data lines."""
    chunks = []
    for line in stream_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.mark.asyncio
async def test_openai_response_stream_error_handling(client):
    """Test that errors during streaming are correctly propagated to the client.

    Ensure errors are propagated as OpenAI-compatible error responses.
    """
    # Mock api_key_security dependency
    from langflow.services.auth.utils import api_key_security
    from langflow.services.database.models.user.model import UserRead

    async def mock_api_key_security():
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        return UserRead(
            id="00000000-0000-0000-0000-000000000000",
            username="testuser",
            is_active=True,
            is_superuser=False,
            create_at=now,
            updated_at=now,
            profile_image=None,
            store_api_key=None,
            last_login_at=None,
            optins=None,
        )

    client.app.dependency_overrides[api_key_security] = mock_api_key_security

    # Mock the flow execution to simulate an error during streaming
    with (
        patch("langflow.api.v1.openai_responses.get_flow_by_id_or_endpoint_name") as mock_get_flow,
        patch("langflow.api.v1.openai_responses.run_flow_generator") as _,
        patch("langflow.api.v1.openai_responses.consume_and_yield") as mock_consume,
    ):
        # Setup mock flow
        mock_flow = MagicMock()
        mock_flow.data = {"nodes": [{"data": {"type": "ChatInput"}}, {"data": {"type": "ChatOutput"}}]}
        mock_get_flow.return_value = mock_flow

        # We need to simulate the event manager queue behavior
        # The run_flow_generator in the actual code puts events into the event_manager
        # which puts them into the queue.

        # Instead of mocking the complex event manager interaction, we can mock
        # consume_and_yield to yield our simulated error event

        # Simulate an error event from the queue
        error_event = json.dumps({"event": "error", "data": {"error": "Simulated streaming error"}}).encode("utf-8")

        # Yield error event then None to end stream
        async def event_generator(*_, **__):
            yield error_event
            yield None

        mock_consume.side_effect = event_generator

        # Make the request
        response = client.post(
            "/api/v1/responses",
            json={"model": "test-flow-id", "input": "test input", "stream": True},
            headers={"Authorization": "Bearer test-key"},
        )

        # Check response
        assert response.status_code == 200
        content = response.content.decode("utf-8")

        # Verify we got the error event in the stream
        assert (
            "event: error" not in content
        )  # OpenAI format doesn't use event: error for the data payload itself usually, but let's check the data

        chunks = parse_data_chunks(content)
        assert any(chunk.get("finish_reason") == "error" for chunk in chunks)
        assert any(chunk.get("status") == "failed" for chunk in chunks)
        assert any(chunk.get("delta", {}).get("content") == "Simulated streaming error" for chunk in chunks)

    # Clean up overrides
    client.app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_openai_response_stream_emits_reasoning_chunks(client):
    """Test that reasoning payloads are preserved in streaming response chunks."""
    from langflow.services.auth.utils import api_key_security
    from langflow.services.database.models.user.model import UserRead

    async def mock_api_key_security():
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        return UserRead(
            id="00000000-0000-0000-0000-000000000000",
            username="testuser",
            is_active=True,
            is_superuser=False,
            create_at=now,
            updated_at=now,
            profile_image=None,
            store_api_key=None,
            last_login_at=None,
            optins=None,
        )

    client.app.dependency_overrides[api_key_security] = mock_api_key_security

    with (
        patch("langflow.api.v1.openai_responses.get_flow_by_id_or_endpoint_name") as mock_get_flow,
        patch("langflow.api.v1.openai_responses.run_flow_generator") as mock_run_flow,
        patch("langflow.api.v1.openai_responses.consume_and_yield") as mock_consume,
    ):
        mock_flow = MagicMock()
        mock_flow.data = {"nodes": [{"data": {"type": "ChatInput"}}, {"data": {"type": "ChatOutput"}}]}
        mock_get_flow.return_value = mock_flow
        mock_run_flow.return_value = None

        token_event = json.dumps(
            {
                "event": "token",
                "data": {"chunk": "Answer token", "reasoning_content": "Reasoning token"},
            }
        ).encode("utf-8")

        async def event_generator(*_, **__):
            yield token_event
            yield None

        mock_consume.side_effect = event_generator

        response = client.post(
            "/api/v1/responses",
            json={"model": "test-flow-id", "input": "test input", "stream": True},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        chunks = parse_data_chunks(response.content.decode("utf-8"))

        assert any(chunk.get("delta", {}).get("reasoning_content") == "Reasoning token" for chunk in chunks)
        assert any(chunk.get("delta", {}).get("content") == "Answer token" for chunk in chunks)

    client.app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_openai_response_stream_emits_tool_start_before_completion(client):
    """Test that tool starts are emitted before tool outputs are available."""
    from langflow.services.auth.utils import api_key_security
    from langflow.services.database.models.user.model import UserRead

    async def mock_api_key_security():
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        return UserRead(
            id="00000000-0000-0000-0000-000000000000",
            username="testuser",
            is_active=True,
            is_superuser=False,
            create_at=now,
            updated_at=now,
            profile_image=None,
            store_api_key=None,
            last_login_at=None,
            optins=None,
        )

    client.app.dependency_overrides[api_key_security] = mock_api_key_security

    with (
        patch("langflow.api.v1.openai_responses.get_flow_by_id_or_endpoint_name") as mock_get_flow,
        patch("langflow.api.v1.openai_responses.run_flow_generator") as mock_run_flow,
        patch("langflow.api.v1.openai_responses.consume_and_yield") as mock_consume,
    ):
        mock_flow = MagicMock()
        mock_flow.data = {"nodes": [{"data": {"type": "ChatInput"}}, {"data": {"type": "ChatOutput"}}]}
        mock_get_flow.return_value = mock_flow
        mock_run_flow.return_value = None

        tool_start_event = json.dumps(
            {
                "event": "add_message",
                "data": {
                    "sender": "AI",
                    "sender_name": "Agent",
                    "text": "",
                    "properties": {"state": "streaming"},
                    "content_blocks": [
                        {
                            "title": "Agent Steps",
                            "contents": [
                                {
                                    "type": "tool_use",
                                    "name": "search_documents",
                                    "tool_input": {"search_query": "documents"},
                                    "output": None,
                                }
                            ],
                        }
                    ],
                },
            }
        ).encode("utf-8")

        tool_done_event = json.dumps(
            {
                "event": "add_message",
                "data": {
                    "sender": "AI",
                    "sender_name": "Agent",
                    "text": "",
                    "properties": {"state": "streaming"},
                    "content_blocks": [
                        {
                            "title": "Agent Steps",
                            "contents": [
                                {
                                    "type": "tool_use",
                                    "name": "search_documents",
                                    "tool_input": {"search_query": "documents"},
                                    "output": [{"text_key": "text", "data": {"text": "doc result"}}],
                                }
                            ],
                        }
                    ],
                },
            }
        ).encode("utf-8")

        async def event_generator(*_, **__):
            yield tool_start_event
            yield tool_done_event
            yield None

        mock_consume.side_effect = event_generator

        response = client.post(
            "/api/v1/responses",
            json={
                "model": "test-flow-id",
                "input": "test input",
                "stream": True,
                "include": ["tool_call.results"],
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        chunks = parse_data_chunks(response.content.decode("utf-8"))

        added_index = next(
            i
            for i, chunk in enumerate(chunks)
            if chunk.get("type") == "response.output_item.added"
            and chunk.get("item", {}).get("type") == "function_call"
        )
        done_index = next(
            i
            for i, chunk in enumerate(chunks)
            if chunk.get("type") == "response.output_item.done"
            and chunk.get("item", {}).get("type") == "tool_call"
        )

        assert added_index < done_index
        assert chunks[added_index]["item"]["name"] == "search_documents"
        assert chunks[done_index]["item"]["tool_name"] == "search_documents"

    client.app.dependency_overrides = {}
