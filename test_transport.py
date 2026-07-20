"""Unit tests for Kiwi transport layer (`transport.py`).

Tests both HttpTransport and P2PTransport classes, framing logic, and factories.
Run with `.venv\\Scripts\\pytest.exe test_transport.py` or `.venv\\Scripts\\python.exe test_transport.py`.
"""

import asyncio
import json
import struct
from typing import AsyncIterator

import pytest
from transport import (
    HttpTransport,
    P2PTransport,
    _read_framed_request,
    create_transport,
)


class MockStreamWriter:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes):
        self.data.extend(data)

    async def drain(self):
        pass

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass


class MockStreamInfo:
    def __init__(self, peer_id: str = "QmTestPeer123"):
        self.peer_id = peer_id


@pytest.mark.asyncio
async def test_factory():
    http_t = create_transport("http", node_url="http://localhost:8000")
    assert isinstance(http_t, HttpTransport)
    assert await http_t.get_address() == "http://localhost:8000"

    p2p_t = create_transport("p2p", listen_port=4005)
    assert isinstance(p2p_t, P2PTransport)


@pytest.mark.asyncio
async def test_framed_request_reader():
    reader = asyncio.StreamReader()
    headers = {"X-Request-ID": "req-101", "X-Step": "prefill"}
    payload = b"dummy safetensors tensor payload"

    header_bytes = json.dumps(headers).encode("utf-8")
    framed = (
        struct.pack(">I", len(header_bytes))
        + header_bytes
        + struct.pack(">I", len(payload))
        + payload
    )
    reader.feed_data(framed)
    reader.feed_eof()

    decoded_headers, decoded_payload = await _read_framed_request(reader)
    assert decoded_headers == headers
    assert decoded_payload == payload


@pytest.mark.asyncio
async def test_p2p_handle_forward_stream():
    p2p_t = P2PTransport()

    async def forward_handler(headers: dict[str, str], payload: bytes) -> bytes:
        assert headers["X-Request-ID"] == "test-req"
        assert payload == b"input_bytes"
        return b"response_tensor_bytes"

    p2p_t.set_forward_handler(forward_handler)

    reader = asyncio.StreamReader()
    headers = {"X-Request-ID": "test-req"}
    payload = b"input_bytes"
    header_bytes = json.dumps(headers).encode("utf-8")
    framed_req = (
        struct.pack(">I", len(header_bytes))
        + header_bytes
        + struct.pack(">I", len(payload))
        + payload
    )
    reader.feed_data(framed_req)
    reader.feed_eof()

    writer = MockStreamWriter()
    await p2p_t._handle_forward_stream(MockStreamInfo(), reader, writer)

    assert writer.closed
    resp_len = struct.unpack(">I", writer.data[:4])[0]
    resp_data = bytes(writer.data[4:4 + resp_len])
    assert resp_data == b"response_tensor_bytes"


@pytest.mark.asyncio
async def test_p2p_handle_generate_stream():
    p2p_t = P2PTransport()

    async def generate_handler(headers: dict[str, str], payload: bytes) -> AsyncIterator[bytes]:
        assert headers["X-Max-Tokens"] == "3"
        yield b'{"token_id": 101}\n'
        yield b'{"token_id": 102}'
        yield b'{"token_id": 103}\n'

    p2p_t.set_generate_handler(generate_handler)

    reader = asyncio.StreamReader()
    headers = {"X-Max-Tokens": "3"}
    payload = b"safetensors_input"
    header_bytes = json.dumps(headers).encode("utf-8")
    framed_req = (
        struct.pack(">I", len(header_bytes))
        + header_bytes
        + struct.pack(">I", len(payload))
        + payload
    )
    reader.feed_data(framed_req)
    reader.feed_eof()

    writer = MockStreamWriter()
    await p2p_t._handle_generate_stream(MockStreamInfo(), reader, writer)

    assert writer.closed
    expected_stream = b'{"token_id": 101}\n{"token_id": 102}\n{"token_id": 103}\n'
    assert bytes(writer.data) == expected_stream


if __name__ == "__main__":
    async def run_all():
        await test_factory()
        await test_framed_request_reader()
        await test_p2p_handle_forward_stream()
        await test_p2p_handle_generate_stream()
        print("\n✅ All test_transport.py tests passed cleanly!")

    asyncio.run(run_all())
