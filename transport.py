"""Kiwi transport abstraction layer for inter-node communication.

Provides a unified interface (KiwiTransport) with two implementations:
- HttpTransport: Uses httpx for HTTP POST calls (current Docker-internal behavior)
- P2PTransport: Uses hivemind.p2p libp2p streams for direct internet P2P connectivity
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any, AsyncIterator, Callable

import httpx

logger = logging.getLogger("kiwi.transport")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)


class KiwiTransport(ABC):
    """Abstract transport interface for inter-node tensor forwarding and generation."""

    @abstractmethod
    async def start(self) -> None:
        """Start the transport service (listen for connections / daemon start)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport service and release resources."""

    @abstractmethod
    async def get_address(self) -> str:
        """Return the address other nodes use to reach this node.

        HTTP mode: returns NODE_URL (e.g. 'http://worker_1:8000')
        P2P mode: returns libp2p PeerID (e.g. 'QmXyz...')
        """

    @abstractmethod
    async def send_forward(
        self, target: str, payload: bytes, headers: dict[str, str], timeout: float = 30.0
    ) -> bytes:
        """Send hidden state tensors to the target node's /forward endpoint..."""

    @abstractmethod
    async def send_generate(
        self, target: str, payload: bytes, headers: dict[str, str], timeout: float = 60.0
    ) -> AsyncIterator[bytes]:
        """Send input_ids payload to the target node's /generate endpoint and stream token responses."""

    @abstractmethod
    def set_forward_handler(
        self, handler: Callable[[dict[str, str], bytes], asyncio.Future[bytes] | bytes]
    ) -> None:
        """Register callback for incoming /forward requests.

        Handler takes (headers, payload_bytes) and returns response_bytes.
        """

    @abstractmethod
    def set_generate_handler(
        self, handler: Callable[[dict[str, str], bytes], AsyncIterator[bytes]]
    ) -> None:
        """Register callback for incoming /generate requests.

        Handler takes (headers, payload_bytes) and yields response lines (bytes).
        """


class HttpTransport(KiwiTransport):
    """HTTP transport implementation wrapping httpx.AsyncClient."""

    def __init__(self, node_url: str = "") -> None:
        self._node_url = node_url.rstrip("/")
        self._forward_handler: Callable | None = None
        self._generate_handler: Callable | None = None

    async def start(self) -> None:
        logger.info(f"[HttpTransport] Started for node_url={self._node_url}")

    async def stop(self) -> None:
        logger.info("[HttpTransport] Stopped.")

    async def get_address(self) -> str:
        return self._node_url

    async def send_forward(
        self, target: str, payload: bytes, headers: dict[str, str], timeout: float = 30.0
    ) -> bytes:
        target_url = f"{target.rstrip('/')}/forward"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(target_url, headers=headers, content=payload)
            response.raise_for_status()
            return response.content

    async def send_generate(
        self, target: str, payload: bytes, headers: dict[str, str], timeout: float = 60.0
    ) -> AsyncIterator[bytes]:
        target_url = f"{target.rstrip('/')}/generate"
        timeout_obj = httpx.Timeout(timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout_obj) as client:
            async with client.stream(
                "POST", target_url, headers=headers, content=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line_str = line.strip()
                    if line_str:
                        yield line_str.encode("utf-8")

    def set_forward_handler(
        self, handler: Callable[[dict[str, str], bytes], asyncio.Future[bytes] | bytes]
    ) -> None:
        # In HTTP mode, FastAPI routes handle requests directly, but we store the handler reference.
        self._forward_handler = handler

    def set_generate_handler(
        self, handler: Callable[[dict[str, str], bytes], AsyncIterator[bytes]]
    ) -> None:
        self._generate_handler = handler


# ── ACK-based chunked stream protocol ─────────────────────────────────────────
# Provides true end-to-end flow control through libp2p circuit relays.
# The sender writes data in 32KB chunks and waits for a 1-byte ACK from the
# receiver after each chunk, preventing the relay's ~64KB buffer from overflowing.

_ACK = b"\x06"
_CHUNK_SIZE = 32768  # 32KB — safely under the 64KB circuit relay window


async def _write_acked(
    writer: asyncio.StreamWriter,
    reader: asyncio.StreamReader,
    data: bytes,
    chunk_size: int = _CHUNK_SIZE,
) -> None:
    """Write *data* in chunks, waiting for a 1-byte ACK after each chunk."""
    offset = 0
    total = len(data)
    while offset < total:
        end = min(offset + chunk_size, total)
        writer.write(data[offset:end])
        await writer.drain()
        ack = await asyncio.wait_for(reader.readexactly(1), timeout=60.0)
        if ack != _ACK:
            raise RuntimeError(f"Expected ACK 0x06, got {ack!r}")
        offset = end


async def _read_acked(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    total_len: int,
    chunk_size: int = _CHUNK_SIZE,
) -> bytes:
    """Read *total_len* bytes in chunks, sending a 1-byte ACK after each chunk."""
    buffer = bytearray()
    while len(buffer) < total_len:
        to_read = min(chunk_size, total_len - len(buffer))
        chunk = await asyncio.wait_for(reader.readexactly(to_read), timeout=60.0)
        buffer.extend(chunk)
        writer.write(_ACK)
        await writer.drain()
    return bytes(buffer)


async def _write_framed_request(
    writer: asyncio.StreamWriter,
    reader: asyncio.StreamReader,
    headers: dict[str, str],
    payload: bytes,
) -> None:
    """Write a framed request [4B header_len][header_json][4B payload_len][payload] with ACKs."""
    header_bytes = json.dumps(headers).encode("utf-8")
    # Header length + header body (small — typically < 1 KB)
    await _write_acked(writer, reader, struct.pack(">I", len(header_bytes)) + header_bytes)
    # Payload length prefix
    await _write_acked(writer, reader, struct.pack(">I", len(payload)))
    # Payload body (potentially large — up to several MB)
    if payload:
        await _write_acked(writer, reader, payload)


async def _read_framed_request(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> tuple[dict[str, str], bytes]:
    """Read a framed request [4B header_len][header_json][4B payload_len][payload] with ACKs."""
    # Header length + header body
    header_block = await _read_acked(reader, writer, 4)
    header_len = struct.unpack(">I", header_block)[0]
    header_json_bytes = await _read_acked(reader, writer, header_len)
    headers = json.loads(header_json_bytes.decode("utf-8"))
    # Payload length prefix
    payload_len_bytes = await _read_acked(reader, writer, 4)
    payload_len = struct.unpack(">I", payload_len_bytes)[0]
    # Payload body
    payload = await _read_acked(reader, writer, payload_len) if payload_len else b""
    return headers, payload


class P2PTransport(KiwiTransport):
    """P2P transport implementation wrapping hivemind.p2p libp2p streams."""

    def __init__(
        self,
        initial_peers: list[str] | None = None,
        listen_port: int = 4001,
        relay_mode: bool = False,
        announce_maddrs: list[str] | None = None,
    ) -> None:
        self.initial_peers = initial_peers or []
        self.listen_port = listen_port
        self.relay_mode = relay_mode
        self.announce_maddrs = announce_maddrs
        self._p2p: Any | None = None
        self._peer_id = ""
        self._forward_handler: Callable | None = None
        self._generate_handler: Callable | None = None

    async def start(self) -> None:
        try:
            from hivemind.p2p.p2p_daemon import P2P
        except ImportError as exc:
            raise RuntimeError(
                "hivemind is not installed. Please install hivemind (`pip install hivemind`) to use P2PTransport."
            ) from exc

        logger.info(f"[P2PTransport] Starting P2P node on port {self.listen_port} (relay_mode={self.relay_mode})...")
        
        host_maddrs = [f"/ip4/0.0.0.0/tcp/{self.listen_port}"]
        
        create_kwargs = dict(
            initial_peers=self.initial_peers if self.initial_peers else None,
            host_maddrs=host_maddrs,
            use_relay=True,
            use_auto_relay=True,
            dht_mode="server",
            force_reachability="public" if self.relay_mode else "private",
        )
        if self.announce_maddrs:
            create_kwargs["announce_maddrs"] = self.announce_maddrs
            print(f"[P2PTransport] Announcing public addresses: {self.announce_maddrs}", flush=True)
        self._p2p = await P2P.create(**create_kwargs)
        self._peer_id = self._p2p.peer_id.to_base58()
        logger.info(f"[P2PTransport] P2P node started! PeerID: {self._peer_id}")
        print(f"[P2PTransport] P2P node started! PeerID: {self._peer_id}", flush=True)

        await self._p2p.add_binary_stream_handler(
            "kiwi/forward/1.0", self._handle_forward_stream
        )
        await self._p2p.add_binary_stream_handler(
            "kiwi/generate/1.0", self._handle_generate_stream
        )
        logger.info("[P2PTransport] Registered stream handlers kiwi/forward/1.0 and kiwi/generate/1.0")

    async def stop(self) -> None:
        if self._p2p is not None:
            logger.info(f"[P2PTransport] Shutting down P2P node {self._peer_id}...")
            await self._p2p.shutdown()
            self._p2p = None

    async def get_address(self) -> str:
        return self._peer_id

    async def _handle_forward_stream(
        self, stream_info: Any, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            headers, payload = await _read_framed_request(reader, writer)
            if not self._forward_handler:
                raise RuntimeError("No forward handler set on P2PTransport")

            response = self._forward_handler(headers, payload)
            if asyncio.isfuture(response) or asyncio.iscoroutine(response):
                response_bytes = await response
            else:
                response_bytes = response

            # Send response length + body with ACKs
            await _write_acked(writer, reader, struct.pack(">I", len(response_bytes)))
            await _write_acked(writer, reader, response_bytes)
        except Exception as exc:
            logger.error(f"[P2PTransport] Error handling forward stream from {getattr(stream_info, 'peer_id', 'unknown')}: {exc}", exc_info=True)
            print(f"[P2PTransport] Error handling forward stream from {getattr(stream_info, 'peer_id', 'unknown')}: {exc}", flush=True)
            try:
                err_bytes = json.dumps({"error": str(exc)}).encode("utf-8")
                await _write_acked(writer, reader, struct.pack(">I", len(err_bytes)))
                await _write_acked(writer, reader, err_bytes)
            except Exception:
                pass  # Best-effort error response; caller will timeout and retry
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _handle_generate_stream(
        self, stream_info: Any, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            headers, payload = await _read_framed_request(reader, writer)
            if not self._generate_handler:
                raise RuntimeError("No generate handler set on P2PTransport")

            async for chunk_bytes in self._generate_handler(headers, payload):
                writer.write(chunk_bytes)
                if not chunk_bytes.endswith(b"\n"):
                    writer.write(b"\n")
                await writer.drain()
        except Exception as exc:
            logger.error(f"[P2PTransport] Error handling generate stream from {getattr(stream_info, 'peer_id', 'unknown')}: {exc}", exc_info=True)
            print(f"[P2PTransport] Error handling generate stream from {getattr(stream_info, 'peer_id', 'unknown')}: {exc}", flush=True)
            err_line = json.dumps({"error": str(exc), "failed_node": self._peer_id}).encode("utf-8") + b"\n"
            writer.write(err_line)
            await writer.drain()
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def send_forward(
        self, target: str, payload: bytes, headers: dict[str, str], timeout: float = 120.0
    ) -> bytes:
        if self._p2p is None:
            raise RuntimeError("P2PTransport not started.")

        from hivemind.p2p.p2p_daemon_bindings import PeerID

        target_peer = PeerID.from_base58(target)

        async def _call() -> bytes:
            _, reader, writer = await self._p2p.call_binary_stream_handler(
                target_peer, "kiwi/forward/1.0"
            )
            try:
                # Send request with ACK-based flow control
                await _write_framed_request(writer, reader, headers, payload)
                # Read response with ACK-based flow control
                resp_len_bytes = await _read_acked(reader, writer, 4)
                resp_len = struct.unpack(">I", resp_len_bytes)[0]
                return await _read_acked(reader, writer, resp_len)
            finally:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()

        max_attempts = 5
        backoff = 0.1
        for attempt in range(1, max_attempts + 1):
            try:
                return await asyncio.wait_for(_call(), timeout=timeout)
            except Exception as e:
                if attempt == max_attempts:
                    logger.error(f"[P2PTransport] send_forward to {target} failed after {max_attempts} attempts: {e}")
                    raise
                logger.warning(
                    f"[P2PTransport] Attempt {attempt} failed forwarding to {target}: {e}. Retrying in {backoff}s..."
                )
                await asyncio.sleep(backoff)
                backoff *= 2

    async def send_generate(
        self, target: str, payload: bytes, headers: dict[str, str], timeout: float = 60.0
    ) -> AsyncIterator[bytes]:
        if self._p2p is None:
            raise RuntimeError("P2PTransport not started.")

        from hivemind.p2p.p2p_daemon_bindings import PeerID

        target_peer = PeerID.from_base58(target)

        max_attempts = 5
        backoff = 0.1
        reader, writer = None, None
        for attempt in range(1, max_attempts + 1):
            try:
                _, reader, writer = await asyncio.wait_for(
                    self._p2p.call_binary_stream_handler(target_peer, "kiwi/generate/1.0"),
                    timeout=10.0,
                )
                # Send request with ACK-based flow control
                await _write_framed_request(writer, reader, headers, payload)
                break
            except Exception as e:
                if writer is not None:
                    writer.close()
                    with suppress(Exception):
                        await writer.wait_closed()
                if attempt == max_attempts:
                    logger.error(f"[P2PTransport] send_generate to {target} failed to establish stream after {max_attempts} attempts: {e}")
                    raise
                logger.warning(
                    f"[P2PTransport] Attempt {attempt} failed to establish generate stream to {target}: {e}. Retrying in {backoff}s..."
                )
                await asyncio.sleep(backoff)
                backoff *= 2

        try:
            while not reader.at_eof():
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(f"Timed out reading from P2P stream to {target}") from exc
                if not line:
                    break
                yield line.strip()
        finally:
            if writer is not None:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()

    def set_forward_handler(
        self, handler: Callable[[dict[str, str], bytes], asyncio.Future[bytes] | bytes]
    ) -> None:
        self._forward_handler = handler

    def set_generate_handler(
        self, handler: Callable[[dict[str, str], bytes], AsyncIterator[bytes]]
    ) -> None:
        self._generate_handler = handler


def create_transport(
    transport_type: str,
    node_url: str = "",
    initial_peers: list[str] | None = None,
    listen_port: int = 4001,
    relay_mode: bool = False,
    announce_maddrs: list[str] | None = None,
) -> KiwiTransport:
    """Factory function to instantiate the requested transport type."""
    t_type = transport_type.lower().strip()
    if t_type == "p2p":
        return P2PTransport(
            initial_peers=initial_peers,
            listen_port=listen_port,
            relay_mode=relay_mode,
            announce_maddrs=announce_maddrs,
        )
    return HttpTransport(node_url=node_url)
