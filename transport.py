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

    async def connect_to_peer(self, maddr_str: str) -> bool:
        """Explicitly connect or register a peer address into the transport pool."""
        return True

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

    async def connect_to_peer(self, maddr_str: str) -> bool:
        return True

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


async def _write_stream_chunked(writer: asyncio.StreamWriter, data: bytes, chunk_size: int = 16384) -> None:
    """Write data in chunks with writer.drain() and micro-pacing to prevent libp2p/Yamux flow-control window overflows across oceanic circuit relays."""
    total_len = len(data)
    chunks_count = (total_len + chunk_size - 1) // chunk_size if total_len > 0 else 1
    if total_len > chunk_size:
        logger.info(f"[P2PTransport] [ChunkedWrite] Starting multi-chunk stream of {total_len:,} bytes across {chunks_count} chunks (chunk_size={chunk_size:,}B)...")

    for idx, i in enumerate(range(0, total_len, chunk_size), 1):
        chunk = data[i : i + chunk_size]
        writer.write(chunk)
        await writer.drain()
        if total_len > 32768:
            await asyncio.sleep(0.005)  # 5ms micro-pacing gives Yamux window ACKs time to traverse trans-oceanic relays
        if total_len > chunk_size and (idx == 1 or idx == chunks_count or idx % 5 == 0):
            logger.info(f"[P2PTransport] [ChunkedWrite] Drained chunk {idx}/{chunks_count} ({len(chunk):,}B)...")

    if total_len > chunk_size:
        logger.info(f"[P2PTransport] [ChunkedWrite] Successfully streamed all {total_len:,} bytes across {chunks_count} chunks!")


async def _read_stream_chunked(reader: asyncio.StreamReader, total_len: int, chunk_size: int = 16384) -> bytes:
    """Read total_len bytes flexibly in chunks to prevent libp2p stream buffer starvation / timeouts."""
    chunks_expected = (total_len + chunk_size - 1) // chunk_size if total_len > 0 else 1
    if total_len > chunk_size:
        logger.info(f"[P2PTransport] [ChunkedRead] Starting multi-chunk read of {total_len:,} bytes expecting ~{chunks_expected} chunks (chunk_size={chunk_size:,}B)...")

    buffer = bytearray()
    remaining = total_len
    chunks_read = 0
    while remaining > 0:
        chunk = await reader.read(min(remaining, chunk_size))
        if not chunk:
            raise asyncio.exceptions.IncompleteReadError(bytes(buffer), total_len)
        buffer.extend(chunk)
        remaining -= len(chunk)
        chunks_read += 1
        if total_len > chunk_size and (chunks_read == 1 or remaining == 0 or chunks_read % 10 == 0):
            logger.info(f"[P2PTransport] [ChunkedRead] Read chunk {chunks_read} ({len(chunk):,}B), remaining: {remaining:,}B")

    if total_len > chunk_size:
        logger.info(f"[P2PTransport] [ChunkedRead] Successfully completed read of all {total_len:,} bytes across {chunks_read} chunks!")
    return bytes(buffer)


async def _read_framed_request(reader: asyncio.StreamReader) -> tuple[dict[str, str], bytes]:
    """Read a framed request of [4B header_len][header_json][4B payload_len][payload]."""
    header_len_bytes = await reader.readexactly(4)
    header_len = struct.unpack(">I", header_len_bytes)[0]
    header_json_bytes = await reader.readexactly(header_len)
    headers = json.loads(header_json_bytes.decode("utf-8"))

    payload_len_bytes = await reader.readexactly(4)
    payload_len = struct.unpack(">I", payload_len_bytes)[0]
    payload = await _read_stream_chunked(reader, payload_len)
    return headers, payload


class P2PTransport(KiwiTransport):
    """P2P transport implementation wrapping hivemind.p2p libp2p streams."""

    def __init__(
        self, initial_peers: list[str] | None = None, listen_port: int = 4001, relay_mode: bool = False
    ) -> None:
        self.initial_peers = initial_peers or []
        self.listen_port = listen_port
        self.relay_mode = relay_mode
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
        
        self._p2p = await P2P.create(
            initial_peers=self.initial_peers if self.initial_peers else None,
            host_maddrs=host_maddrs,
            use_relay=True,
            use_auto_relay=True,
            dht_mode="server",
            force_reachability="public" if self.relay_mode else "private",
            startup_timeout=180.0,
        )
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

    async def connect_to_peer(self, maddr_str: str) -> bool:
        """Explicitly connect to a peer given its multiaddress string."""
        if self._p2p is None or "/p2p/" not in maddr_str:
            return False
        try:
            from hivemind.p2p.p2p_daemon_bindings.p2pclient import PeerID, Multiaddr
            peer_id_str = maddr_str.split("/p2p/")[-1].split("/")[0]
            if peer_id_str == self._peer_id:
                return True
            peer_id = PeerID.from_base58(peer_id_str)
            maddr = Multiaddr(maddr_str)
            await self._p2p._client.connect(peer_id, [maddr])
            return True
        except Exception as exc:
            logger.debug(f"[P2PTransport] Could not explicitly connect to {maddr_str}: {exc}")
            return False

    async def _handle_forward_stream(
        self, stream_info: Any, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            headers, payload = await _read_framed_request(reader)
            if not self._forward_handler:
                raise RuntimeError("No forward handler set on P2PTransport")

            response = self._forward_handler(headers, payload)
            if asyncio.isfuture(response) or asyncio.iscoroutine(response):
                response_bytes = await response
            else:
                response_bytes = response

            await _write_stream_chunked(writer, struct.pack(">I", len(response_bytes)) + response_bytes)
        except Exception as exc:
            logger.error(f"[P2PTransport] Error handling forward stream from {getattr(stream_info, 'peer_id', 'unknown')}: {exc}", exc_info=True)
            print(f"[P2PTransport] Error handling forward stream from {getattr(stream_info, 'peer_id', 'unknown')}: {exc}", flush=True)
            err_bytes = json.dumps({"error": str(exc)}).encode("utf-8")
            await _write_stream_chunked(writer, struct.pack(">I", len(err_bytes)) + err_bytes)
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _handle_generate_stream(
        self, stream_info: Any, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            headers, payload = await _read_framed_request(reader)
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
        self, target: str, payload: bytes, headers: dict[str, str], timeout: float = 30.0
    ) -> bytes:
        if self._p2p is None:
            raise RuntimeError("P2PTransport not started.")

        from hivemind.p2p.p2p_daemon_bindings import PeerID

        target_peer = PeerID.from_base58(target)
        header_bytes = json.dumps(headers).encode("utf-8")
        framed_req = (
            struct.pack(">I", len(header_bytes))
            + header_bytes
            + struct.pack(">I", len(payload))
            + payload
        )

        async def _call() -> bytes:
            _, reader, writer = await self._p2p.call_binary_stream_handler(
                target_peer, "kiwi/forward/1.0"
            )
            try:
                await _write_stream_chunked(writer, framed_req)

                resp_len_bytes = await reader.readexactly(4)
                resp_len = struct.unpack(">I", resp_len_bytes)[0]
                return await _read_stream_chunked(reader, resp_len)
            finally:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()

        max_attempts = 6
        backoff = 0.5
        for attempt in range(1, max_attempts + 1):
            try:
                return await asyncio.wait_for(_call(), timeout=timeout)
            except Exception as e:
                if attempt == max_attempts:
                    logger.error(f"[P2PTransport] send_forward to {target} failed after {max_attempts} attempts: {e}")
                    raise
                logger.warning(
                    f"[P2PTransport] Attempt {attempt} failed forwarding to {target}: {e}. Retrying in {backoff:.1f}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 2.0)

    async def send_generate(
        self, target: str, payload: bytes, headers: dict[str, str], timeout: float = 60.0
    ) -> AsyncIterator[bytes]:
        if self._p2p is None:
            raise RuntimeError("P2PTransport not started.")

        from hivemind.p2p.p2p_daemon_bindings import PeerID

        target_peer = PeerID.from_base58(target)
        header_bytes = json.dumps(headers).encode("utf-8")
        framed_req = (
            struct.pack(">I", len(header_bytes))
            + header_bytes
            + struct.pack(">I", len(payload))
            + payload
        )

        max_attempts = 6
        backoff = 0.5
        reader, writer = None, None
        for attempt in range(1, max_attempts + 1):
            try:
                _, reader, writer = await asyncio.wait_for(
                    self._p2p.call_binary_stream_handler(target_peer, "kiwi/generate/1.0"),
                    timeout=10.0,
                )
                await _write_stream_chunked(writer, framed_req)
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
                    f"[P2PTransport] Attempt {attempt} failed to establish generate stream to {target}: {e}. Retrying in {backoff:.1f}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 2.0)

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
) -> KiwiTransport:
    """Factory function to instantiate the requested transport type."""
    t_type = transport_type.lower().strip()
    if t_type == "p2p":
        return P2PTransport(
            initial_peers=initial_peers, listen_port=listen_port, relay_mode=relay_mode
        )
    return HttpTransport(node_url=node_url)
