"""Simulated WAN latency for the training benchmark (WAN_SIMULATION_PLAN.md).

Every number in benchmark_results_v6 was produced by eight containers on one
host talking over a Docker bridge, where a round trip costs well under a
millisecond. That is not a deployment — it is a lower bound no real cluster
reaches. This module makes the network cost a configurable RTT so the averaging
arms can be compared where coordination is not free.

WHY tc netem AND NOT AN APPLICATION-LEVEL SLEEP
    A `sleep()` in transport.py would be five minutes of work, but it is not a
    network: no handshake cost, no congestion window, no interaction between
    payload size and delay. Since v6 shows the merge round is already
    coordination-bound rather than bandwidth-bound (sparse ships 7x less data
    for the same round time), the payload/latency interaction is exactly the
    thing under test. netem also makes a bandwidth cap a two-word config change
    later — `delay 37ms rate 100mbit` — rather than a refactor.

HOW tc IS RUN: A THROWAWAY SIDECAR, NOT nsenter, NOT NET_ADMIN IN THE IMAGE
    `docker run --rm --network=container:<target> --cap-add=NET_ADMIN <tc image>`
    starts a ~10 MB container that JOINS the target's network namespace, applies
    the qdisc, and exits. Three properties, all of which we need:

      1. The worker image is untouched. It is the system under test; baking in
         iproute2 and an entrypoint would mean the container producing these
         numbers is not the container that produced v6.
      2. No root on the host, no sudo, no nsenter.
      3. It is HOST-OS AGNOSTIC. Everything happens inside Docker's Linux VM, so
         this works identically on a Linux box and on Docker Desktop for
         Windows/macOS. The nsenter approach does not: on Windows the
         orchestrator runs as a Windows process with no nsenter, no sudo and no
         `tc`, and `.State.Pid` is a PID inside a VM it cannot address.

    The container list is derived from `compose ps -q`, so no container can be
    missed — the failure mode of a hand-maintained list is an ASYMMETRIC mesh
    (six delayed workers, one fast) that nothing in the results would announce.

THE DELAY IS PER-SIDE
    netem shapes EGRESS. A round trip between two containers that each carry
    `delay D` costs 2D. Everything in this module takes and reports RTT, and
    halves it exactly once, at the point of application. Getting this wrong
    doubles an entire experiment and the results still look plausible, which is
    why `verify` exists and bounds the measurement from BOTH sides.

CLEANUP
    None needed. The qdisc lives in the container's network namespace, which
    `compose down` destroys along with the container.
"""

import logging
import os
import subprocess
from typing import List, Optional, Tuple

# The interface a container gets on a compose-managed bridge network.
IFACE = "eth0"

# Image providing `tc`. Pinned by env so a locked-down or offline host can
# substitute a locally built one:
#     printf 'FROM alpine\nRUN apk add --no-cache iproute2\n' > Dockerfile.netem
#     docker build -t kiwi-netem:latest -f Dockerfile.netem .
#     KIWI_NETEM_IMAGE=kiwi-netem:latest
# gaiadocker/iproute2 is the default because it is tiny (~10 MB) and is what
# Pumba uses for the same job, so it is widely mirrored.
NETEM_IMAGE = os.environ.get("KIWI_NETEM_IMAGE", "gaiadocker/iproute2")

# Applied delay is rtt_ms / 2 per container. A measured RTT outside these
# multiples of the requested value means the qdisc did not take, or took twice —
# both are run-invalidating and both are silent.
_MEASURE_MIN_RATIO = 0.5
_MEASURE_MAX_RATIO = 1.75

# netem's queue depth, in packets. MUST be set explicitly: the default is 1000,
# and 1000 is far too small here.
#
# A delayed qdisc holds every packet for the delay window, so its occupancy is
# send-rate x delay. The Docker bridge is multi-gigabit, so at 37.5 ms a burst
# can want tens of thousands of packets in flight — while netem silently DROPS
# everything past `limit`. That turns a latency-only experiment into a
# latency-plus-loss experiment: TCP reads the drops as congestion, collapses
# cwnd, and settles at whatever rate keeps the queue near 1000 packets — an
# accidental bandwidth cap of very roughly 1000 x 1500 B / 37.5 ms ~ 300 Mbit/s
# that nobody chose and nothing in the results would report.
#
# It would also land unevenly across the arms this experiment compares: the
# dense and outer arms ship 11.7 MB per round against sparse5's 1.7 MB, so an
# invisible loss-based cap penalises exactly the arms whose payload is large,
# and would read as a latency finding.
#
# 100k packets is a ceiling, not an expectation — TCP's own window keeps real
# occupancy in the low thousands, so nothing is allocated up front. Verify with
# `tc -s qdisc show` (see `show`): a non-zero drop count means raise it further.
_NETEM_LIMIT_PKTS = 100_000


def _docker() -> List[str]:
    """Resolved docker invocation.

    Imported lazily because orchestrator.py imports this module at its top, and
    a module-level import back would be circular. docker_cmd() and compose_cmd()
    memoise into module globals, so repeated calls cost nothing after the first.
    """
    from .orchestrator import docker_cmd
    return docker_cmd()


def _compose() -> List[str]:
    from .orchestrator import compose_cmd
    return compose_cmd()


def _run(cmd: List[str], *, check: bool = True, timeout: int = 120) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _container_ids(compose: str) -> List[str]:
    """Every running container in the compose project.

    Derived, never hardcoded: the two compose files name their containers
    differently (kiwi-worker_N_mesh vs kiwi-worker_N_p2p), and a hand-written
    list is how you end up with one undelayed worker.
    """
    out = _run(_compose() + ["-f", compose, "ps", "-q"])
    ids = [line.strip() for line in out.splitlines() if line.strip()]
    if not ids:
        raise RuntimeError(
            f"No running containers found for {compose}. netem must be applied "
            f"after `compose up -d`, not before."
        )
    return ids


def _container_id(compose: str, service: str) -> Optional[str]:
    """Container ID for one compose service, or None if it is not running."""
    out = _run(_compose() + ["-f", compose, "ps", "-q", service], check=False)
    ids = [line.strip() for line in out.splitlines() if line.strip()]
    return ids[0] if ids else None


def _network_of(cid: str) -> str:
    return _run(_docker() + [
        "inspect", "-f",
        "{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}", cid,
    ])


def _subnet_and_gateway(network: str) -> Tuple[str, str]:
    """The compose network's CIDR and its bridge gateway address.

    The gateway address matters as much as the subnet. The orchestrator drives
    training from the HOST — one POST /train_step per sample to localhost:8080 —
    and inside the container that connection appears to originate from the
    bridge gateway (172.x.0.1), which sits INSIDE the subnet. A naive
    "delay everything on the subnet" filter therefore delays every reply to the
    test harness: ~37 ms x 600 steps of pure measurement contamination, applied
    equally to every arm so nothing looks wrong. The gateway is excluded
    explicitly in apply_wan for exactly this reason.
    """
    subnet = _run(_docker() + [
        "network", "inspect", "-f", "{{(index .IPAM.Config 0).Subnet}}", network])
    gateway = _run(_docker() + [
        "network", "inspect", "-f", "{{(index .IPAM.Config 0).Gateway}}", network])
    if not subnet or not gateway:
        raise RuntimeError(
            f"Could not read subnet/gateway for docker network '{network}' "
            f"(subnet={subnet!r}, gateway={gateway!r})")
    return subnet, gateway


def _ensure_netem_image() -> None:
    """Pull the tc image once, up front.

    Without this the first `docker run` pulls mid-apply: the pull output lands
    in the failure message of whatever goes wrong next, and on a host with no
    registry access the error surfaces as an opaque tc failure rather than
    "the image is missing".
    """
    probe = subprocess.run(_docker() + ["image", "inspect", NETEM_IMAGE],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        return
    logging.info(f"WAN: pulling tc image {NETEM_IMAGE} (one-off, ~10 MB)")
    pull = subprocess.run(_docker() + ["pull", NETEM_IMAGE],
                          capture_output=True, text=True, timeout=300)
    if pull.returncode != 0:
        raise RuntimeError(
            f"Could not pull the tc image '{NETEM_IMAGE}':\n{pull.stderr.strip()}\n"
            f"On an offline host, build a local one and set KIWI_NETEM_IMAGE:\n"
            f"  printf 'FROM alpine\\nRUN apk add --no-cache iproute2\\n' > Dockerfile.netem\n"
            f"  docker build -t kiwi-netem:latest -f Dockerfile.netem .\n"
            f"  KIWI_NETEM_IMAGE=kiwi-netem:latest"
        )
    # Prove tc actually runs before joining a netns. Without this, an image
    # without tc on PATH fails on the first real command with tc's own cryptic
    # complaint about its arguments, several frames deep in apply_wan.
    probe = subprocess.run(_docker() + ["run", "--rm", "--entrypoint", "tc",
                                        NETEM_IMAGE, "-Version"],
                           capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        raise RuntimeError(
            f"Image '{NETEM_IMAGE}' does not provide a working `tc` on PATH:\n"
            f"{(probe.stderr or probe.stdout).strip()}"
        )


def _tc(cid: str, args: List[str], *, check: bool = True) -> str:
    """Run one tc command inside a container's network namespace.

    --network=container:<cid> puts the sidecar in the TARGET's netns, so the
    qdisc it installs is the target's. NET_ADMIN is granted to this throwaway
    container only, never to a worker.

    --entrypoint tc is not optional, and passing "tc" as the first argument
    instead is wrong. gaiadocker/iproute2 already declares ENTRYPOINT ["tc"], so
    an image argument of "tc" becomes tc's OBJECT argument and it dies with
    `Object "tc" is unknown, try "tc help".` Setting the entrypoint explicitly
    works for both kinds of image — one that defaults to tc and one that
    defaults to a shell — which matters because KIWI_NETEM_IMAGE is overridable.
    """
    return _run(_docker() + ["run", "--rm", f"--network=container:{cid}",
                             "--cap-add=NET_ADMIN", "--cap-add=NET_RAW",
                             "--entrypoint", "tc", NETEM_IMAGE] + args, check=check)


def apply_wan(compose: str, rtt_ms: int, iface: str = IFACE) -> None:
    """Apply `rtt_ms` of simulated round-trip latency to every container link.

    No-op at rtt_ms == 0, which is the LAN control arm.

    Shapes container-to-container traffic only. Traffic to the bridge gateway —
    i.e. replies to the orchestrator running on the host — is explicitly
    excluded, because that is the test harness and not part of the system under
    test.

    Must be called AFTER the cluster reports ready. Cluster formation (worker
    registration, model layer coverage, hivemind peer discovery) has its own
    timeouts and delaying it buys flakiness without buying signal; the write-up
    caveat is that the cluster forms on a LAN and trains on a WAN.

    Must be called PER SEED: `up --force-recreate` builds new containers and
    therefore new network namespaces, so a qdisc applied for seed 1 does not
    exist for seed 2.
    """
    if not rtt_ms:
        return

    # Halved exactly once, here. Every other function in this module speaks RTT.
    per_side_ms = rtt_ms / 2.0

    _ensure_netem_image()
    cids = _container_ids(compose)
    subnet, gateway = _subnet_and_gateway(_network_of(cids[0]))
    logging.info(
        f"WAN: applying {rtt_ms} ms RTT ({per_side_ms:g} ms egress per container) "
        f"to {len(cids)} containers on {subnet}, excluding host gateway {gateway}")

    for cid in cids:
        # Idempotent. A fresh container has a default root qdisc and this
        # delete is a no-op; on a retry within one seed it is what stops
        # `qdisc add` failing with "File exists".
        _tc(cid, ["qdisc", "del", "dev", iface, "root"], check=False)

        # Band 1:2 carries unmatched traffic and stays undelayed (the default
        # priomap maps normal TOS-0 traffic there); 1:3 gets netem.
        _tc(cid, ["qdisc", "add", "dev", iface, "root", "handle", "1:", "prio"])
        _tc(cid, ["qdisc", "add", "dev", iface, "parent", "1:3", "handle", "30:",
                  "netem", "delay", f"{per_side_ms:g}ms",
                  "limit", str(_NETEM_LIMIT_PKTS)])

        # Filters are evaluated in prio order and the first match wins, so the
        # gateway exclusion MUST be a lower prio number than the subnet rule.
        _tc(cid, ["filter", "add", "dev", iface, "protocol", "ip", "parent", "1:0",
                  "prio", "1", "u32", "match", "ip", "dst", f"{gateway}/32",
                  "flowid", "1:1"])
        _tc(cid, ["filter", "add", "dev", iface, "protocol", "ip", "parent", "1:0",
                  "prio", "2", "u32", "match", "ip", "dst", subnet,
                  "flowid", "1:3"])


def clear_wan(compose: str, iface: str = IFACE) -> None:
    """Remove the qdisc from every container. Not needed in the benchmark path
    (`compose down` destroys the netns), but useful when poking at a cluster by
    hand without tearing it down."""
    for cid in _container_ids(compose):
        _tc(cid, ["qdisc", "del", "dev", iface, "root"], check=False)


def show(compose: str, iface: str = IFACE) -> str:
    """The active qdisc on every container, with counters.

    `-s` is the point: it prints each qdisc's sent/dropped totals. A non-zero
    netem drop count means the queue depth is too small and the run silently
    contains packet loss on top of the latency it was supposed to isolate
    (see _NETEM_LIMIT_PKTS). Check this after the pilot, not just before it —
    an empty queue before training proves nothing.
    """
    lines = []
    for cid in _container_ids(compose):
        name = _run(_docker() + ["inspect", "-f", "{{.Name}}", cid]).lstrip("/")
        out = _tc(cid, ["-s", "qdisc", "show", "dev", iface], check=False)
        lines.append(f"{name}: {' | '.join(out.splitlines())}")
    return "\n".join(lines)


# Timed from inside a container, so it measures what the workers actually
# experience rather than what tc was asked for. A TCP connect is one round trip
# (SYN -> SYN-ACK), which is the quantity we configured.
#
# The name is resolved BEFORE timing: Docker's embedded DNS lives at 127.0.0.11,
# which is loopback and therefore not delayed, but its lookup time would
# otherwise land in the measurement. min() over several attempts strips
# scheduling noise from a box running seven training containers.
_PROBE = """
import socket, time
ip = socket.gethostbyname({peer!r})
best = None
for _ in range(5):
    t = time.perf_counter()
    s = socket.create_connection((ip, {port}), timeout=10)
    s.close()
    dt = (time.perf_counter() - t) * 1000.0
    best = dt if best is None else min(best, dt)
print(best)
"""


def measure_rtt(compose: str, source: str = "worker_1",
                peer: str = "worker_2", port: int = 8000) -> float:
    """Ground-truth RTT in ms, measured container-to-container.

    Defaults probe worker_1 -> worker_2: the same-stage peer path that merge
    payloads travel, which is the link whose cost this experiment is about.
    """
    cid = _container_id(compose, source)
    if cid is None:
        raise RuntimeError(f"Service '{source}' is not running under {compose}")
    out = _run(_docker() + ["exec", cid, "python", "-c",
                            _PROBE.format(peer=peer, port=port)], timeout=120)
    return float(out.splitlines()[-1])


def verify(compose: str, requested_rtt_ms: int, **kwargs) -> float:
    """Measure the applied latency and REFUSE to continue if it is wrong.

    This raises rather than warns on purpose. A tc call that fails silently
    produces a complete, plausible-looking run whose curves are indistinguishable
    from a legitimate "latency made no difference" finding — the single most
    expensive failure mode in this experiment, and three lines to make
    impossible. The upper bound catches the other half of it: a delay applied as
    RTT rather than half-RTT gives ~2x and looks entirely reasonable on its own.
    """
    measured = measure_rtt(compose, **kwargs)
    logging.info(f"WAN: requested {requested_rtt_ms} ms RTT, "
                 f"measured {measured:.1f} ms")
    low = _MEASURE_MIN_RATIO * requested_rtt_ms
    high = _MEASURE_MAX_RATIO * requested_rtt_ms
    if not (low <= measured <= high):
        raise RuntimeError(
            f"netem did not take as configured: requested {requested_rtt_ms} ms "
            f"RTT, measured {measured:.1f} ms (expected {low:.0f}-{high:.0f} ms). "
            f"Below range means the qdisc was not applied; above means the delay "
            f"was applied as full RTT per side instead of half. Refusing to run — "
            f"a mislabelled network produces results that cannot be distinguished "
            f"from real ones."
        )
    return measured


# ── Standalone CLI ──────────────────────────────────────────────────────────
#
# Exists so the netem plumbing can be smoke-tested against a running cluster
# WITHOUT spending a training run on it — bring the cluster up, apply, measure,
# tear down. That is the cheap way to confirm the mechanism works on a new host
# (a Windows box with Docker Desktop, say) before committing GPU hours on a
# Lambda instance.
#
#   docker compose -f docker-compose.internet.yml up -d
#   python -m benchmarks_training.netem --compose docker-compose.internet.yml --rtt 75
#   docker compose -f docker-compose.internet.yml down
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Apply/measure/clear simulated WAN latency.")
    ap.add_argument("--compose", default="docker-compose.internet.yml",
                    help="compose file whose containers to shape")
    ap.add_argument("--rtt", type=int, default=75,
                    help="round-trip latency in ms (0 clears)")
    ap.add_argument("--source", default="worker_1", help="service to measure from")
    ap.add_argument("--peer", default="worker_2", help="service to measure to")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--clear", action="store_true", help="remove all qdiscs and exit")
    args = ap.parse_args()

    if args.clear:
        clear_wan(args.compose)
        print("cleared.\n" + show(args.compose))
        raise SystemExit(0)

    baseline = measure_rtt(args.compose, args.source, args.peer, args.port)
    print(f"baseline RTT before netem: {baseline:.2f} ms")

    apply_wan(args.compose, args.rtt)
    print("\nqdisc now installed:")
    print(show(args.compose))

    measured = verify(args.compose, args.rtt,
                      source=args.source, peer=args.peer, port=args.port)
    print(f"\nrequested {args.rtt} ms RTT -> measured {measured:.2f} ms "
          f"(baseline was {baseline:.2f} ms)  OK")
