#!/usr/bin/env python3
"""Find Livox lidars on the wire and make sure this host can talk to them.

    python3 tools/setup/livox_host_setup.py --json            # discover + check, read-only
    python3 tools/setup/livox_host_setup.py                   # same, human readable
    python3 tools/setup/livox_host_setup.py --apply --dry-run # print what --apply would run
    python3 tools/setup/livox_host_setup.py --apply           # ask once, then run (sudo per command)

Discovery sends the Livox SDK2 "lidar search" command (UDP broadcast
255.255.255.255:56000) out of every up IPv4 interface and parses the replies,
which carry the serial number and the lidar's IP.  It needs no root and works at
layer 2 even when this host is on the wrong subnet.  The reply comes back to UDP
port 56000, so a running driver (which holds that port) can get in the way; that
is reported, never swallowed.  If the port cannot be bound, a ping sweep of the
local /24s is used as a fallback and its hits are marked unverified.

Wire format (verified against Livox-SDK2 sources sdk_core/comm/sdk_protocol.{h,cpp},
sdk_core/comm/define.h, sdk_core/device_manager.cpp::SendDetectionData and
sdk_core/command_handler/general_command_handler.cpp::HandleDetectionData; checked
by compiling the SDK's own FastCRC and comparing the request bytes, see tests):

  header, 24 bytes little endian:  sof=0xAA u8 | version=0 u8 | length u16 (whole packet)
        | seq u32 | cmd_id u16 (0 = search) | cmd_type u8 (0 = cmd, 1 = ack)
        | sender u8 (0 = host, 1 = lidar) | rsvd[6] | crc16 u16 | crc32 u32
  crc16 covers header bytes 0..17: SDK "FastCRC16::ccitt" = CRC-16/CCITT-FALSE
        (poly 0x1021, init 0xFFFF, no reflection, check 0x29B1; confirmed by
        running the SDK's FastCRC code), stored little endian.
  crc32 covers the payload (zlib CRC-32), 0 when the payload is empty.
  search request: header only, length 24.
  search ack payload, 24 bytes: ret_code u8 | dev_type u8 | sn char[16] |
        lidar_ip u8[4] | cmd_port u16.   dev_type 9 = Mid360, 15 = HAP.

Apply only ever ADDS a secondary address to the NIC the lidar answered on
(`nmcli connection modify <con> +ipv4.addresses <ip>/24`, then `connection up`;
without NetworkManager `ip addr add`, which does not persist).  It never replaces
addresses and never touches method, gateway or DNS; an address already present is a
no-op.  It never runs as root itself: it calls sudo per command.

Python stdlib only.  Network and command I/O sit behind ``Host`` and ``search``
so tests can feed canned bytes and a fake nmcli/ip runner.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import select
import shlex
import socket
import struct
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

SEARCH_PORT = 56000
SOF = 0xAA
HEADER_LEN = 24
CMD_SEARCH = 0
TYPE_CMD, TYPE_ACK = 0, 1
SENDER_HOST, SENDER_LIDAR = 0, 1
DETECTION_FMT = "<BB16s4BH"        # ret_code, dev_type, sn[16], ip[4], cmd_port
DETECTION_LEN = struct.calcsize(DETECTION_FMT)   # 24
DEV_TYPES = {0: "Hub", 1: "Mid40", 2: "Tele", 3: "Horizon", 6: "Mid70", 7: "Avia",
             9: "Mid360", 10: "Industrial HAP", 15: "HAP", 16: "PA"}
FALLBACK_START = 50
SCHEMA = 1


# ------------------------------------------------------------------ protocol

def crc16_sdk(data: bytes) -> int:
    """The SDK's FastCRC16::ccitt: CRC-16/CCITT-FALSE."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def build_search_request(seq: int = 0) -> bytes:
    head = struct.pack("<BBHIHBB6s", SOF, 0, HEADER_LEN, seq & 0xFFFF, CMD_SEARCH,
                       TYPE_CMD, SENDER_HOST, b"\0" * 6)
    return head + struct.pack("<HI", crc16_sdk(head), 0)


def parse_search_reply(buf: bytes) -> dict | None:
    """One search ack -> {serial, ip, dev_type, model, cmd_port}; None if it is not one.

    Both CRCs are checked; a lidar answering with ret_code != 0 is not reported."""
    if len(buf) < HEADER_LEN + DETECTION_LEN:
        return None
    sof, ver, length, _seq, cmd_id, cmd_type, sender, _rsvd, crc16, crc32 = \
        struct.unpack("<BBHIHBB6sHI", buf[:HEADER_LEN])
    if sof != SOF or ver != 0 or cmd_id != CMD_SEARCH or cmd_type != TYPE_ACK:
        return None
    if length < HEADER_LEN + DETECTION_LEN or length > len(buf):
        return None
    if crc16 != crc16_sdk(buf[:18]):
        return None
    payload = buf[HEADER_LEN:length]
    if crc32 != (zlib.crc32(payload) & 0xFFFFFFFF):
        return None
    ret, dev, sn, a, b, c, d, port = struct.unpack(DETECTION_FMT, payload[:DETECTION_LEN])
    if ret != 0:
        return None
    serial = sn.split(b"\0", 1)[0].decode("ascii", "replace")
    return {"serial": serial, "ip": f"{a}.{b}.{c}.{d}", "dev_type": dev,
            "model": DEV_TYPES.get(dev, f"type {dev}"), "cmd_port": port}


# ---------------------------------------------------------------------- host

@dataclass
class Step:
    argv: list
    comment: str | None = None

    def display(self) -> str:
        s = shlex.join(self.argv)
        return s + (f"  # {self.comment}" if self.comment else "")


class Host:
    """All I/O.  Read-only except ``execute`` (reached only from --apply)."""

    def run(self, argv: list, timeout: float = 15) -> tuple:
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            return 127, ""
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 1, exc.__class__.__name__
        return p.returncode, p.stdout

    def euid(self) -> int:
        return os.geteuid()

    def execute(self, step: Step) -> int:
        return subprocess.run(step.argv).returncode

    def search(self, ifaces: list, timeout: float) -> tuple:
        """Broadcast the search out of each interface. -> (replies, error, per-iface errors).

        replies: [{..parse_search_reply.., 'interface': name, 'from': src ip}]."""
        req = build_search_request()
        found, errs = [], {}
        for name in ifaces:
            # one socket per interface: an unprivileged SO_BINDTODEVICE works only on a fresh socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, name.encode() + b"\0")
                except OSError as exc:
                    errs[name] = f"SO_BINDTODEVICE: {exc.strerror}"
                    continue
                try:
                    sock.bind(("0.0.0.0", SEARCH_PORT))
                except OSError as exc:
                    return found, (f"cannot bind UDP {SEARCH_PORT} ({exc.strerror}); a running Livox driver "
                                   f"or another SDK program holds it. Stop it and re-run."), errs
                try:
                    sock.sendto(req, ("255.255.255.255", SEARCH_PORT))
                except OSError as exc:
                    errs[name] = f"send: {exc.strerror}"
                    continue
                end = time.monotonic() + timeout
                while True:
                    left = end - time.monotonic()
                    if left <= 0 or not select.select([sock], [], [], left)[0]:
                        break
                    data, src = sock.recvfrom(2048)
                    r = parse_search_reply(data)
                    if r:
                        found.append({**r, "interface": name, "from": src[0]})
            finally:
                sock.close()
        return found, None, errs


# ------------------------------------------------------------------- parsing

def parse_ip_o_addr(text: str) -> dict:
    """`ip -o -4 addr show` -> {iface: [cidr, ...]}."""
    out: dict = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 4 and p[2] == "inet":
            out.setdefault(p[1], []).append(p[3])
    return out


def split_terse(line: str) -> list:
    """Split an `nmcli -t` line on unescaped colons."""
    parts, cur, i = [], "", 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            cur += line[i + 1]
            i += 2
            continue
        if ch == ":":
            parts.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    parts.append(cur)
    return parts


def parse_nmcli_active(text: str) -> dict:
    """`nmcli -t -f NAME,UUID,DEVICE connection show --active` -> {device: (name, uuid)}."""
    out = {}
    for line in text.splitlines():
        p = split_terse(line)
        if len(p) >= 3 and p[2]:
            out.setdefault(p[2], (p[0], p[1]))
    return out


def local_addrs(host: Host) -> dict:
    rc, out = host.run(["ip", "-o", "-4", "addr", "show"])
    return parse_ip_o_addr(out) if rc == 0 else {}


def usable_ifaces(addrs: dict) -> list:
    return [n for n in addrs if n != "lo" and not n.startswith(("veth", "docker", "br-"))]


def _net24(ip: str) -> ipaddress.IPv4Network:
    return ipaddress.ip_network(f"{ip}/24", strict=False)


# ---------------------------------------------------------------- suggestion

def suggest_host_address(lidar_ip: str, taken: set, answers=lambda ip: False) -> tuple:
    """(address, reason).  Lidar last octet minus 100 if free, else first free from .50.

    ``taken``: addresses on local NICs / known lidars; ``answers(ip)`` -> True if ARP/ping gets an answer."""
    net = _net24(lidar_ip)
    last = int(lidar_ip.rsplit(".", 1)[1])
    pre = str(net.network_address).rsplit(".", 1)[0]

    def free(n):
        ip = f"{pre}.{n}"
        return 1 <= n <= 254 and ip != lidar_ip and ip not in taken and not answers(ip)

    pref = last - 100
    if pref >= 1 and free(pref):
        return f"{pre}.{pref}", f"lidar .{last} minus 100 is free"
    why = f"lidar .{last} minus 100 " + ("is not available" if pref >= 1 else "is not a valid host") + "; "
    for n in range(FALLBACK_START, 255):
        if free(n):
            return f"{pre}.{n}", why + f"first free address from .{FALLBACK_START} is .{n}"
    return "", why + "no free address found in the /24"


# --------------------------------------------------------------------- apply

def apply_steps(host: Host, nic: str, ip: str, active: dict) -> tuple:
    """([Step], note).  Only ever adds ``ip``/24; never a bare ipv4.addresses."""
    cidr = f"{ip}/24"
    if nic in active:
        con, uuid = active[nic]
        return ([Step(["sudo", "nmcli", "connection", "modify", "uuid", uuid, "+ipv4.addresses", cidr],
                      f"adds a secondary address to NetworkManager profile {con!r}"),
                 Step(["sudo", "nmcli", "connection", "up", "uuid", uuid], f"re-activate {con!r}")],
                f"persistent via NetworkManager profile {con!r}")
    return ([Step(["sudo", "ip", "addr", "add", cidr, "dev", nic],
                  "no NetworkManager profile for this NIC: not persistent")],
            "not persistent (no active NetworkManager profile on this NIC)")


# --------------------------------------------------------------------- build

def ping_ok(host: Host, ip: str) -> bool:
    return host.run(["ping", "-c", "1", "-W", "1", ip], timeout=3)[0] == 0


def sweep(host: Host, addrs: dict, ifaces: list, own: set) -> list:
    """Fallback: ping every host of each local /24. -> [{ip, interface}] (unverified)."""
    jobs = []
    for n in ifaces:
        for cidr in addrs.get(n, []):
            if cidr.startswith("169.254."):
                continue
            for h in _net24(cidr.split("/")[0]).hosts():
                if str(h) not in own:
                    jobs.append((n, str(h)))
    with ThreadPoolExecutor(64) as ex:
        res = list(ex.map(lambda j: ping_ok(host, j[1]), jobs))
    return [{"ip": ip, "interface": n} for (n, ip), ok in zip(jobs, res) if ok]


def build_report(host: Host, args) -> dict:
    addrs = local_addrs(host)
    ifaces = [args.nic] if args.nic else usable_ifaces(addrs)
    rc, nm = host.run(["nmcli", "-t", "-f", "NAME,UUID,DEVICE", "connection", "show", "--active"])
    active = parse_nmcli_active(nm) if rc == 0 else {}
    own = {c.split("/")[0] for lst in addrs.values() for c in lst}

    method, error, nic_errs, found = "sdk2-search", None, {}, []
    if not ifaces:
        error = "no IPv4 interface is up"
    else:
        found, error, nic_errs = host.search(ifaces, args.timeout)
    unverified = []
    if (error and ifaces) or args.sweep:
        method = "ping-sweep"
        for r in sweep(host, addrs, ifaces, own):
            if not any(f["ip"] == r["ip"] for f in found):
                unverified.append(r)

    seen, lidars = set(), []
    for r in found:
        if r["ip"] in seen:
            continue
        seen.add(r["ip"])
        net = _net24(r["ip"])
        mine = [c.split("/")[0] for n in ([r["interface"]] if r["interface"] in addrs else addrs)
                for c in addrs.get(n, []) if ipaddress.ip_address(c.split("/")[0]) in net]
        reach = bool(mine)
        item = {"ip": r["ip"], "serial": r["serial"], "model": r["model"], "dev_type": r["dev_type"],
                "cmd_port": r["cmd_port"], "interface": r["interface"], "subnet": str(net),
                "host_address": mine[0] if mine else None,
                "status": "ok" if reach else "needs-address",
                "suggested_host_address": None, "suggest_reason": None, "steps": [], "persistent": None}
        if not reach:
            taken = own | {x["ip"] for x in found}
            sug, why = suggest_host_address(r["ip"], taken, lambda ip: ping_ok(host, ip))
            item["suggested_host_address"], item["suggest_reason"] = sug, why
            if sug:
                steps, note = apply_steps(host, r["interface"], sug, active)
                item["steps"] = [s.display() for s in steps]
                item["persistent"] = not note.startswith("not persistent")
                item["note"] = note
        lidars.append(item)

    need = [x for x in lidars if x["status"] == "needs-address"]
    status = ("error" if error and not unverified and not lidars else
              "none-found" if not lidars else "needs-address" if need else "ok")
    return {"schema": SCHEMA, "status": status,
            "discovery": {"method": method, "port": SEARCH_PORT, "error": error,
                          "interfaces": ifaces, "interface_errors": nic_errs, "timeout_s": args.timeout},
            "lidars": lidars, "unverified": unverified, "host_addresses": addrs,
            "active_connections": {k: v[0] for k, v in active.items()},
            "would_run": [s for x in need for s in x["steps"]],
            "exit": 0 if status == "ok" else 1}


def select_apply(report: dict, args) -> tuple:
    """(lidar item | None, message).  Never guesses between several."""
    need = [x for x in report["lidars"] if x["status"] == "needs-address" and x["suggested_host_address"]]
    if args.lidar_ip:
        need = [x for x in report["lidars"] if x["ip"] == args.lidar_ip]
        if not need:
            return None, f"lidar {args.lidar_ip} was not discovered"
        if need[0]["status"] == "ok":
            return None, (f"nothing to do: this host already has {need[0]['host_address']} "
                          f"on {need[0]['subnet']}")
    if not need:
        return None, "nothing to do" if report["status"] == "ok" else "no lidar needs an address"
    if len(need) > 1:
        return None, "several lidars need an address; pick one with --lidar-ip " + \
            ", ".join(x["ip"] for x in need)
    return need[0], ""


def format_report(rep: dict) -> str:
    d = rep["discovery"]
    lines = [f"discovery: {d['method']} on {', '.join(d['interfaces']) or '-'}"]
    if d["error"]:
        lines.append(f"  ERROR: {d['error']}")
    for n, e in d["interface_errors"].items():
        lines.append(f"  {n}: {e}")
    for x in rep["lidars"]:
        lines.append(f"{x['model']} {x['serial']}  {x['ip']}  via {x['interface']}  {x['status'].upper()}")
        if x["status"] == "ok":
            lines.append(f"  host already has {x['host_address']} on {x['subnet']}")
        else:
            lines.append(f"  host has no address on {x['subnet']}")
            if x["suggested_host_address"]:
                lines.append(f"  suggest {x['suggested_host_address']}/24 ({x['suggest_reason']})")
                lines += [f"    {s}" for s in x["steps"]]
    for u in rep["unverified"]:
        lines.append(f"unverified (answers ping, not confirmed as a lidar): {u['ip']} via {u['interface']}")
    if not rep["lidars"] and not rep["unverified"]:
        lines.append("no lidar found")
    return "\n".join(lines)


def _ask_tty(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="machine-readable report (read-only)")
    ap.add_argument("--apply", action="store_true", help="add the suggested host address (sudo per command)")
    ap.add_argument("--dry-run", action="store_true", help="with --apply: print the commands, run nothing")
    ap.add_argument("--nic", help="only search out of this interface")
    ap.add_argument("--lidar-ip", help="which lidar --apply is for when several need an address")
    ap.add_argument("--host-address", help="use this address instead of the suggestion")
    ap.add_argument("--timeout", type=float, default=1.0, help="seconds to wait per interface (default 1)")
    ap.add_argument("--sweep", action="store_true", help="also ping-sweep the local /24s")
    return ap


def main(argv=None, host: Host | None = None, ask=_ask_tty, out=None) -> int:
    host = host or Host()
    out = out or sys.stdout
    ap = build_parser()
    args = ap.parse_args(argv)
    executing = args.apply and not args.dry_run
    if args.json and executing:
        ap.error("--json reports only; combine it with --apply only together with --dry-run")
    if args.host_address:
        try:
            ipaddress.IPv4Address(args.host_address)
        except ValueError:
            ap.error(f"--host-address {args.host_address!r} is not an IPv4 address")
    if host.euid() == 0 and args.apply:
        print("livox_host_setup: do not run --apply as root or with sudo; it calls sudo itself "
              "(report mode may run as root)", file=sys.stderr)
        return 2
    rep = build_report(host, args)
    plan, msg = (None, "")
    steps: list = []
    if args.apply:
        plan, msg = select_apply(rep, args)
        if plan is not None:
            ip = args.host_address or plan["suggested_host_address"]
            addrs = local_addrs(host)
            if any(c.split("/")[0] == ip for c in addrs.get(plan["interface"], [])):
                plan, msg = None, f"nothing to do: {ip} is already on {plan['interface']}"
            else:
                rc, nm = host.run(["nmcli", "-t", "-f", "NAME,UUID,DEVICE", "connection", "show", "--active"])
                steps, _ = apply_steps(host, plan["interface"], ip, parse_nmcli_active(nm) if rc == 0 else {})
    if args.json:
        if args.apply:
            rep["apply"] = {"message": msg, "would_run": [s.display() for s in steps]}
        json.dump(rep, out, indent=2)
        out.write("\n")
        return rep["exit"]
    print(format_report(rep), file=out)
    if not args.apply:
        return rep["exit"]
    if not steps:
        print(f"\n--apply: {msg}", file=out)
        return 0 if "nothing to do" in msg else 1
    print("\n" + ("would run (nothing executed):" if args.dry_run else "will run:"), file=out)
    for s in steps:
        print("  " + s.display(), file=out)
    if args.dry_run:
        return 0
    if not ask("Add this address now? [y/N] "):
        print("skipped", file=out)
        return 1
    for s in steps:
        rc = host.execute(s)
        if rc != 0:
            print(f"command failed (exit {rc}): {shlex.join(s.argv)}", file=out)
            return 1
    print("done; re-run without --apply to confirm", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
