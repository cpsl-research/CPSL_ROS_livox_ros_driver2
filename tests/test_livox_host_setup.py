"""Hardware-free tests for tools/setup/livox_host_setup.py (stdlib unittest; pytest collects them too).

The request bytes were produced by compiling the Livox-SDK2's own FastCRC16/32 and the
SDK's header layout (2026-10-15).  Reply bytes are built here in the SDK's layout; no
real lidar capture is involved, so these tests prove the parser/logic, not the hardware."""
import io
import json
import struct
import sys
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/setup"))
import livox_host_setup as lh  # noqa: E402

SDK_REQUEST = bytes.fromhex("aa0018000000000000000000000000000000c86400000000")  # SDK-compiled
IP_O = ("1: lo    inet 127.0.0.1/8 scope host lo\n"
        "2: enp3s0    inet 192.168.186.3/24 brd 192.168.186.255 scope global enp3s0\n"
        "2: enp3s0    inet 192.168.33.30/24 brd 192.168.33.255 scope global secondary enp3s0\n")
NM_ACTIVE = "Wired connection 1:aaaa-bbbb:enp3s0\nlo:1111:lo\n"


def reply(ip=(192, 168, 1, 117), sn=b"47MDL1H0010117", dev=9, ret=0, port=56100, bad_crc=False):
    payload = struct.pack(lh.DETECTION_FMT, ret, dev, sn, *ip, port)
    head = struct.pack("<BBHIHBB6s", 0xAA, 0, 24 + len(payload), 0, 0, 1, 1, b"\0" * 6)
    c16 = lh.crc16_sdk(head) ^ (1 if bad_crc else 0)
    return head + struct.pack("<HI", c16, zlib.crc32(payload) & 0xFFFFFFFF) + payload


class FakeHost(lh.Host):
    def __init__(self, ip_o=IP_O, nm=NM_ACTIVE, replies=(), error=None, pingable=(), euid=1000):
        self.ip_o, self.nm, self.replies, self.error = ip_o, nm, list(replies), error
        self.pingable, self._euid, self.executed, self.calls = set(pingable), euid, [], []

    def run(self, argv, timeout=15):
        self.calls.append(tuple(argv))
        if argv[:3] == ["ip", "-o", "-4"]:
            return 0, self.ip_o
        if argv[0] == "nmcli":
            return (0, self.nm) if self.nm is not None else (127, "")
        if argv[0] == "ping":
            return (0, "") if argv[-1] in self.pingable else (1, "")
        return 127, ""

    def search(self, ifaces, timeout):
        return list(self.replies), self.error, {}

    def euid(self):
        return self._euid

    def execute(self, step):
        self.executed.append(step.argv)
        return 0


def found(**kw):
    r = lh.parse_search_reply(reply(**kw))
    return {**r, "interface": "enp3s0", "from": r["ip"]}


def run_main(host, *argv, ask=lambda p: True):
    out = io.StringIO()
    rc = lh.main(list(argv), host=host, ask=ask, out=out)
    return rc, out.getvalue()


class Protocol(unittest.TestCase):
    def test_request_matches_sdk_bytes(self):
        self.assertEqual(lh.build_search_request(), SDK_REQUEST)

    def test_parse_reply(self):
        r = lh.parse_search_reply(reply())
        self.assertEqual((r["ip"], r["serial"], r["model"], r["cmd_port"]),
                         ("192.168.1.117", "47MDL1H0010117", "Mid360", 56100))

    def test_rejects_garbage(self):
        self.assertIsNone(lh.parse_search_reply(b""))
        self.assertIsNone(lh.parse_search_reply(SDK_REQUEST))            # our own request, cmd type
        self.assertIsNone(lh.parse_search_reply(reply(bad_crc=True)))
        self.assertIsNone(lh.parse_search_reply(reply(ret=1)))
        self.assertIsNone(lh.parse_search_reply(reply()[:30]))

    def test_unknown_dev_type_named(self):
        self.assertEqual(lh.parse_search_reply(reply(dev=99))["model"], "type 99")


class Suggestion(unittest.TestCase):
    def test_minus_100(self):
        self.assertEqual(lh.suggest_host_address("192.168.1.117", set())[0], "192.168.1.17")

    def test_taken_falls_back(self):
        ip, why = lh.suggest_host_address("192.168.1.117", {"192.168.1.17"})
        self.assertEqual(ip, "192.168.1.50")
        self.assertIn("not available", why)

    def test_answers_to_ping_counts_as_taken(self):
        ip, _ = lh.suggest_host_address("192.168.1.117", set(), lambda i: i in ("192.168.1.17", "192.168.1.50"))
        self.assertEqual(ip, "192.168.1.51")

    def test_low_last_octet(self):
        ip, why = lh.suggest_host_address("192.168.1.50", set())
        self.assertEqual(ip, "192.168.1.51")      # .50 is the lidar itself
        self.assertIn("not a valid host", why)


class Report(unittest.TestCase):
    def test_needs_address(self):
        rc, out = run_main(FakeHost(replies=[found()]), "--json")
        rep = json.loads(out)
        self.assertEqual(rc, 1)
        self.assertEqual(rep["status"], "needs-address")
        x = rep["lidars"][0]
        self.assertEqual((x["ip"], x["serial"], x["interface"], x["suggested_host_address"]),
                         ("192.168.1.117", "47MDL1H0010117", "enp3s0", "192.168.1.17"))
        self.assertTrue(x["persistent"])
        self.assertTrue(any("+ipv4.addresses" in s and "192.168.1.17/24" in s for s in x["steps"]))
        self.assertEqual(rep["host_addresses"]["enp3s0"], ["192.168.186.3/24", "192.168.33.30/24"])

    def test_already_reachable(self):
        ip_o = IP_O + "2: enp3s0    inet 192.168.1.17/24 scope global secondary enp3s0\n"
        rc, out = run_main(FakeHost(ip_o=ip_o, replies=[found()]), "--json")
        rep = json.loads(out)
        self.assertEqual((rc, rep["status"], rep["lidars"][0]["host_address"]), (0, "ok", "192.168.1.17"))

    def test_none_found_and_error_are_stated(self):
        rep = json.loads(run_main(FakeHost(), "--json")[1])
        self.assertEqual(rep["status"], "none-found")
        h = FakeHost(error="cannot bind UDP 56000 (Address already in use)", pingable={"192.168.186.9"})
        rep = json.loads(run_main(h, "--json")[1])
        self.assertIn("56000", rep["discovery"]["error"])
        self.assertEqual(rep["discovery"]["method"], "ping-sweep")
        self.assertEqual(rep["unverified"], [{"ip": "192.168.186.9", "interface": "enp3s0"}])

    def test_no_networkmanager_fallback_not_persistent(self):
        rep = json.loads(run_main(FakeHost(nm=None, replies=[found()]), "--json")[1])
        x = rep["lidars"][0]
        self.assertFalse(x["persistent"])
        self.assertEqual(x["steps"][0].split("  #")[0], "sudo ip addr add 192.168.1.17/24 dev enp3s0")


class Apply(unittest.TestCase):
    def test_adds_never_overwrites(self):
        h = FakeHost(replies=[found()])
        rc, _ = run_main(h, "--apply")
        self.assertEqual(rc, 0)
        self.assertEqual(len(h.executed), 2)
        mod = h.executed[0]
        self.assertIn("+ipv4.addresses", mod)
        self.assertEqual(mod[mod.index("+ipv4.addresses") + 1], "192.168.1.17/24")
        for argv in h.executed:
            self.assertNotIn("ipv4.addresses", argv)           # no bare overwrite
            self.assertFalse({"ipv4.method", "ipv4.gateway", "ipv4.dns", "-ipv4.addresses"} & set(argv))
            self.assertEqual(argv[0], "sudo")

    def test_second_run_is_noop(self):
        ip_o = IP_O + "2: enp3s0    inet 192.168.1.17/24 scope global secondary enp3s0\n"
        h = FakeHost(ip_o=ip_o, replies=[found()])
        rc, out = run_main(h, "--apply")
        self.assertEqual((rc, h.executed), (0, []))
        self.assertIn("nothing to do", out)

    def test_dry_run_runs_nothing(self):
        h = FakeHost(replies=[found()])
        rc, out = run_main(h, "--apply", "--dry-run")
        self.assertEqual((rc, h.executed), (0, []))
        self.assertIn("+ipv4.addresses", out)

    def test_declined_runs_nothing(self):
        h = FakeHost(replies=[found()])
        run_main(h, "--apply", ask=lambda p: False)
        self.assertEqual(h.executed, [])

    def test_several_lidars_not_guessed(self):
        two = [found(), found(ip=(192, 168, 2, 120), sn=b"47MDL1H0010120")]
        h = FakeHost(replies=two)
        rc, out = run_main(h, "--apply")
        self.assertEqual((rc, h.executed), (1, []))
        self.assertIn("--lidar-ip", out)
        rc, _ = run_main(h, "--apply", "--lidar-ip", "192.168.2.120")
        self.assertEqual(rc, 0)
        self.assertIn("192.168.2.20/24", h.executed[0])

    def test_root_refused(self):
        rc, _ = run_main(FakeHost(replies=[found()], euid=0), "--apply")
        self.assertEqual(rc, 2)

    def test_json_apply_needs_dry_run(self):
        with self.assertRaises(SystemExit):
            run_main(FakeHost(), "--json", "--apply")


if __name__ == "__main__":
    unittest.main()
