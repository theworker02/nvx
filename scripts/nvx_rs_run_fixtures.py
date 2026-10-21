"""Record the golden scenarios of `nvx.py run` for nvx-rs.

nvx-rs run ports `nvx.py run` (microsoft/nvx#638), and must keep its options,
messages, outputs, and exit statuses. This script records what the Python
implementation of `nvx.py run` does, run as a command in fresh directories with
the fake OpenVMM of nvx-rs, nvx-fake-openvmm, as the directory's OpenVMM, into
the scenarios of nvx-rs/tests/fixtures/cli.json whose names start with "run, ",
which tests/cli.rs replays with nvx-rs, as scripts/nvx_rs_fixtures.py recorded
`nvx.py sandbox` until microsoft/nvx#359 retired it:

- the usage errors and help of the run parser;
- the refusals of command_run;
- dry runs of every guest, of cold boots and restores, of shares, file shares,
  and their policy paths and owners, of the network options and egress policy
  files, of CPU profiles and outcome reports, and of the 1024-byte budget of
  the kernel command line; and
- runs of the fake OpenVMM, and the errors of missing artifacts.

Each step records its exit status, standard output and error, and the files
that it names, with the root, process IDs, and random identifiers as
placeholders, and the program names of nvx.py as those of nvx-rs, whose run
parser's name has the same length, so that its usage wraps alike. The script
records without the CPU-profile guidance, which depends on the host's CPU. Its
`posix` and `windows` sections hold the scenarios of each kind of host,
recorded there: the script replaces the run scenarios of the section of the
host that runs it, and keeps every other scenario and the other section.

The outputs depend on the interpreter: record with CPython 3.14.5 or later,
whose argparse quotes the choices of an invalid choice, as nvx-rs does, from
any directory:

    python3 scripts/nvx_rs_run_fixtures.py            # record this host's runs
    python3 scripts/nvx_rs_run_fixtures.py --check    # fail if they would change
    python3 scripts/nvx_rs_run_fixtures.py --parity   # replay them with nvx-rs

It builds nvx-rs and nvx-fake-openvmm with cargo, and runs nvx.py with that
nvx-rs, which looks up its artifacts and compiles its egress policy files.
--parity plays the recorded run scenarios of this host with nvx-rs run, and
fails if any step differs. The script goes with the Python implementation of
`nvx.py run` once nvx.py runs nvx-rs run.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvx_tools import sandbox  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
NVX_RS_DIR = REPO_ROOT / "nvx-rs"
CLI_FIXTURE = NVX_RS_DIR / "tests" / "fixtures" / "cli.json"
CLI_PLATFORM = "windows" if os.name == "nt" else "posix"
CLI_PLATFORMS = ("posix", "windows")
EXE = ".exe" if os.name == "nt" else ""
# The names of the scenarios that this script records start with this prefix.
PREFIX = "run, "
# What recorded the scenarios of the fixture: the scenarios of `nvx.py sandbox`,
# and the run scenarios of this script.
GENERATOR = (
    "scripts/nvx_rs_fixtures.py, retired by microsoft/nvx#359, and "
    "scripts/nvx_rs_run_fixtures.py"
)
# The top-level usage of nvx-rs, which replaces that of nvx.py in the usage
# errors that the top-level parser reports, such as unrecognized arguments.
NVX_RS_USAGE = (
    "usage: nvx-rs [-h] [--root PATH] [--prog PROG]\n"
    "              {artifacts,egress,run,sandbox} ..."
)
CLI_STEP_TIMEOUT = 120
CLI_OPENVMM = "openvmm/target/release/openvmm{exe}"
# The files whose permissions the replay records on POSIX hosts.
CLI_MODE_NAMES = frozenset(
    ("state", "config.json", "runtime.json", "control.capability", "lifecycle.lock")
)
VOLATILE = (
    (re.compile(r'("(?:pid|start_time)": )\d+'), r"\1{int}"),
    (re.compile(r"openvmm-microvm-[0-9a-f]{32}"), "openvmm-microvm-{hex32}"),
    (re.compile(r'("(?:operation_id|instance_id)": ")[0-9a-f]{32}"'), r'\1{hex32}"'),
)
# Runs `nvx.py` with the nvx-rs that NVX_RS names and the artifact root
# NVX_FIXTURE_ROOT, without the CPU-profile guidance, which depends on the
# host's CPU.
PYTHON_RUN = """\
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["NVX_FIXTURE_SCRIPTS"])
import nvx
from nvx_tools import nvx_rs

nvx_rs.ROOT = Path(os.environ["NVX_FIXTURE_ROOT"])
nvx.host_cpu_signature = lambda: None
sys.argv[0] = "nvx.py"
raise SystemExit(nvx.main(sys.argv[1:]))
"""

Entry = dict[str, object]
Step = dict[str, object]
Scenario = dict[str, object]

# The initramfs images that the tree of the fixture lacks.
UBUNTU = {"build/initramfs-ubuntu.cpio.gz": "ubuntu\n"}
AZURELINUX = {"build/initramfs-azurelinux.cpio.gz": "azurelinux\n"}
SNAPSHOT = {"snapshot/state.bin": "snapshot\n"}
POLICY = {
    "policy.json": json.dumps(
        {
            "allow": [
                {"cidr": "192.0.2.0/24", "except": ["192.0.2.128/25"]},
                {"cidr": "2001:db8::/32", "protocol": "tcp", "port": 443},
            ],
            "deny": [{"cidr": "198.51.100.0/24", "protocol": "udp"}],
        }
    )
}
# Host paths with commas, which the mode that follows the last comma allows.
COMMAS = {
    "alpha,beta/data.txt": "data\n",
    "2026, final/data.txt": "data\n",
    "settings,v2.json": "{}\n",
}


def _r(path: str = "") -> str:
    """Return the native path of `path` below the scenario root."""
    return "{root}" + (os.sep + path.replace("/", os.sep) if path else "")


def _run(
    *argv: str,
    write: dict[str, str] | None = None,
    remove: Sequence[str] = (),
    check: Sequence[str] = (),
    env: dict[str, str] | None = None,
) -> Step:
    """Return a step that runs `nvx-rs run ARGV` after writing and removing
    files, and records the files of `check`."""
    step: Step = {"argv": ["run", *argv]}
    if write:
        step["write"] = write
    if remove:
        step["remove"] = list(remove)
    if check:
        step["check"] = list(check)
    if env:
        step["env"] = env
    return step


def _scenario(
    name: str, *steps: Step, platforms: Sequence[str] = CLI_PLATFORMS
) -> Scenario | None:
    """Return a scenario of `steps` in a fresh root, if it runs on this platform."""
    if CLI_PLATFORM not in platforms:
        return None
    return {"name": f"{PREFIX}{name}", "steps": list(steps)}


def _usage_scenarios() -> list[Scenario | None]:
    """Return the usage errors of the run parser, and its help."""
    cases: list[tuple[str, list[str]]] = [
        ("help", ["-h"]),
        ("help after options", ["--guest", "ubuntu", "--help"]),
        ("help before an invalid value", ["--help", "--memory-mib", "x"]),
        ("invalid value before help", ["--memory-mib", "x", "--help"]),
        ("invalid guest", ["--guest", "debian"]),
        ("invalid second guest", ["--guest", "alpine", "--guest", "debian"]),
        ("invalid hypervisor", ["--hypervisor", "hyperv"]),
        ("invalid machine", ["--machine", "microvm-v2"]),
        ("memory not an integer", ["--memory-mib", "x"]),
        ("memory not a whole number", ["--memory-mib", "1.5"]),
        ("memory with a space", ["--memory-mib", "1 2"]),
        ("memory with other digits", ["--memory-mib", "\u0661\u0662\u0668x"]),
        ("memory capacity not an integer", ["--memory-capacity-mib", "1e3"]),
        ("restore memory not an integer", ["--restore-memory-mib", ""]),
        ("processors not an integer", ["--processors", "two"]),
        ("processors not a choice", ["--processors", "3"]),
        ("processors converted before the choice", ["--processors", "1_6"]),
        ("processors with a leading zero", ["--processors", "03"]),
        ("negative processors", ["--processors", "-5"]),
        ("restore processors not an integer", ["--restore-processors", "x"]),
        ("restore processors not a choice", ["--restore-processors", "16"]),
        ("invalid mount owner", ["--mount-owner", "root"]),
        ("invalid network profile", ["--network-profile", "open"]),
        ("invalid network egress", ["--network-egress", "block"]),
        ("invalid network ingress", ["--network-ingress", "open"]),
        ("invalid host loopback", ["--host-loopback", "yes"]),
        ("missing value", ["--memory-mib"]),
        ("missing path", ["--restore-snapshot"]),
        ("option as value", ["--cpu-profile", "--dry-run"]),
        ("dash value", ["--cmdline", "-x"]),
        ("flag with value", ["--dry-run=1"]),
        ("unrecognized option", ["--bogus"]),
        ("unrecognized short option", ["-x"]),
        ("unrecognized positional", ["extra"]),
        ("unrecognized after separator", ["--", "extra"]),
        ("unrecognized after options", ["--dry-run", "--guest", "alpine", "extra"]),
    ]
    return [_scenario(f"usage: {name}", _run(*argv)) for name, argv in cases]


def _budget_cases() -> list[tuple[str, list[str]]]:
    """Return shares whose kernel command line fits the 1024-byte budget
    exactly, exceeds it by one byte, and exceeds it in UTF-8 bytes only."""
    shares = [("/a", "ro", "directory"), ("/b", "rw", "directory")]
    tokens = " ".join(sandbox.aggregate_share_tokens(shares))
    fragment = sandbox.aggregate_command_line_fragment(["ro", "rw"])
    # The user's text, a space, and the tokens, then the fragment and a NUL.
    longest = sandbox.SANDBOX_COMMAND_LINE_MAX_SIZE - len(tokens) - len(fragment) - 2
    mounts = ["--mount", f"/a,{_r('share')}", "--mount", f"/b,{_r('share2')},rw"]
    return [
        ("fits", [*mounts, "--cmdline", "x" * longest]),
        ("exceeded by a byte", [*mounts, "--cmdline", "x" * (longest + 1)]),
        ("in UTF-8 bytes", [*mounts, "--cmdline", "\u00e9" * (longest // 2 + 1)]),
    ]


def _validation_scenarios() -> list[Scenario | None]:
    """Return the refusals of command_run, each in its own root."""
    several = ["--mount", f"/a,{_r('share')}", "--mount", f"/b,{_r('share2')}"]
    single_file = ["--mount", f"/etc/single.txt,{_r('single.txt')}"]
    cases: list[tuple[str, list[str], dict[str, str] | None, Sequence[str]]] = [
        ("net without profile", ["--net", "10.0.0.2/24"], None, ()),
        ("profile without net", ["--network-profile", "portable"], None, ()),
        (
            "ready path without snapshot",
            ["--restore-ready-path", _r("ready.sock")],
            None,
            (),
        ),
        (
            "restore processors without snapshot",
            ["--restore-processors", "1"],
            None,
            (),
        ),
        ("restore memory without snapshot", ["--restore-memory-mib", "1024"], None, ()),
        (
            "memory capacity of a restore",
            ["--restore-snapshot", _r("snapshot"), "--memory-capacity-mib", "1024"],
            None,
            (),
        ),
        (
            "Ubuntu guest of a restore",
            ["--guest", "ubuntu", "--restore-snapshot", _r("snapshot")],
            None,
            (),
        ),
        (
            "Azure Linux guest of a restore",
            ["--restore-snapshot", _r("snapshot"), "--guest", "azurelinux"],
            None,
            (),
        ),
        (
            "memory capacity below memory",
            ["--memory-mib", "512", "--memory-capacity-mib", "256"],
            None,
            (),
        ),
        (
            "memory capacity below the guest default",
            ["--guest", "ubuntu", "--memory-capacity-mib", "256"],
            UBUNTU,
            (),
        ),
        (
            "restore processors above processors",
            [
                "--processors",
                "2",
                "--restore-snapshot",
                _r("snapshot"),
                "--restore-processors",
                "4",
            ],
            None,
            (),
        ),
        (
            "policy file with allow rules",
            [
                "--network-egress",
                "deny",
                "--network-egress-policy-file",
                _r("policy.json"),
                "--network-egress-allow",
                "10.0.0.0/8",
            ],
            POLICY,
            (),
        ),
        (
            "policy file with deny rules",
            [
                "--network-egress-deny",
                "10.0.0.0/8",
                "--network-egress-policy-file",
                _r("policy.json"),
            ],
            POLICY,
            (),
        ),
        (
            "policy file without egress",
            ["--network-egress-policy-file", _r("policy.json")],
            POLICY,
            (),
        ),
        (
            "policy file missing",
            [
                "--network-egress",
                "deny",
                "--network-egress-policy-file",
                _r("./missing.json"),
            ],
            None,
            (),
        ),
        (
            "policy file directory",
            ["--network-egress", "deny", "--network-egress-policy-file", _r("share")],
            None,
            (),
        ),
        (
            "policy file invalid",
            [
                "--network-egress",
                "allow",
                "--network-egress-policy-file",
                "policy.json",
            ],
            {"policy.json": '{"allow": [{"cidr": "10.0.0.0/8", "port": 0}]}'},
            (),
        ),
        (
            "policy file not JSON",
            [
                "--network-egress",
                "allow",
                "--network-egress-policy-file",
                "policy.json",
            ],
            {"policy.json": "{"},
            (),
        ),
        (
            "policy errors before missing artifacts",
            ["--network-egress-policy-file", _r("policy.json")],
            POLICY,
            (CLI_OPENVMM, "build/vmlinux"),
        ),
        ("missing OpenVMM", [], None, (CLI_OPENVMM,)),
        ("missing kernel", [], None, ("build/vmlinux",)),
        ("missing Alpine initramfs", [], None, ("build/initramfs.cpio.gz",)),
        ("missing Ubuntu initramfs", ["--guest", "ubuntu"], None, ()),
        ("missing Azure Linux initramfs", ["--guest", "azurelinux"], None, ()),
        (
            "missing artifacts before invalid shares",
            ["--mount", "/data"],
            None,
            (CLI_OPENVMM,),
        ),
        ("mount without host path", ["--mount", "/data"], None, ()),
        ("mount with empty host path", ["--mount", "/data,"], None, ()),
        ("mount with empty host path and mode", ["--mount", "/data,,rw"], None, ()),
        (
            "mount with invalid mode",
            ["--mount", f"/data,{_r('share')},rx"],
            None,
            (),
        ),
        (
            "mount with commas but no mode",
            ["--mount", f"/data,{_r('alpha,beta')}"],
            COMMAS,
            (),
        ),
        (
            "invalid second mount",
            ["--mount", f"/a,{_r('share')}", "--mount", "relative"],
            None,
            (),
        ),
        (
            "invalid mount before invalid targets",
            ["--mount", f"a,{_r('share')}", "--mount", "/b"],
            None,
            (),
        ),
        ("owner without mount", ["--mount-owner", "vmm"], None, ()),
        (
            "relative deny path with several shares",
            [*several, "--mount-deny", "secrets"],
            None,
            (),
        ),
        (
            "relative allow path with a shared file",
            [*single_file, "--mount-allow", "secrets"],
            None,
            (),
        ),
        (
            "relative write path with several shares",
            [*several, "--mount-write", "./out/"],
            None,
            (),
        ),
        (
            "relative paths in kind order",
            [*several, "--mount-write", "out", "--mount-deny", "secrets"],
            None,
            (),
        ),
        (
            "relative target with several shares",
            ["--mount", f"a,{_r('share')}", "--mount", f"/b,{_r('share2')}"],
            None,
            (),
        ),
        (
            "relative target of a shared file",
            ["--mount", f"single.txt,{_r('single.txt')}"],
            None,
            (),
        ),
        (
            "root target with several shares",
            ["--mount", f"/,{_r('share')}", "--mount", f"/b,{_r('share2')}"],
            None,
            (),
        ),
        (
            "target with a space",
            ["--mount", f"/a b,{_r('share')}", "--mount", f"/b,{_r('share2')}"],
            None,
            (),
        ),
        (
            "target with an equals sign",
            ["--mount", f"/a=b,{_r('share')}", "--mount", f"/b,{_r('share2')}"],
            None,
            (),
        ),
        (
            "target with a backslash",
            ["--mount", f"/a\\b,{_r('share')}", "--mount", f"/b,{_r('share2')}"],
            None,
            (),
        ),
        (
            "target with a dot component",
            ["--mount", f"/a/./b,{_r('share')}", "--mount", f"/b,{_r('share2')}"],
            None,
            (),
        ),
        (
            "target with a parent component",
            ["--mount", f"/a/../b,{_r('share')}", "--mount", f"/c,{_r('share2')}"],
            None,
            (),
        ),
        (
            "target with an empty component",
            ["--mount", f"/a//b,{_r('share')}", "--mount", f"/c,{_r('share2')}"],
            None,
            (),
        ),
        (
            "equal targets",
            ["--mount", f"/a,{_r('share')}", "--mount", f"/a,{_r('share2')}"],
            None,
            (),
        ),
        (
            "nested targets",
            [
                "--mount",
                f"/workspace,{_r('share')}",
                "--mount",
                f"/workspace/cache,{_r('share2')}",
            ],
            None,
            (),
        ),
        (
            "target containing another",
            [
                "--mount",
                f"/workspace/cache,{_r('share')}",
                "--mount",
                f"/workspace,{_r('share2')}",
            ],
            None,
            (),
        ),
        (
            "target containing the aggregate",
            ["--mount", f"/run/nvx,{_r('share')}", "--mount", f"/b,{_r('share2')}"],
            None,
            (),
        ),
        (
            "target in the aggregate",
            [
                "--mount",
                f"/run/nvx/shares/x,{_r('share')}",
                "--mount",
                f"/b,{_r('share2')}",
            ],
            None,
            (),
        ),
        (
            "too many shares for the budget",
            [
                arguments
                for index in range(20)
                for arguments in ("--mount", f"/share/{index:02},{_r('share')}")
            ],
            None,
            (),
        ),
    ]
    scenarios: list[Scenario | None] = [
        _scenario(
            f"validation: {name}",
            _run(*argv, "--dry-run", write=write, remove=remove),
        )
        for name, argv, write, remove in cases
    ]
    for name, argv in _budget_cases()[1:]:
        scenarios.append(
            _scenario(f"validation: budget {name}", _run(*argv, "--dry-run"))
        )
    scenarios.append(
        _scenario(
            "validation: missing OpenVMM override",
            _run("--dry-run", env={"NVX_OPENVMM": _r("alt/openvmm{exe}")}),
        )
    )
    scenarios.append(
        _scenario(
            "validation: caller owner on Windows",
            _run(
                "--mount",
                f"/data,{_r('share')}",
                "--mount-owner",
                "caller",
                "--dry-run",
            ),
            platforms=("windows",),
        )
    )
    return scenarios


def _dry_run_scenarios() -> list[Scenario | None]:
    """Return OpenVMM's command lines, which --dry-run prints without running."""
    several = [
        "--mount",
        f"/data,{_r('share')},rw",
        "--mount",
        f"/other,{_r('share2')}",
    ]
    cases: list[tuple[str, list[str], dict[str, str] | None, Sequence[str]]] = [
        ("Alpine default", [], None, ()),
        ("Ubuntu default", ["--guest", "ubuntu"], UBUNTU, ()),
        ("Azure Linux default", ["--guest", "azurelinux"], AZURELINUX, ()),
        ("Alpine with memory", ["--guest", "alpine", "--memory-mib", "256"], None, ()),
        (
            "Ubuntu with memory and capacity",
            [
                "--guest",
                "ubuntu",
                "--memory-mib",
                "1024",
                "--memory-capacity-mib",
                "2048",
            ],
            UBUNTU,
            (),
        ),
        (
            "Azure Linux with the default capacity",
            ["--guest", "azurelinux", "--memory-capacity-mib", "512"],
            AZURELINUX,
            (),
        ),
        (
            "last guest",
            ["--guest", "ubuntu", "--guest", "azurelinux"],
            AZURELINUX,
            (),
        ),
        (
            "every boot option",
            [
                "--hypervisor",
                "mshv",
                "--machine",
                "microvm",
                "--memory-mib",
                "1024",
                "--memory-capacity-mib",
                "2048",
                "--processors",
                "8",
                "--cpu-profile",
                "intel.alderlake.v1",
                "--cmdline",
                "quiet loglevel=3",
                "--outcome-report",
                _r("reports/openvmm-run.json"),
            ],
            None,
            (),
        ),
        ("auto hypervisor", ["--hypervisor", "auto"], None, ()),
        ("KVM", ["--hypervisor", "kvm"], None, ()),
        ("WHP", ["--hypervisor", "whp"], None, ()),
        (
            "canonical integers",
            [
                "--memory-mib",
                " +12_8 ",
                "--memory-capacity-mib",
                "0256",
                "--processors",
                " 2 ",
            ],
            None,
            (),
        ),
        (
            "other digits",
            ["--processors", "\u0664", "--memory-mib", "\u0665\u0661\u0662"],
            None,
            (),
        ),
        ("negative memory", ["--memory-mib", "-5"], None, ()),
        ("attached negative memory", ["--memory-mib=-128"], None, ()),
        (
            "equal memory and capacity",
            ["--memory-mib", "512", "--memory-capacity-mib", "512"],
            None,
            (),
        ),
        ("last processors", ["--processors", "2", "--processors", "4"], None, ()),
        ("cpu profile auto", ["--cpu-profile", "auto"], None, ()),
        ("cpu profile host", ["--cpu-profile", "host"], None, ()),
        ("restore", ["--restore-snapshot", _r("snapshot")], None, ()),
        (
            "restore without guest artifacts",
            ["--restore-snapshot", _r("snapshot")],
            None,
            ("build/vmlinux", "build/initramfs.cpio.gz"),
        ),
        (
            "restore with every option",
            [
                "--hypervisor",
                "mshv",
                "--processors",
                "4",
                "--restore-snapshot",
                _r("snapshot"),
                "--restore-processors",
                "2",
                "--restore-memory-mib",
                "1024",
                "--restore-ready-path",
                _r("ready.sock"),
                "--cpu-profile",
                "host",
                "--outcome-report",
                _r("reports/restore.json"),
                "--cmdline",
                "quiet",
            ],
            None,
            (),
        ),
        (
            "restore of the Alpine guest",
            ["--guest", "alpine", "--restore-snapshot", _r("snapshot")],
            None,
            (),
        ),
        (
            "restore with equal processors",
            [
                "--processors",
                "2",
                "--restore-processors",
                "2",
                "--restore-snapshot",
                _r("snapshot"),
            ],
            None,
            (),
        ),
        (
            "restore ignores the memory",
            ["--memory-mib", "512", "--restore-snapshot", _r("snapshot")],
            None,
            (),
        ),
        (
            "restore paths",
            [
                "--restore-snapshot",
                "./snapshot/",
                "--restore-ready-path",
                "run//ready.sock",
            ],
            None,
            (),
        ),
        ("one share", ["--mount", f"/data,{_r('share')}"], None, ()),
        ("one writable share", ["--mount", f"/data,{_r('share')},rw"], None, ()),
        ("one relative share", ["--mount", "/data,share"], None, ()),
        (
            "one share with relative policy paths",
            [
                "--mount",
                f"/data,{_r('share')},rw",
                "--mount-write",
                "writable",
                "--mount-allow",
                "hidden/shown",
                "--mount-deny",
                "./hidden/",
            ],
            None,
            (),
        ),
        (
            "policy paths without a share",
            ["--mount-deny", _r("share/hidden"), "--mount-allow", "x"],
            None,
            (),
        ),
        (
            "several shares with policy paths",
            [
                *several,
                "--mount-write",
                _r("share/writable"),
                "--mount-allow",
                _r("share/hidden/shown"),
                "--mount-deny",
                _r("share/hidden"),
                "--mount-deny",
                _r("share2") + os.sep + "." + os.sep + "data.txt",
            ],
            None,
            (),
        ),
        (
            "several shares with a command line",
            [*several, "--cmdline", "quiet"],
            None,
            (),
        ),
        (
            "one shared file",
            ["--mount", f"/etc/single.txt,{_r('single.txt')},ro"],
            None,
            (),
        ),
        (
            "one relative shared file",
            ["--mount", "/etc/single.txt,single.txt"],
            None,
            (),
        ),
        (
            "a directory and a file",
            [
                "--mount",
                f"/work,{_r('share')},rw",
                "--mount",
                f"/config/single.txt,{_r('single.txt')}",
            ],
            None,
            (),
        ),
        (
            "host paths with commas",
            [
                "--mount",
                f"/work,{_r('alpha,beta')},rw",
                "--mount",
                f"/reports,{_r('2026, final')},ro",
                "--mount",
                f"/config/settings.json,{_r('settings,v2.json')},ro",
                "--mount-deny",
                _r("alpha,beta/data.txt"),
            ],
            COMMAS,
            (),
        ),
        (
            "one directory with commas",
            ["--mount", f"/work,{_r('alpha,beta')},rw"],
            COMMAS,
            (),
        ),
        (
            "host path ending in a mode",
            ["--mount", f"/data,{_r('share')},rw,ro"],
            None,
            (),
        ),
        (
            "host path ending in a comma",
            ["--mount", f"/data,{_r('share')},,rw"],
            None,
            (),
        ),
        (
            "vmm owner",
            ["--mount", f"/data,{_r('share')}", "--mount-owner", "vmm"],
            None,
            (),
        ),
        (
            "vmm owner of several shares",
            [*several, "--mount-owner", "vmm"],
            None,
            (),
        ),
        (
            "restore with shares",
            [
                "--restore-snapshot",
                _r("snapshot"),
                *several,
                "--mount",
                f"/etc/single.txt,{_r('single.txt')}",
            ],
            None,
            (),
        ),
        (
            "restore with shares and a command line",
            ["--restore-snapshot", _r("snapshot"), *several, "--cmdline", "quiet"],
            None,
            (),
        ),
        (
            "network",
            [
                "--net",
                "10.0.0.2/24",
                "--network-profile",
                "portable",
                "--network-egress",
                "deny",
                "--network-ingress",
                "allow",
                "--network-egress-allow",
                "10.0.0.0/8:tcp:443",
                "--network-egress-allow",
                "2001:db8::/32",
                "--network-egress-deny",
                "10.1.0.0/16",
                "--host-loopback",
                "allow",
                "--network-proxy",
                "10.0.0.1:3128",
                "--host-loopback-forward",
                "tcp:8080:80",
                "--host-loopback-forward",
                "udp:5353:53",
            ],
            None,
            (),
        ),
        ("egress default", ["--network-egress", "deny"], None, ()),
        ("host loopback forward", ["--host-loopback-forward", "tcp:8080:80"], None, ()),
        (
            "egress policy file",
            [
                "--network-egress",
                "deny",
                "--network-egress-policy-file",
                _r("policy.json"),
            ],
            POLICY,
            (),
        ),
        (
            "empty egress policy file",
            [
                "--network-egress",
                "allow",
                "--network-egress-policy-file",
                _r("./policy.json"),
            ],
            {"policy.json": "{}"},
            (),
        ),
        (
            "outcome report path",
            ["--outcome-report", "reports//./outcome.json"],
            None,
            (),
        ),
        ("quoted command line", ["--cmdline", 'a="b c" d'], None, ()),
        ("command line starting with a dash", ["--cmdline=-x"], None, ()),
        ("command line of help", ["--cmdline=--help"], None, ()),
        ("empty command line", ["--cmdline", ""], None, ()),
        ("blank command line", ["--cmdline", " "], None, ()),
        ("last command line", ["--cmdline", "a", "--cmdline", "b"], None, ()),
        (
            "long command line of one directory share",
            ["--mount", f"/data,{_r('share')}", "--cmdline", "x" * 1100],
            None,
            (),
        ),
        (
            "restore without the budget",
            [
                "--restore-snapshot",
                _r("snapshot"),
                *[
                    arguments
                    for index in range(20)
                    for arguments in ("--mount", f"/share/{index:02},{_r('share')}")
                ],
            ],
            None,
            (),
        ),
    ]
    scenarios = [
        _scenario(
            f"dry run: {name}",
            _run(*argv, "--dry-run", write=write, remove=remove),
        )
        for name, argv, write, remove in cases
    ]
    scenarios.append(
        _scenario("dry run: budget fits", _run(*_budget_cases()[0][1], "--dry-run"))
    )
    scenarios.append(
        _scenario(
            "dry run: caller owner",
            _run(
                "--mount",
                f"/data,{_r('share')}",
                "--mount-owner",
                "caller",
                "--dry-run",
            ),
            platforms=("posix",),
        )
    )
    return scenarios


def _launch_scenarios() -> list[Scenario | None]:
    """Return runs of the fake OpenVMM, and those that fail before it runs."""
    report = _r("reports/openvmm-run.json")
    return [
        _scenario("launch: boot", _run()),
        _scenario("launch: exit status", _run("--cmdline", "fake_exit_status=7")),
        _scenario("launch: failed boot", _run("--cmdline", "quiet fake_exit_status=1")),
        _scenario(
            "launch: Ubuntu",
            _run("--guest", "ubuntu", "--memory-capacity-mib", "1024", write=UBUNTU),
        ),
        _scenario(
            "launch: every boot option",
            _run(
                "--hypervisor",
                "mshv",
                "--processors",
                "8",
                "--cpu-profile",
                "host",
                "--memory-mib",
                "256",
                "--memory-capacity-mib",
                "512",
            ),
        ),
        _scenario(
            "launch: restore",
            _run(
                "--processors",
                "4",
                "--restore-snapshot",
                _r("snapshot"),
                "--restore-processors",
                "2",
                "--restore-memory-mib",
                "1024",
                "--restore-ready-path",
                _r("ready.sock"),
                "--cpu-profile",
                "auto",
                write=SNAPSHOT,
            ),
        ),
        _scenario(
            "launch: restore without guest artifacts",
            _run(
                "--restore-snapshot",
                _r("snapshot"),
                write=SNAPSHOT,
                remove=("build/vmlinux", "build/initramfs.cpio.gz"),
            ),
        ),
        _scenario(
            "launch: missing snapshot",
            _run("--restore-snapshot", _r("missing"), "--cmdline", "quiet"),
        ),
        _scenario(
            "launch: outcome report",
            _run(
                "--outcome-report",
                report,
                check=("reports/openvmm-run.json", "build/fake-openvmm-reports.json"),
            ),
        ),
        _scenario(
            "launch: restore outcome report",
            _run(
                "--restore-snapshot",
                _r("snapshot"),
                "--outcome-report",
                report,
                "--cmdline",
                "fake_exit_status=4",
                write=SNAPSHOT,
                check=("reports/openvmm-run.json", "fake-openvmm-reports.json"),
            ),
        ),
        _scenario(
            "launch: network outcome report",
            _run(
                "--net",
                "10.0.0.2/24",
                "--network-profile",
                "portable",
                "--network-egress",
                "deny",
                "--network-egress-allow",
                "10.0.0.0/8:tcp:443",
                "--host-loopback",
                "deny",
                "--outcome-report",
                report,
                check=("reports/openvmm-run.json",),
            ),
        ),
        _scenario(
            "launch: one share",
            _run(
                "--mount",
                f"/data,{_r('share')},rw",
                "--mount-deny",
                "hidden",
                "--mount-allow",
                "hidden/shown",
                "--mount-write",
                "writable",
                check=("build/fake-openvmm-.json",),
            ),
        ),
        _scenario(
            "launch: several shares and a file",
            _run(
                "--mount",
                f"/data,{_r('share')}",
                "--mount",
                f"/single.txt,{_r('single.txt')}",
                "--mount-deny",
                _r("share/hidden"),
                "--cmdline",
                "fake_exit_status=3",
            ),
        ),
        _scenario(
            "launch: OpenVMM refuses its options",
            _run("--host-loopback-forward", "tcp:8080:80"),
        ),
        _scenario(
            "launch: relative OpenVMM override",
            _run(env={"NVX_OPENVMM": CLI_OPENVMM.replace("/", os.sep)}),
        ),
        _scenario("launch: missing OpenVMM", _run(remove=(CLI_OPENVMM,))),
        _scenario("launch: missing kernel", _run(remove=("build/vmlinux",))),
        _scenario(
            "launch: missing initramfs", _run(remove=("build/initramfs.cpio.gz",))
        ),
        _scenario("launch: missing Ubuntu initramfs", _run("--guest", "ubuntu")),
    ]


def run_scenarios() -> list[Scenario]:
    """Return the run scenarios of this platform."""
    scenarios = [
        *_usage_scenarios(),
        *_validation_scenarios(),
        *_dry_run_scenarios(),
        *_launch_scenarios(),
    ]
    return [scenario for scenario in scenarios if scenario is not None]


def _expand(text: str, root: Path) -> str:
    return text.replace("{root}", str(root)).replace("{exe}", EXE)


def _below(root: Path, relative: str) -> Path:
    return root.joinpath(*_expand(relative, root).split("/"))


def _make_tree(root: Path, tree: list[Entry], fake: Path) -> None:
    for entry in tree:
        path = _below(root, cast(str, entry["path"]))
        if entry.get("openvmm"):
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fake, path)
        elif "text" in entry:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(cast(str, entry["text"]).encode())
        else:
            path.mkdir(parents=True, exist_ok=True)


def _write_file(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cast(str, content).encode())


def _child_environment() -> dict[str, str]:
    """Return the environment of a recorded command: no terminal size, color,
    or OpenVMM override, and UTF-8 standard streams."""
    variables = ("COLUMNS", "LINES", "FORCE_COLOR", "PYTHON_COLORS", "NVX_OPENVMM")
    environment = {
        name: value for name, value in os.environ.items() if name not in variables
    }
    environment["NO_COLOR"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _lines(text: str) -> str:
    """Return `text` as `print` writes it to a standard stream."""
    return text.replace("\n", os.linesep) + os.linesep


def _nvx_rs_prog(stdout: str, stderr: str) -> tuple[str, str]:
    """Return the output of nvx.py with the program names of nvx-rs.

    The run parser's name, `nvx.py run`, has the length of `nvx-rs run`, so its
    usage wraps the same way. Errors that the top-level parser reports show its
    usage, which nvx-rs replaces with its own."""
    stdout = stdout.replace("usage: nvx.py run", "usage: nvx-rs run", 1)
    stderr = stderr.replace("usage: nvx.py run", "usage: nvx-rs run", 1)
    stderr = stderr.replace("nvx.py run: error: ", "nvx-rs run: error: ")
    lines = stderr.splitlines(keepends=True)
    if (
        lines
        and lines[-1].startswith("nvx.py: error: ")
        and lines[0].startswith("usage: nvx.py ")
    ):
        last = lines[-1].replace("nvx.py: error: ", "nvx-rs: error: ", 1)
        stderr = _lines(NVX_RS_USAGE) + last
    return stdout, stderr


def _normalize(text: str, root: Path) -> str:
    """Replace the root, as printed and as JSON writes it, and volatile values,
    such as process IDs and random identifiers, with placeholders."""
    raw = str(root)
    escaped = json.dumps(raw)[1:-1]
    if escaped != raw:
        text = text.replace(escaped, "{root_json}")
    text = text.replace(raw, "{root}")
    for pattern, replacement in VOLATILE:
        text = pattern.sub(replacement, text)
    return text


def _decode(data: bytes) -> str:
    return data.decode("utf-8", "backslashreplace")


def _snapshot(root: Path, relative: str) -> object:
    """Return what a step shows of a file or directory below the root."""
    path = _below(root, relative)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    entry: Entry = {}
    if stat.S_ISDIR(info.st_mode):
        entry["entries"] = sorted(os.listdir(path))
    else:
        entry["text"] = _normalize(_decode(path.read_bytes()), root)
    if os.name != "nt" and (
        path.name in CLI_MODE_NAMES or path.name.startswith("exec")
    ):
        entry["mode"] = stat.S_IMODE(info.st_mode)
    return entry


def _step_environment(step: Step, root: Path) -> dict[str, str]:
    return {
        name: _expand(value, root)
        for name, value in cast(dict[str, str], step.get("env", {})).items()
    }


def _python_command(
    nvx_rs: Path, argv: list[str], root: Path, step: Step
) -> tuple[int, str, str]:
    """Return what `nvx.py ARGV` gives, with the nvx-rs at `nvx_rs`."""
    child = _child_environment()
    child.update(_step_environment(step, root))
    child.update(
        NVX_FIXTURE_SCRIPTS=str(REPO_ROOT / "scripts"),
        NVX_FIXTURE_ROOT=str(root),
        NVX_RS=str(nvx_rs),
    )
    result = subprocess.run(
        [sys.executable, "-u", "-c", PYTHON_RUN, *argv],
        cwd=root,
        env=child,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=CLI_STEP_TIMEOUT,
        check=False,
    )
    stdout, stderr = _nvx_rs_prog(_decode(result.stdout), _decode(result.stderr))
    return result.returncode, stdout, stderr


def _rust_command(
    nvx_rs: Path, argv: list[str], root: Path, step: Step
) -> tuple[int, str, str]:
    """Return what `nvx-rs --root ROOT ARGV` gives."""
    child = _child_environment()
    child.update(_step_environment(step, root))
    result = subprocess.run(
        [str(nvx_rs), "--root", str(root), *argv],
        cwd=root,
        env=child,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=CLI_STEP_TIMEOUT,
        check=False,
    )
    return result.returncode, _decode(result.stdout), _decode(result.stderr)


def play(
    scenario: Scenario, tree: list[Entry], binaries: tuple[Path, Path], python: bool
) -> list[Entry]:
    """Play a scenario in a fresh root with nvx.py, or else with nvx-rs, and
    return what each step gave."""
    nvx_rs, fake = binaries
    with tempfile.TemporaryDirectory(
        prefix="nvx-cli-", ignore_cleanup_errors=True
    ) as temporary:
        root = Path(temporary).resolve()
        _make_tree(root, tree, fake)
        results: list[Entry] = []
        for step in cast(list[Step], scenario["steps"]):
            for relative in cast(list[str], step.get("remove", [])):
                _below(root, relative).unlink()
            for relative, content in cast(
                dict[str, object], step.get("write", {})
            ).items():
                _write_file(_below(root, relative), content)
            argv = [_expand(value, root) for value in cast(list[str], step["argv"])]
            command = _python_command if python else _rust_command
            status, stdout, stderr = command(nvx_rs, argv, root, step)
            result: Entry = {
                "status": status,
                "stdout": _normalize(stdout, root),
                "stderr": _normalize(stderr, root),
            }
            checks = cast(list[str], step.get("check", []))
            if checks:
                result["files"] = {
                    relative: _snapshot(root, relative) for relative in checks
                }
            results.append(result)
        return results


def cli_binaries() -> tuple[Path, Path]:
    """Build nvx-rs and nvx-fake-openvmm, and return their paths."""
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--quiet",
            "--features",
            "cli,testing",
            "--bin",
            "nvx-rs",
            "--bin",
            "nvx-fake-openvmm",
        ],
        cwd=NVX_RS_DIR,
        env=dict(os.environ, NVX_BUNDLE="skip"),
        check=True,
    )
    target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    directory = (target if target.is_absolute() else NVX_RS_DIR / target) / "debug"
    return directory / f"nvx-rs{EXE}", directory / f"nvx-fake-openvmm{EXE}"


def read_fixture() -> dict[str, object]:
    return cast(dict[str, object], json.loads(CLI_FIXTURE.read_text("utf-8")))


def record(fixture: dict[str, object], binaries: tuple[Path, Path]) -> list[Entry]:
    """Record the run scenarios of this platform with nvx.py."""
    if sys.version_info < (3, 14, 5):
        raise SystemExit(
            "recording the run scenarios needs CPython 3.14.5 or later, whose "
            "argparse quotes the choices of an invalid choice, as nvx-rs does"
        )
    tree = cast(list[Entry], fixture["tree"])
    section: list[Entry] = []
    for scenario in run_scenarios():
        results = play(scenario, tree, binaries, python=True)
        steps = [
            {**step, **result}
            for step, result in zip(
                cast(list[Step], scenario["steps"]), results, strict=True
            )
        ]
        section.append({**scenario, "steps": steps})
    return section


def merged(fixture: dict[str, object], section: list[Entry]) -> dict[str, object]:
    """Return the fixture with the run scenarios of this platform's section
    replaced by `section`."""
    current = cast(list[Entry], fixture[CLI_PLATFORM])
    kept = [
        scenario
        for scenario in current
        if not cast(str, scenario["name"]).startswith(PREFIX)
    ]
    return {**fixture, "generator": GENERATOR, CLI_PLATFORM: [*kept, *section]}


def fixture_text(document: dict[str, object]) -> str:
    """Return the fixture with one case per line, in ASCII."""
    lines = ["{"]
    items = list(document.items())
    for position, (key, value) in enumerate(items):
        comma = "," if position < len(items) - 1 else ""
        if isinstance(value, list) and value and isinstance(value[0], dict):
            cases = cast(list[object], value)
            lines.append(f"{json.dumps(key)}: [")
            for index, case in enumerate(cases):
                separator = "," if index < len(cases) - 1 else ""
                lines.append(json.dumps(case, separators=(",", ":")) + separator)
            lines.append(f"]{comma}")
        else:
            lines.append(f"{json.dumps(key)}: {json.dumps(value)}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _differences(name: str, expected: list[Entry], actual: list[Entry]) -> list[str]:
    problems: list[str] = []
    for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
        for field in ("status", "stdout", "stderr", "files"):
            if want.get(field) != got.get(field):
                diff = "\n".join(
                    difflib.unified_diff(
                        json.dumps(want.get(field), indent=1).splitlines(),
                        json.dumps(got.get(field), indent=1).splitlines(),
                        "nvx.py",
                        "nvx-rs",
                        lineterm="",
                    )
                )
                problems.append(f"{name}, step {index + 1}, {field}:\n{diff}")
    return problems


def check_parity(fixture: dict[str, object], binaries: tuple[Path, Path]) -> list[str]:
    """Play the recorded run scenarios of this platform with nvx-rs, and return
    how any step differs from the recording."""
    tree = cast(list[Entry], fixture["tree"])
    problems: list[str] = []
    for scenario in cast(list[Scenario], fixture[CLI_PLATFORM]):
        name = cast(str, scenario["name"])
        if not name.startswith(PREFIX):
            continue
        expected = cast(list[Entry], scenario["steps"])
        problems += _differences(
            name, expected, play(scenario, tree, binaries, python=False)
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record the golden scenarios of nvx.py run for nvx-rs."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the recording would change",
    )
    mode.add_argument(
        "--parity",
        action="store_true",
        help="replay the recorded run scenarios of this host with nvx-rs",
    )
    args = parser.parse_args(argv)
    binaries = cli_binaries()
    fixture = read_fixture()
    if args.parity:
        problems = check_parity(fixture, binaries)
        for problem in problems:
            print(f"nvx-rs differs from nvx.py: {problem}", file=sys.stderr)
        count = sum(
            1
            for scenario in cast(list[Scenario], fixture[CLI_PLATFORM])
            if cast(str, scenario["name"]).startswith(PREFIX)
        )
        print(f"played {count} {CLI_PLATFORM} run scenarios with nvx-rs")
        return 1 if problems else 0
    text = fixture_text(merged(fixture, record(fixture, binaries)))
    if CLI_FIXTURE.read_text("utf-8") == text:
        return 0
    if args.check:
        print(f"{CLI_FIXTURE} is out of date; record it again", file=sys.stderr)
        return 1
    CLI_FIXTURE.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {CLI_FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
