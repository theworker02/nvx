#!/usr/bin/env python3
"""Build, run, benchmark, and package the OpenVMM/NVX distribution."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from nvx_tools import nvx_rs
from nvx_tools.aci_edge_sandboxes_tests import (
    configure_parser as configure_aci_edge_sandboxes_test_parser,
)
from nvx_tools.adversarial import configure_parser as configure_adversarial_parser
from nvx_tools.agent_bundling_tests import (
    configure_parser as configure_agent_bundling_test_parser,
)
from nvx_tools.benchmark import configure_parser as configure_benchmark_parser
from nvx_tools.benchmark import run as run_benchmark
from nvx_tools.build import (
    build_all,
    build_distro_layer,
    build_docker_initramfs,
    build_guest,
    build_initramfs,
    build_kernel,
    build_openvmm,
    materialize_kernel_provenance_inputs,
    record_openvmm_provenance,
    verify_guest_determinism,
)
from nvx_tools.build_config import (
    BuildConfig,
    DistroLayerBuildConfig,
    DockerBuildConfig,
    KernelBuildConfig,
    OpenVmmBuildConfig,
)
from nvx_tools.build_constants import (
    BuildConstants,
    InitramfsBuildConstants,
    KernelBuildConstants,
    UbuntuBuildConstants,
)
from nvx_tools.ci import (
    OPENVMM_TEST_BACKENDS,
    REQUIRED_CI_RESULT_ENVIRONMENTS,
    CiChanges,
    classify_ci_changes,
    describe_ci_changes,
    kernel_patch_identity,
    required_ci_failures,
    run_openvmm_tests,
    run_openvmm_unit_tests,
    setup_cross_os_cache,
)
from nvx_tools.collect_alpine_sources import (
    configure_parser as configure_alpine_sources_parser,
)
from nvx_tools.collect_ubuntu_sources import (
    configure_parser as configure_ubuntu_sources_parser,
)
from nvx_tools.common import (
    ScriptError,
    artifact_path,
)
from nvx_tools.create_linux_source_archive import (
    configure_parser as configure_linux_source_archive_parser,
)
from nvx_tools.doctor import configure_parser as configure_doctor_parser
from nvx_tools.doctor import host_cpu_signature
from nvx_tools.elf import check_static_executables
from nvx_tools.guest_binaries import (
    build_guest_binaries,
    check_guest_binaries,
    guest_binary_source_sha256,
    test_guest_binaries,
)
from nvx_tools.guests import GUEST_NAMES, guest_descriptor
from nvx_tools.microvm_tests import configure_parser as configure_microvm_test_parser
from nvx_tools.performance import configure_parser as configure_performance_parser
from nvx_tools.release import (
    collect_release_sources,
    create_release_archive,
    download_checkout_release,
    package_release,
    verify_source_tree,
)
from nvx_tools.time_abi import (
    benchmark_cpu_profile_refusal,
    cpu_profile_guidance_applies,
    host_cpu_profile_guidance,
    openvmm_cpu_generations,
)

DEFAULT_RELEASE_REPOSITORY = "microsoft/nvx"
HYPERVISORS = ("auto", "whp", "kvm", "mshv")
# `vmm` performs every share operation as OpenVMM; `caller` performs each as the
# guest caller's identity, with guest root squashed to the share owner.
MOUNT_OWNERS = ("vmm", "caller")
NETWORK_PROFILES = ("portable",)
NETWORK_EGRESS_RULE_METAVAR = "CIDR[:PROTOCOL[:PORT[-PORT]]]"
NETWORK_EGRESS_RULE_HELP = (
    "{action} guest egress to an IPv4 or IPv6 CIDR, optionally only for PROTOCOL "
    "tcp, udp, or icmp, and for tcp or udp only on PORT or on the inclusive range "
    "PORT-PORT; repeat to add rules"
)
SANDBOX_OPERATIONS = ("run", "provision", "start", "exec", "stop", "deprovision")
# The namespace attribute in which the run and sandbox parsers record their
# options for nvx-rs.
NVX_RS_ARGUMENTS = "nvx_rs_arguments"


class _NvxRsOption(argparse.Action):
    """Record an option of `nvx.py run` or `nvx.py sandbox` for nvx-rs, which
    checks and uses every value.

    argparse resolves abbreviations and the --option=VALUE form, so nvx-rs
    receives each option under its canonical name, in command-line order,
    which attributes each --mount-deny, --mount-allow, and --mount-write of
    `sandbox` to the --mount before it, and with its value attached, so that a
    value that starts with "-" stays a value. With `keep`, the value, or `True`
    for a flag, also goes to the option's destination, where nvx.py reads it.
    """

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: int | str | None = None,
        *,
        keep: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            option_strings, dest, nargs=nargs, default=argparse.SUPPRESS, **kwargs
        )
        self.keep = keep

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[object] | None,
        option_string: str | None = None,
    ) -> None:
        name = self.option_strings[0]
        argument = name if self.nargs == 0 else f"{name}={cast(str, values)}"
        recorded = cast(list[str], getattr(namespace, NVX_RS_ARGUMENTS, None) or [])
        setattr(namespace, NVX_RS_ARGUMENTS, [*recorded, argument])
        if self.keep:
            setattr(namespace, self.dest, True if self.nargs == 0 else values)


def _run(
    args: list[str | os.PathLike[str]], *, cwd: Path = BuildConstants.REPO_ROOT
) -> None:
    command = [os.fspath(arg) for arg in args]
    print(f">> {shlex.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def command_init(_: argparse.Namespace) -> None:
    _run(["git", "submodule", "update", "--init", "--recursive"])


def _openvmm_build_config(args: argparse.Namespace) -> OpenVmmBuildConfig:
    return OpenVmmBuildConfig(
        skip_restore=getattr(args, "skip_restore", False),
        backend=getattr(args, "backend", None),
    )


def _build_config(args: argparse.Namespace) -> BuildConfig:
    return BuildConfig(
        guest=getattr(args, "guest", InitramfsBuildConstants.DEFAULT_GUEST),
        native_guest=getattr(args, "native", False),
        debug_kernel=getattr(args, "debug_kernel", False),
        openvmm=_openvmm_build_config(args),
    )


def command_build_guest(args: argparse.Namespace) -> None:
    build_guest(_build_config(args))


def command_check_guest_binaries(args: argparse.Namespace) -> None:
    check_guest_binaries(args.toolchain)


def command_build_guest_binaries(args: argparse.Namespace) -> None:
    result = build_guest_binaries(args.output)
    print(
        json.dumps(
            {
                "source_sha256": result.source_sha256,
                "binaries": result.binaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_test_guest_binaries(args: argparse.Namespace) -> None:
    test_guest_binaries(args.binaries)


def command_build_kernel(args: argparse.Namespace) -> None:
    build_kernel(
        KernelBuildConfig.debug_variant()
        if getattr(args, "debug", False)
        else KernelBuildConfig()
    )


def command_build_initramfs(args: argparse.Namespace) -> None:
    if not guest_descriptor(args.guest).native_build_supported:
        build_docker_initramfs(DockerBuildConfig(), args.guest)
        return
    build_initramfs(BuildConfig.initramfs_config(args.guest))


def command_build_distro_layer(args: argparse.Namespace) -> None:
    build_distro_layer(
        DistroLayerBuildConfig(
            guest=args.guest,
            work=(
                BuildConstants.BUILD_DIR
                / InitramfsBuildConstants.DISTRO_WORK_DIRECTORY_TEMPLATE.format(
                    guest=args.guest
                )
            ),
            output=args.output,
            replace=args.replace,
        )
    )


def command_verify_guest_determinism(args: argparse.Namespace) -> None:
    verify_guest_determinism(args.work_dir, args.guest)


def command_build_openvmm(args: argparse.Namespace) -> None:
    build_openvmm(_openvmm_build_config(args))


def command_record_openvmm_provenance(_: argparse.Namespace) -> None:
    record_openvmm_provenance(OpenVmmBuildConfig())


def command_record_nvx_rs_provenance(args: argparse.Namespace) -> None:
    path = nvx_rs.record_provenance(args.target)
    print(f">> recorded the provenance of nvx-rs in {path}")


def command_materialize_kernel_provenance_inputs(_: argparse.Namespace) -> None:
    materialize_kernel_provenance_inputs()


def command_setup_cross_os_cache(_: argparse.Namespace) -> None:
    setup_cross_os_cache()


def command_classify_ci_changes(args: argparse.Namespace) -> None:
    contents = args.changed_files.read_bytes().decode("utf-8")
    changed_files = contents.split("\0") if args.null else contents.splitlines()
    changes = classify_ci_changes(args.event_name, changed_files)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8", newline="") as summary:
            summary.write(describe_ci_changes(args.event_name, changed_files, changes))
    for name, selected in changes.outputs().items():
        print(f"{name}={str(selected).lower()}")


def command_kernel_patch_identity(args: argparse.Namespace) -> None:
    print(kernel_patch_identity(args.root))


def command_guest_binary_source_identity(args: argparse.Namespace) -> None:
    print(guest_binary_source_sha256(args.root))


def command_check_required_ci(args: argparse.Namespace) -> None:
    results = {
        job: os.environ.get(environment, "")
        for job, environment in REQUIRED_CI_RESULT_ENVIRONMENTS.items()
    }
    failures = required_ci_failures(
        args.event_name,
        same_repository=args.same_repository == "true",
        changes=CiChanges(
            run_tests=args.run_tests == "true",
            run_workloads=args.run_workloads == "true",
            run_openvmm_unit_tests=args.run_openvmm_unit_tests == "true",
            run_openvmm_vmm_tests=args.run_openvmm_vmm_tests == "true",
        ),
        results=results,
    )
    if failures:
        for failure in failures:
            print(f"::error::{failure}")
        raise ScriptError(f"{len(failures)} required CI job result(s) did not match")


def command_check_static_elf(args: argparse.Namespace) -> None:
    check_static_executables(args.paths)


def command_test_openvmm(args: argparse.Namespace) -> None:
    run_openvmm_tests(args.backend)


def command_test_openvmm_unit(_: argparse.Namespace) -> None:
    run_openvmm_unit_tests()


def command_benchmark(args: argparse.Namespace) -> int:
    """Run the benchmark coordinator, after refusing a host whose CPU no
    built-in CPU profile of the selected OpenVMM checkout serves before it
    builds or measures anything. Only the phase 2 suite, which boots no
    microVM, runs on any host. A KVM worker, which the Windows coordinator
    reinvokes inside WSL without its --openvmm-dir, runs the binaries that the
    coordinator staged after this check, so it skips the check."""
    if args.suite != "phase2" and not args._kvm_worker:
        try:
            generations = openvmm_cpu_generations(args.openvmm_dir)
        except ValueError as error:
            raise ScriptError(
                "benchmark cannot confirm that a built-in CPU profile serves this "
                f"host's CPU: {error}"
            ) from error
        hypervisor = "whp" if args.backend == "both" else args.backend
        refusal = benchmark_cpu_profile_refusal(
            host_cpu_signature(), "openvmm", hypervisor, generations
        )
        if refusal is not None:
            raise ScriptError(refusal)
    return run_benchmark(args)


def command_build(args: argparse.Namespace) -> None:
    build_all(_build_config(args))


def _hypervisor(selected: str) -> str:
    if selected != "auto":
        return selected
    return "whp" if os.name == "nt" else "kvm"


def _release_platform(hypervisor: str) -> str:
    selected = _hypervisor(hypervisor)
    if sys.platform == "win32":
        host = "windows"
        supported = ("whp",)
    elif sys.platform.startswith("linux"):
        host = "linux"
        supported = ("kvm", "mshv")
    else:
        raise ScriptError(f"release downloads are unsupported on {sys.platform}")
    if selected not in supported:
        raise ScriptError(f"{selected} is not supported on {host}")
    return f"{host}-{selected}"


def command_download(args: argparse.Namespace) -> None:
    download_checkout_release(args.repository, _release_platform(args.hypervisor))


def _format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def _cold_boot_guidance(args: argparse.Namespace) -> list[str]:
    """Return the hidden --cold-boot-guidance option of nvx-rs run, which
    holds the guidance that nvx-rs prints after OpenVMM fails a cold boot on a
    host whose CPU no built-in CPU profile serves, or nothing.

    The CPU catalog of the guidance stays in time_abi.py, which doctor and
    benchmark use too. A dry run and a restore never boot the guest, and only
    a guidance that applies needs the OpenVMM executable, which the guidance's
    command names, so nvx.py looks it up only then."""
    if getattr(args, "dry_run", False) or hasattr(args, "restore_snapshot"):
        return []
    cpu_profile = cast(str | None, getattr(args, "cpu_profile", None))
    host = host_cpu_signature()
    if not cpu_profile_guidance_applies(cpu_profile, host):
        return []
    guidance = host_cpu_profile_guidance(
        cpu_profile,
        host,
        _format_command([os.fspath(nvx_rs.artifacts().openvmm)]),
        _hypervisor(cast(str, getattr(args, "hypervisor", "auto"))),
    )
    return [] if guidance is None else [f"--cold-boot-guidance={guidance}"]


def command_run(args: argparse.Namespace) -> int:
    """Run the run command with nvx-rs, which checks its options, builds the
    OpenVMM command line, and runs OpenVMM on the standard streams, and return
    the status that nvx.py exits with."""
    return nvx_rs.run(
        ["run", *getattr(args, NVX_RS_ARGUMENTS), *_cold_boot_guidance(args)],
        program=args.program,
    )


def command_sandbox(args: argparse.Namespace) -> int:
    """Run the sandbox command with nvx-rs, which checks its options and runs it
    on the standard streams, and return the status that nvx.py exits with."""
    return nvx_rs.run(
        ["sandbox", args.sandbox_operation, *getattr(args, NVX_RS_ARGUMENTS)],
        program=args.program,
    )


def command_collect_sources(_: argparse.Namespace) -> None:
    collect_release_sources(DockerBuildConfig())


def command_package(args: argparse.Namespace) -> None:
    package_release(
        version=args.version,
        destination=args.destination,
        include_source=args.include_source,
        force=args.force,
    )


def command_archive_release(args: argparse.Namespace) -> None:
    create_release_archive(args.source, args.destination)


def command_verify(_: argparse.Namespace) -> None:
    verify_source_tree()


def _add_guest_options(
    parser: argparse.ArgumentParser,
    *,
    allow_all: bool,
) -> None:
    choices = (*GUEST_NAMES, "all") if allow_all else GUEST_NAMES
    parser.add_argument(
        "--guest",
        choices=choices,
        default=InitramfsBuildConstants.DEFAULT_GUEST,
        help=(
            "guest userland to build "
            f"(default: {InitramfsBuildConstants.DEFAULT_GUEST})"
        ),
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="build directly on Linux instead of using Docker",
    )
    parser.add_argument(
        "--debug-kernel",
        action="store_true",
        help=(
            "also build the CI debug kernel (build/vmlinux-debug), which adds "
            "the soft-lockup, hung-task, and RCU stall diagnostics"
        ),
    )


def _add_openvmm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-restore", action="store_true")
    parser.add_argument(
        "--backend",
        choices=OPENVMM_TEST_BACKENDS,
        help=(
            "select build target: kvm=GNU, mshv=musl, whp=MSVC "
            "(default: native target for the host OS)"
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="initialize the private submodule")
    init.set_defaults(handler=command_init)

    guest = subparsers.add_parser("build-guest", help="build Linux guest artifacts")
    _add_guest_options(guest, allow_all=True)
    guest.set_defaults(handler=command_build_guest)

    guest_binary_checks = subparsers.add_parser(
        "check-guest-binaries",
        help="lint and test the Rust guest binaries for musl",
    )
    guest_binary_checks.add_argument(
        "--toolchain",
        help="Rust toolchain to use (default: the rust-toolchain.toml pin)",
    )
    guest_binary_checks.set_defaults(handler=command_check_guest_binaries)

    guest_binaries = subparsers.add_parser(
        "build-guest-binaries",
        help="build the pinned Rust executables for Linux guests",
    )
    guest_binaries.add_argument("--output", type=Path, required=True)
    guest_binaries.set_defaults(handler=command_build_guest_binaries)

    guest_binary_tests = subparsers.add_parser(
        "test-guest-binaries",
        help="test prebuilt Rust guest executables",
    )
    guest_binary_tests.add_argument("--binaries", type=Path, required=True)
    guest_binary_tests.set_defaults(handler=command_test_guest_binaries)

    kernel = subparsers.add_parser(
        "build-kernel",
        help="fetch, patch, and build the pinned kernel natively on Linux",
    )
    kernel.add_argument(
        "--debug",
        action="store_true",
        help=(
            "build the CI debug variant (build/vmlinux-debug) by applying "
            "kernel/config-microvm-debug"
        ),
    )
    kernel.set_defaults(handler=command_build_kernel)

    initramfs = subparsers.add_parser(
        "build-initramfs",
        help=(
            "build a selected guest initramfs (natively on Linux, or via "
            "Docker for guests that require it)"
        ),
    )
    initramfs.add_argument(
        "--guest",
        choices=GUEST_NAMES,
        default=InitramfsBuildConstants.DEFAULT_GUEST,
        help=(
            "guest userland to build "
            f"(default: {InitramfsBuildConstants.DEFAULT_GUEST})"
        ),
    )
    initramfs.set_defaults(handler=command_build_initramfs)

    distro_layer = subparsers.add_parser(
        "build-distro-layer",
        help="build a deterministic EROFS distro layer natively on Linux",
    )
    distro_layer.add_argument("--guest", choices=GUEST_NAMES, required=True)
    distro_layer.add_argument(
        "--output",
        type=Path,
        default=artifact_path(UbuntuBuildConstants.DISTRO_NAME),
    )
    distro_layer.add_argument("--replace", action="store_true")
    distro_layer.set_defaults(handler=command_build_distro_layer)

    determinism = subparsers.add_parser(
        "verify-guest-determinism",
        help="build Ubuntu guest artifacts twice and compare them",
    )
    determinism.add_argument("--guest", choices=GUEST_NAMES, required=True)
    determinism.add_argument(
        "--work-dir",
        type=Path,
        default=(
            BuildConstants.BUILD_DIR
            / InitramfsBuildConstants.DETERMINISM_DIRECTORY_NAME
        ),
    )
    determinism.set_defaults(handler=command_verify_guest_determinism)

    openvmm = subparsers.add_parser("build-openvmm", help="build OpenVMM")
    _add_openvmm_options(openvmm)
    openvmm.set_defaults(handler=command_build_openvmm)

    provenance = subparsers.add_parser(
        "record-openvmm-provenance",
        help="bind an existing OpenVMM binary to the pinned source revision",
    )
    provenance.set_defaults(handler=command_record_openvmm_provenance)

    nvx_rs_provenance = subparsers.add_parser(
        "record-nvx-rs-provenance",
        help="bind an existing nvx-rs executable to the nvx-rs sources",
    )
    nvx_rs_provenance.add_argument(
        "--target",
        help=(
            "Rust target that built nvx-rs (default: the host target that "
            "rustc reports)"
        ),
    )
    nvx_rs_provenance.set_defaults(handler=command_record_nvx_rs_provenance)

    kernel_provenance = subparsers.add_parser(
        "materialize-kernel-provenance-inputs",
        help="write kernel provenance inputs from raw run-head blobs",
    )
    kernel_provenance.set_defaults(handler=command_materialize_kernel_provenance_inputs)

    cache = subparsers.add_parser(
        "setup-cross-os-cache",
        help="install GNU tar and zstd for GitHub Actions cross-OS caches",
    )
    cache.set_defaults(handler=command_setup_cross_os_cache)

    classify_changes = subparsers.add_parser(
        "classify-ci-changes",
        help="select the CI suites that a list of changed files runs",
    )
    classify_changes.add_argument(
        "--event-name",
        choices=("pull_request", "push"),
        required=True,
    )
    classify_changes.add_argument(
        "--changed-files",
        type=Path,
        required=True,
        help="file listing one changed repository path per line",
    )
    classify_changes.add_argument(
        "--null",
        action="store_true",
        help="read NUL-delimited paths, as emitted by git diff --name-only -z",
    )
    classify_changes.add_argument(
        "--summary",
        type=Path,
        help="Markdown file to append the reasons for each decision to",
    )
    classify_changes.set_defaults(handler=command_classify_ci_changes)

    patch_identity = subparsers.add_parser(
        "kernel-patch-identity",
        help="print the digest of the kernel patch paths and contents that keys the kernel cache",
    )
    patch_identity.add_argument(
        "--root",
        type=Path,
        default=BuildConstants.REPO_ROOT / KernelBuildConstants.PATCH_DIRECTORY,
    )
    patch_identity.set_defaults(handler=command_kernel_patch_identity)

    source_identity = subparsers.add_parser(
        "guest-binary-source-identity",
        help=(
            "print the digest of the Rust guest binary source paths and contents "
            "that keys the guest image caches"
        ),
    )
    source_identity.add_argument(
        "--root",
        type=Path,
        default=BuildConstants.REPO_ROOT,
    )
    source_identity.set_defaults(handler=command_guest_binary_source_identity)

    required_ci = subparsers.add_parser(
        "check-required-ci",
        help="validate required GitHub Actions job results",
    )
    required_ci.add_argument(
        "--event-name",
        choices=("pull_request", "push", "schedule", "workflow_dispatch"),
        required=True,
    )
    required_ci.add_argument(
        "--same-repository",
        choices=("false", "true"),
        required=True,
    )
    required_ci.add_argument("--run-tests", choices=("false", "true"), required=True)
    required_ci.add_argument(
        "--run-workloads", choices=("false", "true"), required=True
    )
    required_ci.add_argument(
        "--run-openvmm-unit-tests", choices=("false", "true"), required=True
    )
    required_ci.add_argument(
        "--run-openvmm-vmm-tests", choices=("false", "true"), required=True
    )
    required_ci.set_defaults(handler=command_check_required_ci)

    static_elf = subparsers.add_parser(
        "check-static-elf",
        help=(
            "check that executables are static ELF files without an interpreter "
            "or an executable stack"
        ),
    )
    static_elf.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    static_elf.set_defaults(handler=command_check_static_elf)

    openvmm_unit_tests = subparsers.add_parser(
        "test-openvmm-unit",
        help="run OpenVMM unit and documentation tests",
    )
    openvmm_unit_tests.set_defaults(handler=command_test_openvmm_unit)

    openvmm_tests = subparsers.add_parser(
        "test-openvmm",
        help="run OpenVMM Petri VMM tests",
    )
    openvmm_tests.add_argument(
        "--backend",
        choices=OPENVMM_TEST_BACKENDS,
        required=True,
    )
    openvmm_tests.set_defaults(handler=command_test_openvmm)

    microvm_tests = subparsers.add_parser(
        "test-microvm",
        help="run NVX-owned OpenVMM microVM correctness tests",
    )
    configure_microvm_test_parser(microvm_tests)

    doctor = subparsers.add_parser(
        "doctor",
        help="qualify this host for the NVX time ABI",
    )
    configure_doctor_parser(doctor)

    aci_edge_sandboxes_tests = subparsers.add_parser(
        "test-aci-edge-sandboxes",
        help="run the aci_edge_sandboxes crate lifecycle test on a real hypervisor",
    )
    configure_aci_edge_sandboxes_test_parser(aci_edge_sandboxes_tests)

    agent_bundling_tests = subparsers.add_parser(
        "test-agent-bundling",
        help="check the aci_edge_sandboxes agent bundle staging without a hypervisor",
    )
    configure_agent_bundling_test_parser(agent_bundling_tests)

    adversarial_tests = subparsers.add_parser(
        "test-adversarial",
        help="run a brokered Copilot-driven adversarial campaign",
    )
    configure_adversarial_parser(adversarial_tests)

    build = subparsers.add_parser("build", help="build guest artifacts and OpenVMM")
    _add_guest_options(build, allow_all=True)
    _add_openvmm_options(build)
    build.set_defaults(handler=command_build)

    download = subparsers.add_parser(
        "download",
        help=(
            "download and install the platform package for HEAD or its nearest "
            "released first-parent ancestor"
        ),
    )
    download.add_argument(
        "--repository",
        default=DEFAULT_RELEASE_REPOSITORY,
        metavar="OWNER/REPOSITORY",
    )
    download.add_argument("--hypervisor", choices=HYPERVISORS, default="auto")
    download.set_defaults(handler=command_download)

    run = subparsers.add_parser("run", help="run an OpenVMM microVM")

    # nvx-rs checks the values and applies the defaults; these options define
    # the command line and its help, and argparse checks only the choices of
    # those that take a string. nvx.py keeps the options that decide on its
    # CPU-profile guidance.
    def run_option(*names: str, flag: bool = False, **kwargs: Any) -> None:
        run.add_argument(
            *names, action=_NvxRsOption, nargs=0 if flag else None, **kwargs
        )

    run_option("--guest", choices=GUEST_NAMES)
    run_option("--hypervisor", choices=HYPERVISORS, keep=True)
    run_option("--machine", choices=("microvm",))
    run_option(
        "--memory-mib",
        help=(
            "guest RAM in MiB (default by --guest: "
            + ", ".join(
                f"{name} {guest_descriptor(name).default_memory_mib}"
                for name in GUEST_NAMES
            )
            + ")"
        ),
    )
    run_option("--memory-capacity-mib")
    # nvx-rs converts the processor counts before it checks their choices, as
    # argparse checks those of an option of type int.
    run_option("--processors", metavar="{1,2,4,8}")
    run_option(
        "--cpu-profile",
        keep=True,
        metavar="ID",
        help=(
            "OpenVMM CPU profile: auto (OpenVMM's default) for the built-in "
            "profile of the host's CPU, a built-in profile ID, or host to "
            "derive a development profile from this host (doc/usage.md)"
        ),
    )
    run_option(
        "--mount",
        metavar="GUEST_TARGET,HOST_PATH[,ro|rw]",
        help=(
            "live-share a host directory, or a single regular host file without "
            "its directory; the mode follows the last comma, so HOST_PATH may "
            "contain commas when the mode is given; repeat for more shares, "
            "each with its own guest target and mode, as children of one "
            "virtio-fs device"
        ),
    )
    run_option("--mount-deny")
    run_option(
        "--mount-allow",
        metavar="HOST_PATH",
        help="expose one existing path inside a --mount-deny path again",
    )
    run_option(
        "--mount-write",
        metavar="HOST_PATH",
        help=(
            "make one existing path one of the only writable parts of a "
            "read-write --mount"
        ),
    )
    run_option(
        "--mount-owner",
        choices=MOUNT_OWNERS,
        help=(
            "host identity of the share's file operations: vmm (default) or "
            "caller, with guest root squashed to the owner of the shared "
            "directory or file (Linux only)"
        ),
    )
    run_option("--net", metavar="IPV4/PREFIX")
    run_option("--network-profile", choices=NETWORK_PROFILES)
    run_option("--network-egress", choices=("allow", "deny"))
    run_option("--network-ingress", choices=("allow", "deny"))
    run_option(
        "--network-egress-allow",
        metavar=NETWORK_EGRESS_RULE_METAVAR,
        help=NETWORK_EGRESS_RULE_HELP.format(action="allow"),
    )
    run_option(
        "--network-egress-deny",
        metavar=NETWORK_EGRESS_RULE_METAVAR,
        help=NETWORK_EGRESS_RULE_HELP.format(action="deny"),
    )
    run_option(
        "--network-egress-policy-file",
        metavar="PATH",
        help="load bounded MXC-shaped IPv4 and IPv6 egress rules from JSON",
    )
    run_option("--host-loopback", choices=("allow", "deny"))
    run_option("--network-proxy", metavar="IPV4:TCP-PORT")
    run_option("--host-loopback-forward")
    run_option("--outcome-report", help="write a bounded local JSON outcome report")
    run_option("--cmdline")
    run_option("--restore-snapshot", keep=True)
    run_option("--restore-processors", metavar="{1,2,4,8}")
    run_option("--restore-memory-mib")
    run_option("--restore-ready-path")
    run_option("--dry-run", flag=True, keep=True)
    run.set_defaults(handler=command_run, program=parser.prog, **{NVX_RS_ARGUMENTS: []})

    sandbox = subparsers.add_parser(
        "sandbox",
        help="run or manage workloads over EROFS layers and private ext4 scratch",
    )

    # nvx-rs checks the values and applies the defaults; these options define
    # the command line and its help.
    def sandbox_option(*names: str, flag: bool = False, **kwargs: Any) -> None:
        sandbox.add_argument(
            *names, action=_NvxRsOption, nargs=0 if flag else None, **kwargs
        )

    sandbox.add_argument(
        "sandbox_operation",
        nargs="?",
        choices=SANDBOX_OPERATIONS,
        default="run",
    )
    sandbox_option("--layer", metavar="ROLE,PATH,EROFS_UUID")
    sandbox_option("--scratch")
    sandbox_option("--state-dir")
    sandbox_option("--entrypoint")
    sandbox_option("--arg", dest="sandbox_arg")
    sandbox_option("--hostname")
    sandbox_option(
        "--workload-user",
        metavar="UID:GID",
        help="fixed non-root workload identity (default: 65534:65534)",
    )
    sandbox_option("--memory-max")
    sandbox_option("--pids-max")
    sandbox_option("--memory-mib")
    sandbox_option(
        "--timeout",
        help=(
            "control operation timeout, and the longest wait for another "
            "lifecycle transition of the sandbox, in seconds (default: 60)"
        ),
    )
    sandbox_option(
        "--exec-timeout-ms",
        help="guest workload timeout in milliseconds; zero disables it",
    )
    sandbox_option(
        "--cwd",
        help="absolute guest working directory for managed exec (default: /)",
    )
    sandbox_option(
        "--environment",
        metavar="KEY=VALUE",
        help="set the exact managed exec environment; repeat for multiple entries",
    )
    sandbox_option(
        "--environment-file",
        metavar="PATH",
        help="read the exact managed exec environment from a JSON string array",
    )
    sandbox_option(
        "--inherit-default-environment",
        flag=True,
        help=(
            "layer the managed exec environment over the guest default "
            "environment instead of replacing it"
        ),
    )
    sandbox_option("--hypervisor", choices=HYPERVISORS)
    sandbox_option(
        "--mount",
        metavar="GUEST_TARGET,HOST_PATH[,ro|rw]",
        help=(
            "live-share a host directory, or a single regular host file without "
            "its directory, inside the container rootfs; the mode follows the "
            "last comma, so HOST_PATH may contain commas when the mode is given; "
            "repeat to attach more shares, each with its own target and mode, as "
            "children of one virtio-fs device"
        ),
    )
    sandbox_option(
        "--mount-deny",
        metavar="HOST_PATH",
        help=(
            "hide one existing path inside a --mount host directory; with "
            "several --mount options, it applies to the --mount before it"
        ),
    )
    sandbox_option(
        "--mount-allow",
        metavar="HOST_PATH",
        help=(
            "expose one existing path inside a --mount-deny path of the same "
            "share again; the hidden directories on the way list only it"
        ),
    )
    sandbox_option(
        "--mount-write",
        metavar="HOST_PATH",
        help=(
            "make one existing path one of the only writable parts of a "
            "read-write --mount; the rest of the share is read-only"
        ),
    )
    sandbox_option(
        "--mount-owner",
        choices=MOUNT_OWNERS,
        help=(
            "host identity of the share's file operations: vmm (default) or "
            "caller, the workload identity with guest root squashed to the "
            "owner of the shared directory or file (Linux only)"
        ),
    )
    sandbox_option("--net", metavar="IPV4/PREFIX")
    sandbox_option("--network-profile", choices=NETWORK_PROFILES)
    sandbox_option("--network-egress", choices=("allow", "deny"))
    sandbox_option("--network-ingress", choices=("allow", "deny"))
    sandbox_option(
        "--network-egress-allow",
        metavar=NETWORK_EGRESS_RULE_METAVAR,
        help=NETWORK_EGRESS_RULE_HELP.format(action="allow"),
    )
    sandbox_option(
        "--network-egress-deny",
        metavar=NETWORK_EGRESS_RULE_METAVAR,
        help=NETWORK_EGRESS_RULE_HELP.format(action="deny"),
    )
    sandbox_option(
        "--network-egress-policy-file",
        metavar="PATH",
        help=(
            "load bounded IPv4 and IPv6 ranges and rule-local exclusions for "
            "run/provision"
        ),
    )
    sandbox_option("--host-loopback", choices=("allow", "deny"))
    sandbox_option("--network-proxy", metavar="IPV4:TCP-PORT")
    sandbox_option("--host-loopback-forward")
    sandbox_option(
        "--outcome-report",
        help="write a bounded local JSON outcome report for run or exec",
    )
    sandbox_option("--cmdline")
    sandbox_option(
        "--dry-run",
        flag=True,
        help="print the one-shot run command without launching it",
    )
    sandbox.set_defaults(
        handler=command_sandbox, program=parser.prog, **{NVX_RS_ARGUMENTS: []}
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="run the OpenVMM-native benchmark coordinator",
    )
    configure_benchmark_parser(benchmark, BuildConstants.REPO_ROOT)
    benchmark.set_defaults(handler=command_benchmark)

    performance = subparsers.add_parser(
        "performance",
        help="collect, persist, and gate CI performance results",
    )
    configure_performance_parser(performance)

    sources = subparsers.add_parser(
        "collect-sources",
        help="materialize verified Linux, Alpine, and Ubuntu release sources",
    )
    sources.set_defaults(handler=command_collect_sources)

    alpine_sources = subparsers.add_parser(
        "collect-alpine-sources",
        help="collect exact Alpine recipes and upstream sources",
    )
    configure_alpine_sources_parser(alpine_sources)

    ubuntu_sources = subparsers.add_parser(
        "collect-ubuntu-sources",
        help="collect exact Ubuntu source packages",
    )
    configure_ubuntu_sources_parser(ubuntu_sources)

    linux_source_archive = subparsers.add_parser(
        "create-linux-source-archive",
        help="create the Linux corresponding-source archive from pinned inputs",
    )
    configure_linux_source_archive_parser(linux_source_archive)

    package = subparsers.add_parser("package", help="stage a binary distribution")
    package.add_argument("--version")
    package.add_argument("--destination", type=Path)
    source_mode = package.add_mutually_exclusive_group(required=True)
    source_mode.add_argument("--include-source", action="store_true")
    source_mode.add_argument(
        "--binary-only",
        action="store_true",
        help="stage binaries only; corresponding source must be published separately",
    )
    package.add_argument("--force", action="store_true")
    package.set_defaults(handler=command_package)

    archive_release = subparsers.add_parser(
        "archive-release",
        help="create a deterministic archive from a staged distribution",
    )
    archive_release.add_argument("--source", type=Path, required=True)
    archive_release.add_argument("--destination", type=Path, required=True)
    archive_release.set_defaults(handler=command_archive_release)

    verify = subparsers.add_parser("verify", help="verify source and submodule inputs")
    verify.set_defaults(handler=command_verify)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = args.handler(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except (
        ScriptError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return result if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
