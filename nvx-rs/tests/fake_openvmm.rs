//! Drives VM instances end to end against the `nvx-fake-openvmm` test double, which serves the
//! real control protocol without running a VM.
#![cfg(all(
    feature = "host",
    feature = "testing",
    any(target_os = "linux", windows)
))]

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use nvx::host::instance::{self, Instance, WorkloadOutcome};
use nvx::host::launch::{LaunchSpec, PROCESS_COMMAND_LINE_LIMIT};
use nvx::host::sandbox::mount::aggregate_child_name;
use nvx::protocol::{Workload, WorkloadEnvironment};
use nvx::{Artifacts, ErrorKind, Hypervisor};
use serde_json::{Value, json};

const TIMEOUT: Duration = Duration::from_secs(30);

/// Creates the fake guest artifacts and an instance directory below `directory`.
fn setup(directory: &Path) -> (LaunchSpec, Instance) {
    fs::write(directory.join("vmlinux"), b"kernel").unwrap();
    fs::write(directory.join("initramfs.cpio.gz"), b"initrd").unwrap();
    let artifacts = Artifacts::new(
        env!("CARGO_BIN_EXE_nvx-fake-openvmm"),
        directory.join("vmlinux"),
        directory.join("initramfs.cpio.gz"),
    );
    let hypervisor = Hypervisor::platform_default().expect("tests run on Linux or Windows");
    let state = directory.join("vm");
    instance::create_private_dir(&state).unwrap();
    (
        LaunchSpec::new(artifacts, hypervisor),
        Instance::new(state, "test"),
    )
}

fn workload(argv: &[&str]) -> Workload<'static> {
    Workload {
        argv: argv.iter().map(|argument| (*argument).to_owned()).collect(),
        timeout_ms: 0,
        cwd: None,
        environment: WorkloadEnvironment::Default,
    }
}

struct Output {
    outcome: WorkloadOutcome,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run(instance: &Instance, workload: &Workload<'_>, cancel: &AtomicBool) -> Output {
    let runtime = instance.runtime().unwrap().expect("the instance runs");
    let capability = instance.read_capability().unwrap();
    let execution = instance
        .exec(&runtime, &capability, workload, TIMEOUT, TIMEOUT)
        .unwrap();
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    let outcome = execution
        .pump(
            &mut |chunk: &[u8]| {
                stdout.extend_from_slice(chunk);
                Ok(())
            },
            &mut |chunk: &[u8]| {
                stderr.extend_from_slice(chunk);
                Ok(())
            },
            cancel,
        )
        .unwrap();
    Output {
        outcome,
        stdout,
        stderr,
    }
}

#[test]
fn instances_start_run_workloads_and_stop() {
    let directory = tempfile::tempdir().unwrap();
    let (spec, instance) = setup(directory.path());
    assert_eq!(instance.reconcile().unwrap(), None);
    let started = instance.start(&spec, TIMEOUT).unwrap();
    assert_eq!(instance.reconcile().unwrap(), Some(started.runtime.clone()));
    // Only a running instance keeps a launch capability.
    assert!(instance.launch().unwrap().is_none());

    let idle = AtomicBool::new(false);
    let output = run(
        &instance,
        &workload(&["/bin/sh", "-c", "echo hello; echoerr warning; exit 3"]),
        &idle,
    );
    assert_eq!(output.outcome, WorkloadOutcome::Exited(3));
    assert_eq!(output.stdout, b"hello\n");
    assert_eq!(output.stderr, b"warning\n");
    let signaled = run(&instance, &workload(&["/bin/sh", "-c", "signal 15"]), &idle);
    assert_eq!(signaled.outcome, WorkloadOutcome::Signaled(15));
    let mut timed = workload(&["/bin/sh", "-c", "sleep 10000"]);
    timed.timeout_ms = 50;
    assert_eq!(
        run(&instance, &timed, &idle).outcome,
        WorkloadOutcome::TimedOut
    );
    let mut missing = workload(&["/bin/pwd"]);
    missing.cwd = Some("/missing");
    let refused = run(&instance, &missing, &idle);
    assert_eq!(refused.outcome, WorkloadOutcome::WorkingDirectory);
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("/missing"),
        "{:?}",
        refused.stderr
    );

    let stopped = instance.stop(&started.runtime, TIMEOUT).unwrap();
    assert!(!stopped.forced);
    let report = stopped.report.expect("the VM reports its outcome");
    assert_eq!(report.category.as_deref(), Some("success"));
    assert_eq!(report.teardown_complete, Some(true));
    assert_eq!(instance.reconcile().unwrap(), None);
    assert!(instance.read_capability().is_err());
}

#[test]
fn timeouts_are_checked_before_they_are_used() {
    let directory = tempfile::tempdir().unwrap();
    let (spec, instance) = setup(directory.path());
    for timeout in [Duration::ZERO, Duration::MAX] {
        let error = instance.start(&spec, timeout).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
        // Nothing was launched.
        assert!(instance.launch().unwrap().is_none());
        assert!(instance.runtime().unwrap().is_none());
    }
    let started = instance.start(&spec, TIMEOUT).unwrap();
    let capability = instance.read_capability().unwrap();
    let echo = workload(&["/bin/echo", "hello"]);
    for (control_timeout, response_grace) in [
        (Duration::ZERO, TIMEOUT),
        (Duration::MAX, TIMEOUT),
        (TIMEOUT, Duration::MAX),
    ] {
        let Err(error) = instance.exec(
            &started.runtime,
            &capability,
            &echo,
            control_timeout,
            response_grace,
        ) else {
            panic!("exec accepted {control_timeout:?} and {response_grace:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
    }
    for timeout in [Duration::ZERO, Duration::MAX] {
        let error = instance.stop(&started.runtime, timeout).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
    }
    // The refused requests left the VM running.
    let idle = AtomicBool::new(false);
    let output = run(&instance, &echo, &idle);
    assert_eq!(output.outcome, WorkloadOutcome::Exited(0));
    assert_eq!(output.stdout, b"hello\n");
    instance.stop(&started.runtime, TIMEOUT).unwrap();
}

#[test]
fn launches_beyond_the_process_command_line_limit_are_refused() {
    let directory = tempfile::tempdir().unwrap();
    let (mut spec, instance) = setup(directory.path());
    // The program's path counts toward the command line, which every host holds to the limit.
    spec.openvmm = directory
        .path()
        .join("o".repeat(PROCESS_COMMAND_LINE_LIMIT));
    let error = instance.start(&spec, TIMEOUT).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
    // Nothing was launched or recorded.
    assert!(instance.launch().unwrap().is_none());
    assert!(instance.runtime().unwrap().is_none());
    assert!(instance.read_capability().is_err());
}

#[test]
fn cancellation_ends_a_running_workload() {
    let directory = tempfile::tempdir().unwrap();
    let (spec, instance) = setup(directory.path());
    let started = instance.start(&spec, TIMEOUT).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let canceller = thread::spawn({
        let cancel = Arc::clone(&cancel);
        move || {
            thread::sleep(Duration::from_millis(200));
            cancel.store(true, Ordering::Release);
        }
    });
    let output = run(
        &instance,
        &workload(&["/bin/sh", "-c", "echo started; sleep 60000"]),
        &cancel,
    );
    canceller.join().unwrap();
    assert_eq!(output.outcome, WorkloadOutcome::Cancelled);
    assert_eq!(output.stdout, b"started\n");
    instance.stop(&started.runtime, TIMEOUT).unwrap();
}

#[test]
fn guests_without_required_features_are_refused() {
    let directory = tempfile::tempdir().unwrap();
    let (mut spec, instance) = setup(directory.path());
    // Cancellation alone.
    spec.kernel_parameters = "fake_guest_features=1".to_owned();
    let error = instance.start(&spec, TIMEOUT).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unavailable);
    assert!(
        error.message().contains("lacks control features"),
        "{error}"
    );
    assert!(error.message().contains("sandbox test failed to start"));
    // The rejected VM is gone and leaves no runtime state behind.
    assert_eq!(instance.reconcile().unwrap(), None);
    assert!(instance.runtime().unwrap().is_none());
}

#[test]
fn launches_that_exit_fail_to_start() {
    let directory = tempfile::tempdir().unwrap();
    let (mut spec, instance) = setup(directory.path());
    spec.kernel_parameters = "fake_exit_on_start=4".to_owned();
    let error = instance.start(&spec, TIMEOUT).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Exited, "{error}");
    assert!(
        error
            .message()
            .contains(&instance.log_path().display().to_string()),
        "{error}"
    );
    assert_eq!(instance.reconcile().unwrap(), None);
}

/// Runs the fake in `directory` as `nvx.py run` runs OpenVMM on the terminal: with two
/// processors and the backend of this host, and then `arguments`.
fn run_microvm(directory: &Path, arguments: &[&str]) -> std::process::Output {
    let hypervisor = Hypervisor::platform_default().expect("tests run on Linux or Windows");
    Command::new(env!("CARGO_BIN_EXE_nvx-fake-openvmm"))
        .args([
            "--single-process",
            "--machine",
            "microvm",
            "--processors",
            "2",
        ])
        .args(["--hypervisor", hypervisor.as_str()])
        .args(arguments)
        .current_dir(directory)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn text(data: &[u8]) -> String {
    String::from_utf8_lossy(data).into_owned()
}

#[test]
fn runs_boot_or_restore_a_microvm_on_the_terminal() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let path = |name: &str| root.join(name).to_str().unwrap().to_owned();
    fs::write(root.join("vmlinux"), b"kernel").unwrap();
    fs::write(root.join("initramfs.cpio.gz"), b"initrd").unwrap();
    fs::create_dir(root.join("reports")).unwrap();
    fs::create_dir(root.join("snapshot")).unwrap();
    let (kernel, initrd) = (path("vmlinux"), path("initramfs.cpio.gz"));
    let boot = ["--memory", "128M", "--kernel", &kernel, "--initrd", &initrd];

    // A cold boot prints that it booted, writes its outcome report, records its command line next
    // to the kernel, and exits with 0.
    let report = root.join("reports").join("boot.json");
    let report_path = report.to_str().unwrap().to_owned();
    let mut arguments = boot.to_vec();
    arguments.extend(["--cpu-profile", "host", "--microvm-report", &report_path]);
    let output = run_microvm(root, &arguments);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert_eq!(
        text(&output.stdout),
        "nvx-fake-openvmm: booted the microVM\n"
    );
    assert_eq!(text(&output.stderr), "");
    let document: Value = serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(
        document["outcome"],
        json!({"operation": "run", "category": "success", "status_code": 0})
    );
    assert_eq!(document["network_policy"]["status"], "not-requested");
    let recorded: Vec<String> =
        serde_json::from_str(&fs::read_to_string(root.join("fake-openvmm-reports.json")).unwrap())
            .unwrap();
    assert_eq!(recorded[recorded.len() - 1], report_path);

    // The kernel command line sets the status, also of a restore, which prints that it restored
    // the snapshot and records its command line next to it.
    let mut arguments = boot.to_vec();
    arguments.extend(["--cmdline", "quiet fake_exit_status=5"]);
    let output = run_microvm(root, &arguments);
    assert_eq!(output.status.code(), Some(5), "{}", text(&output.stderr));
    let snapshot = path("snapshot");
    let restore = [
        "--restore-snapshot",
        &snapshot,
        "--restore-processors",
        "1",
        "--cmdline",
        "fake_exit_status=6",
    ];
    fs::remove_file(root.join("fake-openvmm-.json")).unwrap();
    let output = run_microvm(root, &restore);
    assert_eq!(output.status.code(), Some(6), "{}", text(&output.stderr));
    assert_eq!(
        text(&output.stdout),
        "nvx-fake-openvmm: restored the microVM\n"
    );
    let recorded: Vec<String> =
        serde_json::from_str(&fs::read_to_string(root.join("fake-openvmm-.json")).unwrap())
            .unwrap();
    assert_eq!(recorded[recorded.len() - restore.len()..], restore);

    // A restore without its snapshot fails as OpenVMM fails it, and OpenVMM refuses a command
    // line that it does not take with status 2.
    let missing = path("missing");
    let output = run_microvm(root, &["--restore-snapshot", &missing]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(text(&output.stdout), "");
    assert_eq!(
        text(&output.stderr),
        format!("nvx-fake-openvmm: cannot restore {missing}: no such snapshot directory\n")
    );
    let output = run_microvm(root, &["--restore-snapshot", &snapshot, "--memory", "128M"]);
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(
        text(&output.stderr),
        "nvx-fake-openvmm: a restore takes no --memory\n"
    );
}

#[test]
fn runs_find_absolute_policy_paths_in_relative_share_roots() {
    // nvx.py run and nvx-rs run pass the host path of each share as given, which may be relative
    // to the working directory, and with several shares, absolute policy paths, which OpenVMM
    // attributes to the share whose root contains them once it has resolved both.
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::write(root.join("vmlinux"), b"kernel").unwrap();
    fs::write(root.join("initramfs.cpio.gz"), b"initrd").unwrap();
    for path in ["share/hidden/shown", "share/writable", "other", "outside"] {
        fs::create_dir_all(root.join(path)).unwrap();
    }
    let absolute = |relative: &str| root.join(relative).to_str().unwrap().to_owned();
    let (kernel, initrd) = (absolute("vmlinux"), absolute("initramfs.cpio.gz"));
    let boot = ["--memory", "128M", "--kernel", &kernel, "--initrd", &initrd];
    let share = format!("{},share,rw", aggregate_child_name(0, "/data"));
    let other = format!("{},other,ro", aggregate_child_name(1, "/other"));
    let aggregate = [
        "--mount-aggregate",
        "/run/nvx/shares",
        "--mount-child",
        &share,
        "--mount-child",
        &other,
    ];
    let (hidden, shown, writable) = (
        absolute("share/hidden"),
        absolute("share/hidden/shown"),
        absolute("share/writable"),
    );
    let policy = [
        "--mount-deny",
        &hidden,
        "--mount-allow",
        &shown,
        "--mount-write",
        &writable,
    ];
    let mut arguments = boot.to_vec();
    arguments.extend(aggregate);
    arguments.extend(policy);
    let output = run_microvm(root, &arguments);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert_eq!(
        text(&output.stdout),
        "nvx-fake-openvmm: booted the microVM\n"
    );

    // One share with a relative root takes absolute policy paths too, beside the relative ones
    // that OpenVMM resolves in it.
    let mut arguments = boot.to_vec();
    arguments.extend([
        "--mount",
        "/data,share,rw",
        "--mount-deny",
        &hidden,
        "--mount-allow",
        "hidden/shown",
    ]);
    let output = run_microvm(root, &arguments);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));

    // A path outside every share stays refused, as one that does not exist does.
    for path in [absolute("outside"), absolute("share/missing")] {
        let mut arguments = boot.to_vec();
        arguments.extend(aggregate);
        arguments.extend(["--mount-deny", &path]);
        let output = run_microvm(root, &arguments);
        assert_eq!(output.status.code(), Some(2), "{path}");
        assert_eq!(
            text(&output.stderr),
            format!("nvx-fake-openvmm: invalid --mount-deny {path}\n")
        );
    }
}
