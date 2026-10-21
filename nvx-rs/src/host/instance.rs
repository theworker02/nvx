//! Supervision of one VM instance: its OpenVMM process, its control endpoint, and the records
//! that name them, through the [`store`](super::store), the [`supervisor`], and the control
//! [`session`].
//!
//! An instance lives in a [`SandboxStore`] of its own: a directory private to the current user,
//! in which the store keeps the instance's lock, the journal of its launch, and the runtime
//! record of its VM, and in which the caller may keep configuration records of its own under
//! other names:
//!
//! ```text
//! <dir>/
//!   sandbox.lock            the lock that serializes the instance's lifecycle transitions
//!   launch.json             the journal of a start in progress, or of one that was interrupted
//!   runtime.json            the runtime record of the running VM
//!   control.capability      launch capability (running only)
//!   control.sock            Linux control endpoint (running only)
//!   openvmm.log             OpenVMM output of the latest start
//!   outcome.json            OpenVMM outcome report of the latest run
//! ```
//!
//! Lifecycle transitions run under the instance's lock, through the [`LockedInstance`] that
//! [`Instance::lock`] returns, so that every operation can run in a different process:
//!
//! - [`LockedInstance::find`] finds the instance's VM as an orchestrator does after it restarts
//!   within the same host boot, with [`supervisor::reattach`]: it ends a start whose caller died,
//!   forgets a VM that exited, and compares the caller's runtime identity with the one that a
//!   running VM was launched with. A VM of another runtime identity is [`Found::Mismatched`]: it
//!   runs through artifacts or contracts that the caller does not have, so the instance does not
//!   drive it.
//! - [`LockedInstance::start`] launches a detached OpenVMM process with the managed lifecycle
//!   through [`supervisor::launch`], passes it a fresh 32-byte capability through standard input,
//!   and waits until the guest agent answers on the authenticated control console. The agent must
//!   also advertise the control features in [`GuestFeatures::REQUIRED`]; a guest image that lacks
//!   one is terminated, because it would silently ignore the policy or request that needs the
//!   feature. Start then hands the agent the table of mapped host paths, and fails if the agent
//!   cannot mount them all.
//! - [`LockedInstance::stop`] asks the guest to shut down and falls back to terminating OpenVMM
//!   through its verified process identity.
//!
//! [`Instance::exec`] runs a workload through the control console of a [`Running`] VM without the
//! lock, and [`Execution::pump`] streams its output live. The guest serializes executions: a
//! second execution waits until the first finishes. Messages call the instance `sandbox NAME`,
//! after the name that [`Instance::create`] or [`Instance::open`] gives it.

use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde::de::DeserializeOwned;

use super::artifacts::Artifacts;
use super::launch::{LaunchSpec, timeout_problem};
use super::platform;
use super::process::KILL_TIMEOUT;
use super::report::{self, Report};
use super::session::{self, ControlSession, ExecEvent, SessionError};
use super::store::{
    IdentityPart, LockedStore, OUTCOME_NAME, Recovery, RuntimeIdentity, RuntimeRecord, SOCKET_NAME,
    SandboxStore,
};
use super::supervisor::{self, Reattached, Starting, StopOutcome, Vmm, VmmCommand};
use crate::error::{Error, ErrorKind, Result};
use crate::protocol::{
    CAPABILITY_LEN, CWD_FAILED, ExitCategory, GuestFeatures, LAUNCH_FAILED, MAPPINGS_INCOMPLETE,
    MAX_OUTPUT_BYTES, Workload,
};

/// Longest delay between a cancellation request and its delivery to the guest.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Builds an error mapper that preserves an I/O source under `context`.
fn io_error(context: String) -> impl FnOnce(io::Error) -> Error {
    move |error| Error::io(context).with_source(error)
}

/// Returns create-and-write options that request owner-only permissions on Unix.
fn private_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.write(true).create(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options
}

/// Atomically replaces `path` with `contents`, readable only by the current user, as callers do
/// with the configuration records that they keep beside an instance's store.
pub fn write_private(path: &Path, contents: &[u8]) -> Result<()> {
    let mut suffix = [0u8; 8];
    getrandom::fill(&mut suffix)
        .map_err(|error| Error::other("cannot name a temporary file").with_source(error))?;
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temporary = path.with_file_name(format!(".{name}.{}.tmp", u64::from_le_bytes(suffix)));
    let written = (|| -> io::Result<()> {
        let mut file = private_options().create_new(true).open(&temporary)?;
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written.map_err(io_error(format!("cannot write {}", path.display())))
}

/// Atomically replaces `path` with `value` as pretty JSON and a newline, readable only by the
/// current user.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut text = serde_json::to_vec_pretty(value)
        .map_err(|error| Error::other("cannot encode sandbox state").with_source(error))?;
    text.push(b'\n');
    write_private(path, &text)
}

/// Reads the JSON at `path`, or returns `None` if the file does not exist.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    let text = match fs::read(path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io_error(format!("cannot read {}", path.display()))(error)),
    };
    serde_json::from_slice(&text).map(Some).map_err(|error| {
        Error::io(format!("sandbox state {} is malformed", path.display())).with_source(error)
    })
}

/// Removes the file at `path` if it exists.
pub fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => {
            Err(io_error(format!("cannot remove {}", path.display()))(error))
        }
        _ => Ok(()),
    }
}

/// Returns a fresh launch capability, which is never all zero.
pub fn generate_capability() -> Result<[u8; CAPABILITY_LEN]> {
    let mut capability = [0u8; CAPABILITY_LEN];
    while capability == [0; CAPABILITY_LEN] {
        getrandom::fill(&mut capability).map_err(|error| {
            Error::other("cannot generate a control capability").with_source(error)
        })?;
    }
    Ok(capability)
}

/// Returns a control endpoint of an instance whose Linux control socket is `socket`: the socket
/// on Linux, and a fresh named pipe, whose name always has the same length, on Windows, as a
/// launch chooses it.
fn control_endpoint(socket: &Path) -> Result<String> {
    platform::control_endpoint(socket)
        .map_err(|error| Error::io("cannot choose a control endpoint").with_source(error))
}

/// Fails with [`ErrorKind::Unsupported`] if OpenVMM's command line for `spec`, launched for an
/// instance in the directory `dir`, would not fit the limit of a Windows process command line;
/// see [`LaunchSpec::check_process_command_line`].
///
/// The directory need not exist yet, so that a caller can check a launch before it creates the
/// instance. Every control endpoint of an instance has the same length.
pub fn check_launch(dir: &Path, spec: &LaunchSpec) -> Result<()> {
    let dir = std::path::absolute(dir).map_err(io_error(format!(
        "cannot resolve the instance directory {}",
        dir.display()
    )))?;
    let endpoint = control_endpoint(&dir.join(SOCKET_NAME))?;
    spec.check_process_command_line(&endpoint, &dir.join(OUTCOME_NAME))
}

/// Classifies a session failure.
fn session_kind(error: &SessionError) -> ErrorKind {
    match error {
        SessionError::TimedOut => ErrorKind::TimedOut,
        SessionError::ProcessExited => ErrorKind::Exited,
        SessionError::Refused(_) | SessionError::Io(_) => ErrorKind::Io,
        SessionError::Closed
        | SessionError::Reset
        | SessionError::AuthenticationFailed
        | SessionError::Protocol(_) => ErrorKind::Protocol,
    }
}

/// Explains why a launch did not become ready.
fn readiness_failure(error: &SessionError, runs_as_host_user: bool) -> String {
    let mut reason = error.to_string();
    // An image that predates account creation ends its boot, before the guest agent starts, when
    // it finds no account for the host user that the workloads run as. The VM then vanishes, or
    // its control endpoint closes first.
    let vm_ended = matches!(error, SessionError::ProcessExited | SessionError::Closed);
    if runs_as_host_user && vm_ended {
        reason.push_str(
            "; a guest image that predates workload account creation exits during boot when \
             workloads run under the host user's IDs, so use an NVX release or build that \
             includes it",
        );
    }
    reason
}

/// A VM that [`LockedInstance::start`] launched and the guest agent of which is ready.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Started {
    /// The runtime record of its launch: its OpenVMM process, its control endpoint, and the
    /// runtime identity that it was launched with.
    pub runtime: RuntimeRecord,
    /// Time from launch to guest readiness.
    pub boot: Duration,
}

/// A VM that [`LockedInstance::stop`] stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stopped {
    /// Whether OpenVMM had to be terminated because the guest did not shut down in time.
    pub forced: bool,
    /// Summary of OpenVMM's outcome report, if it wrote one.
    pub report: Option<Report>,
}

/// The running VM of an instance, which [`LockedInstance::find`] found under the caller's runtime
/// identity, and the capability with which its control sessions attach.
pub struct Running {
    /// The VM, held through its verified process identity.
    vmm: Vmm,
    /// The launch capability of the VM.
    capability: [u8; CAPABILITY_LEN],
}

impl Running {
    /// Returns the VM, held through its verified process identity.
    pub fn vmm(&self) -> &Vmm {
        &self.vmm
    }

    /// Returns the runtime record of the VM's launch.
    pub fn runtime(&self) -> &RuntimeRecord {
        self.vmm.record()
    }
}

impl fmt::Debug for Running {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The capability authenticates control sessions, so it is never shown.
        formatter
            .debug_struct("Running")
            .field("vmm", &self.vmm)
            .finish_non_exhaustive()
    }
}

/// The running VM of an instance, which [`LockedInstance::find`] found under another runtime
/// identity than the caller's.
#[derive(Debug)]
pub struct Mismatched {
    /// The instance's name in messages.
    name: String,
    /// The VM, held through its verified process identity.
    vmm: Vmm,
    /// The parts of the runtime identity that differ.
    differences: Vec<IdentityPart>,
}

impl Mismatched {
    /// Returns the VM, held through its verified process identity.
    pub fn vmm(&self) -> &Vmm {
        &self.vmm
    }

    /// Returns the parts of the runtime identity that differ, as
    /// [`RuntimeIdentity::differences`] returns them.
    pub fn differences(&self) -> &[IdentityPart] {
        &self.differences
    }

    /// Returns the error that reports the VM, which the instance does not drive: only the build
    /// and artifacts that started it speak its contracts.
    pub fn error(&self) -> Error {
        let parts: Vec<&str> = self.differences.iter().map(|part| part.name()).collect();
        Error::other(format!(
            "sandbox {} runs under another runtime identity, which differs in {}, so it is not \
             driven: stop it with the NVX build and artifacts that started it (OpenVMM process {})",
            self.name,
            parts.join(", "),
            self.vmm.record().process().pid
        ))
    }
}

/// What [`LockedInstance::find`] found of an instance's VM.
#[derive(Debug)]
pub enum Found {
    /// No VM of the instance runs: none was started, the last one stopped or exited, or a start
    /// that was interrupted is over.
    Stopped,
    /// The VM runs under the caller's runtime identity.
    Running(Running),
    /// The VM runs under another runtime identity, which the caller reports, as
    /// [`Mismatched::error`] does.
    Mismatched(Mismatched),
}

/// One VM instance in a sandbox store of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instance {
    /// The store of the instance's runtime records, logs, and control endpoint.
    store: SandboxStore,
    /// Sandbox name used in diagnostics.
    name: String,
}

impl Instance {
    /// Creates the directory `dir` of a new instance, whose parent must exist, private to the
    /// current user, as [`SandboxStore::create`] does, and supervises the instance in it, which
    /// messages call `sandbox NAME`.
    pub fn create(dir: impl Into<PathBuf>, name: impl Into<String>) -> Result<Self> {
        Ok(Self {
            store: SandboxStore::create(dir)?,
            name: name.into(),
        })
    }

    /// Supervises the instance in the existing directory `dir`, which must be private to the
    /// current user, as [`SandboxStore::open`] requires, and which messages call `sandbox NAME`.
    pub fn open(dir: impl Into<PathBuf>, name: impl Into<String>) -> Result<Self> {
        Ok(Self {
            store: SandboxStore::open(dir)?,
            name: name.into(),
        })
    }

    /// Returns the store of the instance's runtime records.
    pub fn store(&self) -> &SandboxStore {
        &self.store
    }

    /// Returns the instance's directory, as an absolute path.
    pub fn dir(&self) -> &Path {
        self.store.dir()
    }

    /// Returns the instance's name in messages.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the path of the OpenVMM log of the latest start.
    pub fn log_path(&self) -> PathBuf {
        self.store.log_path()
    }

    /// Returns the path of OpenVMM's outcome report of the latest run.
    pub fn report_path(&self) -> PathBuf {
        self.store.outcome_path()
    }

    /// Takes the instance's lock, which serializes its lifecycle transitions, waiting for it
    /// until `deadline`, or for as long as it takes without one, as [`SandboxStore::lock`] does.
    pub fn lock(&self, deadline: Option<Instant>) -> Result<LockedInstance> {
        Ok(LockedInstance {
            locked: self.store.lock(deadline)?,
            instance: self.clone(),
        })
    }

    /// Describes why this instance did not start and points the caller to its log.
    fn start_failure(&self, kind: ErrorKind, reason: &str) -> Error {
        Error::new(
            kind,
            format!(
                "sandbox {} failed to start: {reason}; see {}",
                self.name,
                self.log_path().display()
            ),
        )
    }

    /// Ends a launch that did not become ready, as [`Starting::abandon`] does, and describes why
    /// the instance did not start.
    ///
    /// A launch whose OpenVMM cannot be ended keeps its journal, and the next transition of the
    /// instance ends it, when [`LockedInstance::find`] recovers it.
    fn abort_start(&self, starting: Starting<'_>, kind: ErrorKind, reason: &str) -> Error {
        let unended = match starting.abandon() {
            Ok(Recovery::Indeterminate(unended)) => unended,
            Ok(_) => return self.start_failure(kind, reason),
            Err(error) => error.to_string(),
        };
        self.start_failure(
            kind,
            &format!(
                "{reason}; the launch could not be ended, and remains for recovery: {unended}"
            ),
        )
    }

    /// Ends a launch whose guest image lacks control features that NVX needs.
    ///
    /// An older image would boot and answer, but it would silently ignore host path mappings and
    /// cancellation requests, so the VM must not run at all.
    fn reject_guest(&self, starting: Starting<'_>, missing: &[&str]) -> Error {
        let reason = format!(
            "the guest image lacks control features that the openvmm backend needs ({}); use an \
             NVX release or build that includes them, for example through {}",
            missing.join(", "),
            Artifacts::ENV_DIR
        );
        self.abort_start(starting, ErrorKind::Unavailable, &reason)
    }

    /// Maps a control-session failure onto an instance-facing error and message.
    fn session_error(&self, error: SessionError, control_timeout: Duration) -> Error {
        match error {
            SessionError::ProcessExited => {
                Error::exited(format!("sandbox {} stopped unexpectedly", self.name))
            }
            SessionError::TimedOut => Error::timed_out(format!(
                "sandbox {} did not accept a control session within {} s; another exec may \
                 still be running",
                self.name,
                control_timeout.as_secs()
            )),
            other => Error::new(
                session_kind(&other),
                format!(
                    "the control session with sandbox {} failed: {other}",
                    self.name
                ),
            )
            .with_source(other),
        }
    }

    /// Starts `workload` in the instance's running VM, `running`, without the instance's lock.
    ///
    /// It waits up to `control_timeout` for the control session, which the guest grants one
    /// execution at a time. [`Execution::pump`] then waits for the outcome up to the workload's
    /// own timeout plus `response_grace`, and without a limit when the workload has no timeout.
    /// Fails with [`ErrorKind::Exited`] when the VM is gone, and with
    /// [`ErrorKind::Unsupported`] when `control_timeout` is zero or either duration is longer
    /// than [`MAX_TIMEOUT`](super::launch::MAX_TIMEOUT).
    pub fn exec(
        &self,
        running: &Running,
        workload: &Workload<'_>,
        control_timeout: Duration,
        response_grace: Duration,
    ) -> Result<Execution> {
        check_timeout("control timeout", control_timeout, true)?;
        check_timeout("response grace", response_grace, false)?;
        let deadline = Instant::now() + control_timeout;
        let process = running.vmm.record().process();
        let mut session = session::connect(
            running.vmm.endpoint(),
            process.pid,
            process.start_time,
            deadline,
        )
        .and_then(|transport| ControlSession::attach(transport, &running.capability, deadline))
        .map_err(|error| self.session_error(error, control_timeout))?;
        let request_id = session
            .start_exec(workload, deadline)
            .map_err(|error| self.session_error(error, control_timeout))?;
        let timeout_ms = workload.timeout_ms;
        let response_deadline = (timeout_ms > 0)
            .then(|| Instant::now() + Duration::from_millis(timeout_ms.into()) + response_grace);
        Ok(Execution {
            session,
            request_id,
            deadline: response_deadline,
            control_timeout,
        })
    }
}

/// An instance whose lock this holds, through which its lifecycle transitions change its
/// records. Dropping it releases the lock.
#[derive(Debug)]
pub struct LockedInstance {
    /// The instance.
    instance: Instance,
    /// Its locked store.
    locked: LockedStore,
}

impl LockedInstance {
    /// Returns the instance.
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    /// Finds the instance's VM with [`supervisor::reattach`], under the runtime identity that
    /// `runtime` returns: the one with which the caller would launch it now.
    ///
    /// `runtime` is asked only when the store records a VM or a start, so that an instance that
    /// never started, or that stopped, needs no runtime identity. Reattach ends a start that was
    /// interrupted, signaling only an OpenVMM process that it verified, and forgets a VM that
    /// exited, as one that a record of an earlier host boot names has. Fails as `runtime` fails,
    /// for records that the store does not read, such as those of an earlier format, whose
    /// message names it, and when whether the VM runs cannot be determined, because an
    /// interrupted start cannot be ended yet or the recorded process cannot be examined: the
    /// records then remain for a later attempt.
    pub fn find(&mut self, runtime: impl FnOnce() -> Result<RuntimeIdentity>) -> Result<Found> {
        if self.locked.journal()?.is_none() && self.locked.runtime()?.is_none() {
            return Ok(Found::Stopped);
        }
        let runtime = runtime()?;
        Ok(match supervisor::reattach(&mut self.locked, &runtime)? {
            Reattached::Live(vmm) => Found::Running(Running {
                vmm,
                capability: self.locked.read_capability()?,
            }),
            Reattached::Mismatched { vmm, differences } => Found::Mismatched(Mismatched {
                name: self.instance.name.clone(),
                vmm,
                differences,
            }),
            Reattached::Exited { .. } => Found::Stopped,
            Reattached::Indeterminate(reason) => {
                return Err(Error::other(format!(
                    "cannot determine whether the OpenVMM process of sandbox {} runs: {reason}",
                    self.instance.name
                )));
            }
        })
    }

    /// Launches the VM that `spec` describes under the runtime identity `runtime`, and waits up
    /// to `timeout` until its guest is ready.
    ///
    /// The caller has found the instance stopped with [`find`](Self::find). Start first checks
    /// that the mapped host paths still name the objects they named when they were planned; see
    /// [`ExportPlan::verify`](super::filesystem::ExportPlan::verify). It launches OpenVMM with
    /// [`supervisor::launch`], whose journal records the launch before OpenVMM is spawned and its
    /// process identity right after, so that [`find`](Self::find) ends a start whose caller died.
    /// The capability travels through OpenVMM's standard-input pipe, which is closed before
    /// OpenVMM starts, and never appears in arguments, the environment, or the log. Once the guest
    /// is ready, has the required features, and has mounted the mapped host paths, start commits
    /// the launch: the runtime record names the VM from then on.
    ///
    /// Fails with [`ErrorKind::Unavailable`] when the guest image lacks a feature in
    /// [`GuestFeatures::REQUIRED`], and with [`ErrorKind::Unsupported`] when `timeout` is zero or
    /// longer than [`MAX_TIMEOUT`](super::launch::MAX_TIMEOUT), when the specification fails
    /// [`LaunchSpec::check`] or exceeds a limit, or when OpenVMM's command line would not fit
    /// [`PROCESS_COMMAND_LINE_LIMIT`](super::launch::PROCESS_COMMAND_LINE_LIMIT), which holds on
    /// every host; such a launch records nothing. Unlike [`check_launch`], start keeps no reserve.
    /// A launch that does not become ready is terminated through its verified process identity;
    /// one that cannot be ended keeps its journal, and the next transition ends it.
    pub fn start(
        &mut self,
        spec: &LaunchSpec,
        runtime: &RuntimeIdentity,
        timeout: Duration,
    ) -> Result<Started> {
        check_timeout("start timeout", timeout, true)?;
        if let Some(plan) = &spec.filesystem {
            plan.verify()?;
        }
        let instance = &self.instance;
        let report = instance.report_path();
        // Every control endpoint of the instance has the same length, so a launch that would not
        // fit is refused before the store records it.
        let endpoint = control_endpoint(&instance.store.socket_path())?;
        spec.fit_process_command_line(&spec.openvmm_arguments(&endpoint, &report)?, 0)?;
        let capability = generate_capability()?;
        let started = Instant::now();
        let starting = supervisor::launch(&mut self.locked, runtime, &capability, |endpoint| {
            let arguments = spec.openvmm_arguments(endpoint, &report)?;
            spec.fit_process_command_line(&arguments, 0)?;
            let mut command = VmmCommand::new(&spec.openvmm, arguments, instance.dir());
            command.breakaway_from_job = spec.breakaway_from_job;
            Ok(command)
        })?;
        let (pid, start_time) = {
            let reference = starting.process().reference();
            (reference.pid, reference.start_time)
        };
        let deadline = started + timeout;
        let ready = session::connect(starting.endpoint(), pid, start_time, deadline)
            .and_then(|transport| ControlSession::attach(transport, &capability, deadline))
            .and_then(|mut session| {
                session.ping(deadline)?;
                let features = session.features(deadline)?;
                Ok((session, features))
            });
        let (mut session, features) = match ready {
            Ok(ready) => ready,
            Err(error) => {
                let reason = readiness_failure(&error, spec.identity.create_account);
                return Err(instance.abort_start(starting, session_kind(&error), &reason));
            }
        };
        let missing = features.missing(GuestFeatures::REQUIRED);
        if !missing.is_empty() {
            drop(session);
            return Err(instance.reject_guest(starting, &missing));
        }
        // Workloads run only once the guest has mounted every mapped host path.
        if let Some(plan) = &spec.filesystem
            && let Err(error) = session.map_host_paths(&plan.guest_table(), deadline)
        {
            drop(session);
            let reason = format!("the host paths were not mapped: {error}");
            return Err(instance.abort_start(starting, session_kind(&error), &reason));
        }
        drop(session);
        let vmm = starting.commit()?;
        Ok(Started {
            runtime: vmm.record().clone(),
            boot: started.elapsed(),
        })
    }

    /// Asks the guest of the running VM `running` to shut down, and terminates OpenVMM through
    /// its verified process identity if it has not exited within `timeout`, as [`Vmm::stop`]
    /// does with a forced step.
    ///
    /// A running execution keeps the control session until it ends, so a stop that cannot reach
    /// the guest in time terminates the VM, and the execution then fails. Once OpenVMM has
    /// exited, the stop clears the runtime record and reads OpenVMM's outcome report. Fails with
    /// [`ErrorKind::Unsupported`] when `timeout` is zero or longer than
    /// [`MAX_TIMEOUT`](super::launch::MAX_TIMEOUT), and when OpenVMM cannot be terminated, or does
    /// not exit after it was, which keeps its runtime record.
    pub fn stop(&mut self, running: Running, timeout: Duration) -> Result<Stopped> {
        check_timeout("stop timeout", timeout, true)?;
        let deadline = Instant::now() + timeout;
        let Running { vmm, capability } = running;
        let stopped = vmm.stop(
            &mut self.locked,
            deadline,
            Some(KILL_TIMEOUT),
            |vmm, deadline| {
                let process = vmm.record().process();
                let transport =
                    session::connect(vmm.endpoint(), process.pid, process.start_time, deadline)?;
                ControlSession::attach(transport, &capability, deadline)?.stop(deadline)
            },
        )?;
        let forced = match stopped.outcome {
            StopOutcome::Graceful => false,
            StopOutcome::Forced => true,
            StopOutcome::TimedOut => {
                return Err(Error::other(format!(
                    "OpenVMM process {} of sandbox {} did not exit after it was terminated",
                    vmm.record().process().pid,
                    self.instance.name
                )));
            }
        };
        Ok(Stopped {
            forced,
            report: report::read(&self.instance.report_path()),
        })
    }
}

/// Fails with [`ErrorKind::Unsupported`] when the duration that `name` describes is longer than
/// [`MAX_TIMEOUT`](super::launch::MAX_TIMEOUT), or zero while it must be `positive`, so that no
/// deadline derived from it overflows.
fn check_timeout(name: &str, value: Duration, positive: bool) -> Result<()> {
    match timeout_problem(value, positive) {
        Some(problem) => Err(Error::unsupported(format!("the {name} {problem}"))),
        None => Ok(()),
    }
}

/// Receives a workload's output.
pub trait OutputSink {
    /// Writes one chunk. After an error, the chunks of this stream are discarded, and the
    /// workload continues.
    fn write(&mut self, chunk: &[u8]) -> io::Result<()>;
}

impl<F: FnMut(&[u8]) -> io::Result<()>> OutputSink for F {
    fn write(&mut self, chunk: &[u8]) -> io::Result<()> {
        self(chunk)
    }
}

/// How a workload ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadOutcome {
    /// The workload exited with this code.
    Exited(i32),
    /// A signal with this number ended the workload.
    Signaled(i32),
    /// The workload's timeout elapsed, and the guest agent killed it.
    TimedOut,
    /// The workload exceeded [`MAX_OUTPUT_BYTES`], and the guest agent killed it.
    OutputLimitExceeded,
    /// The guest agent could not supervise the workload.
    Failed,
    /// A cancellation request ended the workload.
    Cancelled,
    /// The guest agent could not launch the workload.
    LaunchFailed,
    /// The workload could not enter its working directory, so nothing of it ran. The guest agent
    /// wrote a diagnostic that names the directory to its standard error.
    WorkingDirectory,
}

/// A workload that [`Instance::exec`] started.
pub struct Execution {
    /// Exclusive control session that carries this workload's traffic.
    session: ControlSession,
    /// Identifier correlating requests, output, cancellation, and the final outcome.
    request_id: u64,
    /// Workload response deadline, absent for a workload without a timeout.
    deadline: Option<Instant>,
    /// Maximum time allowed for control traffic after cancellation or failure.
    control_timeout: Duration,
}

impl Execution {
    /// Relays the workload's output to `stdout` and `stderr` and returns its outcome.
    ///
    /// Events are read in short slices, so that setting `cancel` reaches the guest within 20 ms.
    /// The guest then terminates the workload and reports the cancelled outcome as usual. Once
    /// cancellation is sent, a missing response fails within the control timeout of
    /// [`Instance::exec`], or the earlier workload-response deadline, even when the workload has
    /// no timeout. The guest agent limits the combined output to [`MAX_OUTPUT_BYTES`], and a
    /// guest that sends more fails the execution. If the process that pumps an execution dies,
    /// the guest kills the workload as soon as it notices that the control session is gone.
    pub fn pump(
        mut self,
        stdout: &mut dyn OutputSink,
        stderr: &mut dyn OutputSink,
        cancel: &AtomicBool,
    ) -> Result<WorkloadOutcome> {
        let mut deadline = self.deadline;
        let mut forwarded = 0usize;
        let mut stdout_open = true;
        let mut stderr_open = true;
        let mut cancel_sent = false;
        loop {
            if !cancel_sent && cancel.load(Ordering::Acquire) {
                let cancellation_deadline = Instant::now() + self.control_timeout;
                self.session
                    .cancel_exec(self.request_id, cancellation_deadline)
                    .map_err(|error| {
                        Error::new(
                            session_kind(&error),
                            format!("cannot deliver the cancellation request: {error}"),
                        )
                        .with_source(error)
                    })?;
                deadline = Some(deadline.map_or(cancellation_deadline, |deadline| {
                    deadline.min(cancellation_deadline)
                }));
                cancel_sent = true;
            }
            let slice = Instant::now() + CANCEL_POLL_INTERVAL;
            let poll_deadline = if cancel_sent {
                deadline
            } else {
                Some(deadline.map_or(slice, |deadline| deadline.min(slice)))
            };
            let event = match self.session.next_exec_event(self.request_id, poll_deadline) {
                Err(SessionError::TimedOut)
                    if poll_deadline != deadline
                        && deadline.is_none_or(|deadline| Instant::now() < deadline) =>
                {
                    continue;
                }
                event => event,
            };
            let event = event.map_err(|error| match error {
                SessionError::TimedOut => {
                    Error::timed_out("timed out waiting for the workload outcome")
                }
                SessionError::Closed | SessionError::Reset => Error::protocol(
                    "the control session ended before the workload finished; the sandbox VM may \
                     have stopped",
                ),
                other => Error::new(
                    session_kind(&other),
                    format!("the control session failed: {other}"),
                )
                .with_source(other),
            })?;
            let (chunk, sink, open): (Vec<u8>, &mut dyn OutputSink, &mut bool) = match event {
                ExecEvent::Stdout(chunk) => (chunk, &mut *stdout, &mut stdout_open),
                ExecEvent::Stderr(chunk) => (chunk, &mut *stderr, &mut stderr_open),
                ExecEvent::Exit { category, status } => {
                    return Ok(match category {
                        ExitCategory::Exit => WorkloadOutcome::Exited(status),
                        ExitCategory::Signal => WorkloadOutcome::Signaled(if status > 128 {
                            status - 128
                        } else {
                            status
                        }),
                        ExitCategory::Timeout => WorkloadOutcome::TimedOut,
                        ExitCategory::OutputLimit => WorkloadOutcome::OutputLimitExceeded,
                        ExitCategory::Failed => WorkloadOutcome::Failed,
                        ExitCategory::Cancelled => WorkloadOutcome::Cancelled,
                    });
                }
                ExecEvent::Rejected { status, category } => {
                    return match category.as_str() {
                        LAUNCH_FAILED => Ok(WorkloadOutcome::LaunchFailed),
                        // The guest already wrote a diagnostic that names the directory to stderr.
                        CWD_FAILED => Ok(WorkloadOutcome::WorkingDirectory),
                        // Only a start that was interrupted leaves a guest without its mappings.
                        MAPPINGS_INCOMPLETE => Err(Error::protocol(
                            "the guest has not mounted every mapped host path, because its start \
                             did not complete; stop the sandbox and start it again",
                        )),
                        _ => Err(Error::protocol(format!(
                            "the guest agent rejected the workload: {category} (status {status})"
                        ))),
                    };
                }
            };
            forwarded = forwarded.saturating_add(chunk.len());
            if forwarded > MAX_OUTPUT_BYTES {
                return Err(Error::protocol(
                    "the guest exceeded the output limit of the control protocol",
                ));
            }
            if *open && sink.write(&chunk).is_err() {
                *open = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_failures_explain_images_without_workload_accounts() {
        let hint = "predates workload account creation";
        for ended in [SessionError::ProcessExited, SessionError::Closed] {
            assert!(readiness_failure(&ended, true).contains(hint));
            // Only a sandbox that needs the account suggests it.
            assert!(!readiness_failure(&ended, false).contains(hint));
        }
        for other in [
            SessionError::TimedOut,
            SessionError::Protocol("invalid record".to_owned()),
        ] {
            assert_eq!(readiness_failure(&other, true), other.to_string());
        }
    }

    /// The instance's store, which exists on Linux and Windows hosts.
    #[cfg(any(target_os = "linux", windows))]
    mod store {
        use serde_json::json;

        use super::super::*;
        use crate::host::launch::PROCESS_COMMAND_LINE_LIMIT;
        use crate::host::store::{ArtifactIdentity, JOURNAL_NAME, RUNTIME_NAME};

        fn instance() -> (tempfile::TempDir, Instance) {
            let directory = tempfile::tempdir().unwrap();
            let instance =
                Instance::create(directory.path().join("vm"), "aci-edge-sandboxes:0123").unwrap();
            (directory, instance)
        }

        /// Returns the runtime identity of the launches of these tests.
        fn runtime() -> RuntimeIdentity {
            let artifact = |digit: &str| ArtifactIdentity::sha256(&digit.repeat(64)).unwrap();
            RuntimeIdentity::new(artifact("1"), artifact("2"), artifact("3"))
        }

        #[test]
        fn instances_that_record_no_vm_need_no_runtime_identity() {
            let (_directory, instance) = instance();
            let mut locked = instance.lock(None).unwrap();
            let found = locked
                .find(|| panic!("an instance that records no VM needs no runtime identity"))
                .unwrap();
            assert!(matches!(found, Found::Stopped), "{found:?}");
            drop(locked);
            let reopened = Instance::open(instance.dir(), instance.name()).unwrap();
            assert_eq!(reopened, instance);
        }

        #[test]
        fn records_of_the_earlier_format_fail_naming_it() {
            // Before the store, instances kept runtime records and launch markers of format 3.
            for (name, record) in [
                (
                    RUNTIME_NAME,
                    json!({ "format": 3, "pid": 7, "startTime": 8, "endpoint": "e" }),
                ),
                (
                    JOURNAL_NAME,
                    json!({ "format": 3, "endpoint": "e", "process": { "pid": 7, "startTime": 8 } }),
                ),
            ] {
                let (_directory, instance) = instance();
                let path = instance.dir().join(name);
                fs::write(&path, record.to_string()).unwrap();
                let mut locked = instance.lock(None).unwrap();
                let error = locked.find(|| Ok(runtime())).unwrap_err();
                assert_eq!(error.kind(), ErrorKind::Io);
                assert!(
                    error.message().contains("has the unsupported format 3"),
                    "{error}"
                );
                assert!(
                    error
                        .message()
                        .contains("stop the sandbox with the build that started it"),
                    "{error}"
                );
                // Nothing is migrated or removed.
                assert!(path.is_file());
            }
        }

        #[test]
        fn launches_are_checked_before_the_instance_exists() {
            let directory = tempfile::tempdir().unwrap();
            let mut spec = LaunchSpec::new(
                Artifacts::new("openvmm", "vmlinux", "initramfs.cpio.gz"),
                crate::Hypervisor::Kvm,
            );
            let dir = directory.path().join("vm");
            check_launch(&dir, &spec).unwrap();
            spec.openvmm = directory
                .path()
                .join("o".repeat(PROCESS_COMMAND_LINE_LIMIT));
            let error = check_launch(&dir, &spec).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
            assert!(!dir.exists());
        }
    }
}
