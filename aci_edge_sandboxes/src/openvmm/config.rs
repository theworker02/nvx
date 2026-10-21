use std::path::{Path, PathBuf};
use std::time::Duration;

use aci_edge_sandboxes_model::spec::{is_valid_guest_network, is_valid_hostname};
use nvx::Hypervisor;
use nvx::host::artifacts::{Artifacts, absolute};
use nvx::host::launch::{self, WorkloadIdentity};
use nvx::host::store::{RuntimeIdentity, SOCKET_NAME};

use crate::error::{Error, Result};

/// Configuration of the [`OpenVmmBackend`](super::OpenVmmBackend).
///
/// Construct it with [`OpenVmmConfig::discover`], [`OpenVmmConfig::from_artifacts`],
/// [`OpenVmmConfig::new`], [`OpenVmmConfig::from_release_dir`], or
/// [`OpenVmmConfig::from_repo_layout`], then adjust the public fields. Defaults match
/// `scripts/nvx.py sandbox`. Artifacts named by path, as by [`OpenVmmConfig::new`], also need
/// their [`runtime_identity`](Self::runtime_identity).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OpenVmmConfig {
    /// The `openvmm` executable.
    pub openvmm: PathBuf,
    /// The NVX guest kernel (`vmlinux`).
    pub kernel: PathBuf,
    /// The NVX Alpine initramfs (`initramfs.cpio.gz`): the guest userland and its control agent.
    pub initrd: PathBuf,
    /// The runtime identity of the artifacts, which the store records with each sandbox's VM:
    /// the identities of OpenVMM, the kernel, and the initramfs, with the contract versions of
    /// this build. A VM that runs under another runtime identity, as after an upgrade, is reported
    /// and neither driven nor stopped.
    ///
    /// Leave it unset for artifacts located in an NVX release or repository checkout, as by
    /// [`from_release_dir`](Self::from_release_dir), [`from_repo_layout`](Self::from_repo_layout),
    /// [`discover`](Self::discover), or [`from_artifacts`](Self::from_artifacts): every operation
    /// that needs it then reads it from the install metadata of the release or checkout, a
    /// release's `SHA256SUMS` or a checkout's build provenance, without hashing an artifact, so an
    /// install upgraded in place is identified anew. Artifacts named by path have no install
    /// metadata: without it, the backend is unavailable.
    pub runtime_identity: Option<RuntimeIdentity>,
    /// Hypervisor used by OpenVMM.
    pub hypervisor: Hypervisor,
    /// Directory that holds per-sandbox state. Choose a directory private to the current user.
    pub state_root: PathBuf,
    /// Default guest memory in MiB. Provision requests may override it.
    pub memory_mib: u32,
    /// Non-root user ID of workloads. The account must exist in the guest's `/etc/passwd`.
    pub workload_uid: u32,
    /// Primary group ID of workloads.
    pub workload_gid: u32,
    /// Linux hosts only: run the workloads of a sandbox that maps host paths under the calling
    /// user's own IDs, so host file permissions apply to them as they do to that user.
    ///
    /// virtio-fs shows host owners and modes unchanged on Linux, so [`Self::workload_uid`] could
    /// otherwise not read private files or write to the user's directories. The guest creates an
    /// account for the IDs if its image lacks a usable one, replacing an image account for the
    /// UID whose primary group or home does not fit. Root callers keep the configured identity.
    /// Windows hosts present mapped files as accessible to every guest user and ignore this.
    pub map_host_identity: bool,
    /// UTS hostname of workloads.
    pub hostname: String,
    /// Static guest IPv4 address and prefix used when a network device is attached.
    ///
    /// OpenVMM derives the guest's IPv6 address by embedding this one in `fd00::/96`: the default
    /// `10.0.0.2/24` yields `fd00::a00:2/120`, whose gateway is `fd00::a00:1`.
    pub guest_network: String,
    /// Extra kernel parameters. `nvx_*`, `tsc=`, and `hostname=` tokens are reserved.
    ///
    /// They share the guest's 1024-byte kernel command line with the `hostname=` token and the
    /// bind mounts of mapped host paths. A configuration whose parameters and hostname alone do
    /// not fit is rejected.
    pub kernel_command_line: String,
    /// Time allowed for the guest agent to become ready after `start`.
    pub start_timeout: Duration,
    /// Time allowed to connect to a sandbox's control endpoint and receive a cancellation
    /// outcome.
    pub control_timeout: Duration,
    /// Time allowed for a graceful stop before the VM is terminated.
    pub stop_timeout: Duration,
    /// Extra time to wait for an outcome after a workload's own timeout elapses.
    pub exec_response_grace: Duration,
    /// Windows only: start OpenVMM outside the caller's job object so the sandbox VM survives
    /// the caller. The job must permit breakaway.
    pub breakaway_from_job: bool,
    /// Test hook: skips the hypervisor check so tests can drive a fake `openvmm` executable on
    /// hosts without a hypervisor. Available only with the `testing` feature.
    #[cfg(feature = "testing")]
    pub skip_hypervisor_probe: bool,
    /// The artifacts as [`from_artifacts`](Self::from_artifacts) received them, with the release
    /// or checkout in which they were located, whose install metadata identifies them.
    located: Option<Artifacts>,
}

impl OpenVmmConfig {
    /// Default guest memory in MiB.
    pub const DEFAULT_MEMORY_MIB: u32 = launch::DEFAULT_MEMORY_MIB;
    /// Default workload user and group ID (Alpine's `nobody`).
    pub const DEFAULT_WORKLOAD_ID: u32 = WorkloadIdentity::DEFAULT.uid;
    /// Default workload hostname.
    pub const DEFAULT_HOSTNAME: &'static str = launch::DEFAULT_HOSTNAME;
    /// Default guest network address.
    pub const DEFAULT_GUEST_NETWORK: &'static str = launch::DEFAULT_GUEST_NETWORK;
    /// Longest accepted `start_timeout`, `control_timeout`, `stop_timeout`, or
    /// `exec_response_grace`: 30 days.
    ///
    /// The bound keeps every deadline derived from these timeouts, including a workload's own
    /// timeout plus `exec_response_grace`, representable as an [`Instant`](std::time::Instant) and
    /// within the roughly 49-day range of Windows' millisecond waits.
    pub const MAX_TIMEOUT: Duration = launch::MAX_TIMEOUT;

    /// Creates a configuration from explicit artifact paths.
    pub fn new(
        openvmm: impl Into<PathBuf>,
        kernel: impl Into<PathBuf>,
        initrd: impl Into<PathBuf>,
        hypervisor: Hypervisor,
        state_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            openvmm: openvmm.into(),
            kernel: kernel.into(),
            initrd: initrd.into(),
            runtime_identity: None,
            hypervisor,
            state_root: state_root.into(),
            memory_mib: Self::DEFAULT_MEMORY_MIB,
            workload_uid: Self::DEFAULT_WORKLOAD_ID,
            workload_gid: Self::DEFAULT_WORKLOAD_ID,
            map_host_identity: true,
            hostname: Self::DEFAULT_HOSTNAME.to_owned(),
            guest_network: Self::DEFAULT_GUEST_NETWORK.to_owned(),
            kernel_command_line: String::new(),
            start_timeout: Duration::from_secs(60),
            control_timeout: Duration::from_secs(60),
            stop_timeout: Duration::from_secs(30),
            exec_response_grace: Duration::from_secs(30),
            breakaway_from_job: false,
            #[cfg(feature = "testing")]
            skip_hypervisor_probe: false,
            located: None,
        }
    }

    /// Creates a configuration from [`Artifacts`], whose release or checkout, if they were located
    /// in one, identifies them; see [`runtime_identity`](Self::runtime_identity).
    pub fn from_artifacts(
        artifacts: Artifacts,
        hypervisor: Hypervisor,
        state_root: impl Into<PathBuf>,
    ) -> Self {
        let mut config = Self::new(
            artifacts.openvmm.clone(),
            artifacts.kernel.clone(),
            artifacts.initrd.clone(),
            hypervisor,
            state_root,
        );
        config.located = Some(artifacts);
        config
    }

    /// Returns the runtime identity of the configured artifacts:
    /// [`runtime_identity`](Self::runtime_identity) if it is set, and otherwise the one that the
    /// install metadata of the release or checkout in which they were located names, read now.
    ///
    /// Fails with [`nvx::ErrorKind::Unavailable`] for artifacts named by path without one, and
    /// when the metadata is missing or unreadable, or does not name the configured paths.
    pub(crate) fn identify(&self) -> nvx::Result<RuntimeIdentity> {
        if let Some(identity) = &self.runtime_identity {
            return Ok(identity.clone());
        }
        let Some(mut artifacts) = self
            .located
            .clone()
            .filter(|artifacts| artifacts.artifact_dir().is_some())
        else {
            return Err(nvx::Error::unavailable(
                "the artifacts are named by path, so no install metadata identifies them: set \
                 OpenVmmConfig::runtime_identity, or locate them in an NVX release or repository \
                 checkout",
            ));
        };
        artifacts.openvmm.clone_from(&self.openvmm);
        artifacts.kernel.clone_from(&self.kernel);
        artifacts.initrd.clone_from(&self.initrd);
        artifacts.runtime_identity()
    }

    /// Creates a configuration with no NVX-specific input: the artifacts come from
    /// [`Artifacts::discover`], the hypervisor from [`Hypervisor::from_env_or_default`], and the
    /// state root from [`OpenVmmConfig::default_state_root`].
    pub fn discover() -> Result<Self> {
        Ok(Self::from_artifacts(
            Artifacts::discover()?,
            Hypervisor::from_env_or_default()?,
            Self::default_state_root()?,
        ))
    }

    /// Uses the artifacts of an extracted NVX release archive (`bin/` and `guest/`).
    ///
    /// Fails with [`ErrorCode::BackendUnavailable`](crate::ErrorCode::BackendUnavailable) when
    /// the release's `SOURCE-MANIFEST.json` declares an incompatible control contract.
    pub fn from_release_dir(
        release_dir: impl AsRef<Path>,
        hypervisor: Hypervisor,
        state_root: impl Into<PathBuf>,
    ) -> Result<Self> {
        Ok(Self::from_artifacts(
            Artifacts::from_release_dir(release_dir)?,
            hypervisor,
            state_root,
        ))
    }

    /// Uses the artifacts of an NVX repository checkout after `nvx.py download` or a local build.
    pub fn from_repo_layout(
        repo_root: impl AsRef<Path>,
        hypervisor: Hypervisor,
        state_root: impl Into<PathBuf>,
    ) -> Result<Self> {
        Ok(Self::from_artifacts(
            Artifacts::from_repo_layout(repo_root)?,
            hypervisor,
            state_root,
        ))
    }

    /// Returns the conventional per-user state root.
    ///
    /// This is `%LOCALAPPDATA%\nvx\sandboxes` on Windows and
    /// `$XDG_STATE_HOME/nvx/sandboxes` (default `~/.local/state/nvx/sandboxes`) elsewhere.
    pub fn default_state_root() -> Result<PathBuf> {
        let base = if cfg!(windows) {
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("XDG_STATE_HOME")
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .or_else(|| {
                    std::env::var_os("HOME")
                        .map(|home| PathBuf::from(home).join(".local").join("state"))
                })
        };
        base.filter(|path| path.is_absolute())
            .map(|path| path.join("nvx").join("sandboxes"))
            .ok_or_else(|| {
                Error::backend_unavailable("cannot determine the per-user NVX state directory")
            })
    }

    /// Resolves relative paths and checks every field.
    pub(crate) fn normalized(mut self) -> Result<Self> {
        self.openvmm = absolute(&self.openvmm)?;
        self.kernel = absolute(&self.kernel)?;
        self.initrd = absolute(&self.initrd)?;
        self.state_root = absolute(&self.state_root)?;
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<()> {
        if !self.hypervisor.supported_on_host() {
            return Err(Error::backend_unavailable(format!(
                "the {} hypervisor is not available on this host",
                self.hypervisor
            )));
        }
        let invalid = |message: String| Err(Error::backend_unavailable(message));
        if self.memory_mib == 0 {
            return invalid("memory_mib must be positive".to_owned());
        }
        if self.workload_uid == 0 || self.workload_gid == 0 {
            return invalid("workloads must run as a non-root UID and GID".to_owned());
        }
        if !is_valid_hostname(&self.hostname) {
            return invalid(format!(
                "hostname {:?} must be a lowercase RFC 1123 label of up to 63 characters",
                self.hostname
            ));
        }
        if !is_valid_guest_network(&self.guest_network) {
            return invalid(format!(
                "guest_network {:?} must be an IPv4 address with a /1 to /30 prefix that is not \
                 its network's network, broadcast, or gateway (first) address",
                self.guest_network
            ));
        }
        // Provisioning checks the tokens that each sandbox adds.
        if let Some(problem) =
            launch::kernel_parameters_problem(&self.kernel_command_line, &self.hostname)
        {
            return invalid(format!("kernel_command_line {problem}"));
        }
        for (name, value, positive) in [
            ("start_timeout", self.start_timeout, true),
            ("control_timeout", self.control_timeout, true),
            ("stop_timeout", self.stop_timeout, true),
            ("exec_response_grace", self.exec_response_grace, false),
        ] {
            if let Some(problem) = launch::timeout_problem(value, positive) {
                return invalid(format!("{name} {problem}"));
            }
        }
        if cfg!(unix) {
            // sockaddr_un holds 108 bytes including the terminating NUL.
            let socket = self.state_root.join("0".repeat(32)).join(SOCKET_NAME);
            if socket.as_os_str().len() >= 108 {
                return invalid(format!(
                    "state_root {} is too long for a Unix control socket path",
                    self.state_root.display()
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorCode;

    #[test]
    fn validation_rejects_bad_settings() {
        let Some(hypervisor) = Hypervisor::platform_default() else {
            return;
        };
        let directory = tempfile::tempdir().unwrap();
        let base = OpenVmmConfig::new("openvmm", "vmlinux", "initrd", hypervisor, directory.path());
        base.clone().normalized().unwrap();
        let mut cases = Vec::new();
        let mut config = base.clone();
        config.memory_mib = 0;
        cases.push(config);
        let mut config = base.clone();
        config.workload_uid = 0;
        cases.push(config);
        let mut config = base.clone();
        config.hostname = "Bad_Host".to_owned();
        cases.push(config);
        let mut config = base.clone();
        config.guest_network = "10.0.0.2/31".to_owned();
        cases.push(config);
        for non_canonical in ["10.0.0.2/024", "10.0.0.2/+24"] {
            let mut config = base.clone();
            config.guest_network = non_canonical.to_owned();
            cases.push(config);
        }
        let mut config = base.clone();
        config.kernel_command_line = "quiet nvx_exec=/bin/sh".to_owned();
        cases.push(config);
        // The parameters, a space, the hostname token, and a NUL fill the command line exactly.
        let room =
            launch::COMMAND_LINE_BUDGET - launch::base_command_line("", &base.hostname).len() - 2;
        let mut fitting = base.clone();
        fitting.kernel_command_line = "x".repeat(room);
        fitting.normalized().unwrap();
        let mut config = base.clone();
        config.kernel_command_line = "x".repeat(room + 1);
        cases.push(config);
        let mut config = base.clone();
        config.hypervisor = if hypervisor == Hypervisor::Whp {
            Hypervisor::Kvm
        } else {
            Hypervisor::Whp
        };
        cases.push(config);
        let timeouts: [fn(&mut OpenVmmConfig, Duration); 4] = [
            |config, value| config.start_timeout = value,
            |config, value| config.control_timeout = value,
            |config, value| config.stop_timeout = value,
            |config, value| config.exec_response_grace = value,
        ];
        let mut longest = base.clone();
        for set in timeouts {
            for value in [
                OpenVmmConfig::MAX_TIMEOUT + Duration::from_nanos(1),
                Duration::MAX,
            ] {
                let mut config = base.clone();
                set(&mut config, value);
                cases.push(config);
            }
            set(&mut longest, OpenVmmConfig::MAX_TIMEOUT);
        }
        longest.normalized().unwrap();
        for config in cases {
            assert_eq!(
                config.clone().normalized().unwrap_err().code(),
                ErrorCode::BackendUnavailable,
                "{config:?}"
            );
        }
        // The messages name the configuration's fields.
        let mut config = base.clone();
        config.kernel_command_line = "x".repeat(room + 1);
        assert_eq!(
            config.normalized().unwrap_err().message(),
            "kernel_command_line and the hostname leave no room in the 1024-byte kernel command \
             line"
        );
        let mut config = base.clone();
        config.stop_timeout = Duration::ZERO;
        assert_eq!(
            config.normalized().unwrap_err().message(),
            "stop_timeout must be positive"
        );
        let mut config = base;
        config.control_timeout = Duration::MAX;
        assert_eq!(
            config.normalized().unwrap_err().message(),
            "control_timeout must be at most 30 days"
        );
    }

    #[test]
    fn helpers_accept_expected_values() {
        assert!(is_valid_hostname("nvx-sandbox"));
        assert!(!is_valid_hostname("-nvx"));
        assert!(!is_valid_hostname(&"a".repeat(64)));
        assert!(is_valid_guest_network("10.0.0.2/24"));
        assert!(is_valid_guest_network("10.0.0.254/24"));
        assert!(!is_valid_guest_network("10.0.0.2"));
        assert!(!is_valid_guest_network("10.0.0.256/24"));
        for prefix in ["", "0", "024", "+24", "-24", " 24"] {
            assert!(
                !is_valid_guest_network(&format!("10.0.0.2/{prefix}")),
                "{prefix:?}"
            );
        }
        // OpenVMM rejects the network, broadcast, and gateway addresses as guest addresses.
        for reserved in ["10.0.0.0/24", "10.0.0.255/24", "10.0.0.1/24", "10.0.0.5/30"] {
            assert!(!is_valid_guest_network(reserved), "{reserved}");
        }
        assert!(is_valid_guest_network("10.0.0.6/30"));
        assert_eq!("mshv".parse::<Hypervisor>().unwrap(), Hypervisor::Mshv);
    }
}
