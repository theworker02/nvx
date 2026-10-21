//! Persistent per-sandbox state of the OpenVMM backend.
//!
//! ```text
//! <state_root>/
//!   <token>/
//!     sandbox.json            provisioned configuration
//!     sandbox.lock            serializes lifecycle transitions of the sandbox
//!     launch.json             the journal of a start in progress, or of one that was interrupted
//!     runtime.json            the runtime record of the running VM
//!     control.capability      launch capability (running only)
//!     control.sock            Linux control endpoint (running only)
//!     openvmm.log             OpenVMM output of the latest start
//!     outcome.json            OpenVMM outcome report of the latest run
//! ```
//!
//! Each sandbox's directory is the store of its VM [`Instance`], which keeps every file but
//! `sandbox.json`, the configuration that this crate records in a format of its own,
//! [`STATE_FORMAT`].

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use nvx::host::filesystem::ExportPlan;
use nvx::host::instance::{self, Instance, LockedInstance};
use nvx::host::platform;
use nvx::host::store::{FILE_NAMES, LOCK_NAME, LOG_NAME};
use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorCode, Result};
use crate::id::SandboxId;
use crate::model::{HostLoopbackForward, NetworkPolicy};

/// Version of `sandbox.json`, the configuration that this crate records, which holds the
/// sandbox's [`ExportPlan`] in the form of `nvx`.
///
/// Version 1 described sandboxes assembled from image layers and a scratch disk, and version 2
/// exported the mapped host paths as one directory whose bind mounts the kernel command line
/// listed.
pub(crate) const STATE_FORMAT: u32 = 3;
/// Backend key recorded in every sandbox.
pub(crate) const BACKEND_KEY: &str = "openvmm";

const RECORD_NAME: &str = "sandbox.json";

/// Provisioned configuration of one sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SandboxRecord {
    pub(crate) format: u32,
    pub(crate) backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) network: Option<NetworkPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) filesystem: Option<ExportPlan>,
    /// Host loopback ports that OpenVMM forwards to guest ports. A record without forwards keeps
    /// the form of earlier records, and earlier versions of this crate refuse one with forwards.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) host_loopback_forwards: Vec<HostLoopbackForward>,
    pub(crate) memory_mib: u32,
    pub(crate) workload_uid: u32,
    pub(crate) workload_gid: u32,
    /// Whether the guest creates an account for the workload identity if its image lacks a usable
    /// one.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(crate) create_workload_account: bool,
    pub(crate) hostname: String,
}

/// Directory tree that holds every sandbox of one backend instance.
#[derive(Debug)]
pub(crate) struct StateStore {
    root: PathBuf,
}

fn io_error(context: String) -> impl FnOnce(io::Error) -> Error {
    move |error| Error::backend_error(context).with_source(error)
}

fn not_provisioned(sandbox_id: &SandboxId) -> Error {
    Error::stale_id(format!("sandbox {sandbox_id} is not provisioned"))
}

/// Removes the directory `dir` of a sandbox whose configuration is gone, and returns whether it
/// is gone.
///
/// The directory stays while it is not empty: a transition that waited for the sandbox's lock may
/// hold a lock file in it, and a file system without POSIX deletion keeps a removed lock file
/// there until its last handle closes. Each such transition finds no configuration, and removes
/// its lock file and the directory as it releases the lock.
fn remove_emptied(dir: &Path) -> Result<bool> {
    match fs::remove_dir(dir) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::DirectoryNotEmpty => Ok(false),
        Err(error) => Err(io_error(format!(
            "cannot remove sandbox state {}",
            dir.display()
        ))(error)),
    }
}

impl StateStore {
    /// Opens `root`, creating it and restricting it to the current user if needed.
    pub(crate) fn open(root: &Path) -> io::Result<Self> {
        if let Some(parent) = root.parent() {
            fs::create_dir_all(parent)?;
        }
        platform::create_private_dir(root)?;
        Ok(Self {
            root: root.to_path_buf(),
        })
    }

    pub(crate) fn dir(&self, sandbox_id: &SandboxId) -> PathBuf {
        self.root.join(sandbox_id.token())
    }

    /// Returns the path of the OpenVMM log of a sandbox's latest start.
    pub(crate) fn log_path(&self, sandbox_id: &SandboxId) -> PathBuf {
        self.dir(sandbox_id).join(LOG_NAME)
    }

    /// Takes the exclusive lifecycle lock of a sandbox, the lock of its instance's store, and
    /// loads its configuration.
    ///
    /// An ID without configuration fails with [`ErrorCode::StaleId`], also when its state
    /// disappears while this call waits for the lock, as it does when a concurrent deprovision
    /// removes it. A stale ID never becomes valid again, so the call removes the lock file that
    /// it may have created, and the sandbox's directory, such as one that an interrupted removal
    /// left, if that empties it: a stale ID leaves nothing behind.
    pub(crate) fn lock_and_load(
        &self,
        sandbox_id: &SandboxId,
    ) -> Result<(LockedInstance, SandboxRecord)> {
        let dir = self.dir(sandbox_id);
        let locked = match Instance::open(&dir, sandbox_id.to_string())
            .and_then(|instance| instance.lock(None))
        {
            Ok(locked) => locked,
            // The directory or its lock file went away, or never existed, with the sandbox.
            Err(_) if matches!(dir.join(RECORD_NAME).try_exists(), Ok(false)) => {
                let _ = remove_emptied(&dir);
                return Err(not_provisioned(sandbox_id));
            }
            Err(error) => return Err(error.into()),
        };
        match self.load(sandbox_id) {
            Ok(record) => Ok((locked, record)),
            Err(error) if error.code() == ErrorCode::StaleId => {
                let _ = fs::remove_file(dir.join(LOCK_NAME));
                drop(locked);
                let _ = remove_emptied(&dir);
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    /// Creates the state directory of a new sandbox, the store of its instance, private to the
    /// current user, and records its configuration.
    pub(crate) fn create(&self, sandbox_id: &SandboxId, record: &SandboxRecord) -> Result<()> {
        let dir = self.dir(sandbox_id);
        // Sandbox IDs are random, so the directory of a new one never exists.
        if !matches!(dir.try_exists(), Ok(false)) {
            return Err(Error::backend_error(format!(
                "cannot create sandbox state {}: it exists already",
                dir.display()
            )));
        }
        let created = Instance::create(&dir, sandbox_id.to_string())
            .and_then(|_| instance::write_json(&dir.join(RECORD_NAME), record));
        if created.is_err() {
            let _ = fs::remove_dir_all(&dir);
        }
        Ok(created?)
    }

    /// Loads the configuration of a provisioned sandbox.
    pub(crate) fn load(&self, sandbox_id: &SandboxId) -> Result<SandboxRecord> {
        let path = self.dir(sandbox_id).join(RECORD_NAME);
        // Records of other formats have other fields, so their format is checked first.
        let record: serde_json::Value = match instance::read_json(&path)? {
            Some(record) => record,
            None => return Err(not_provisioned(sandbox_id)),
        };
        if record.get("format").and_then(serde_json::Value::as_u64) != Some(STATE_FORMAT.into())
            || record.get("backend").and_then(serde_json::Value::as_str) != Some(BACKEND_KEY)
        {
            return Err(Error::backend_error(format!(
                "sandbox state {} has an unsupported format",
                path.display()
            )));
        }
        serde_json::from_value(record).map_err(|error| {
            Error::backend_error(format!("sandbox state {} is malformed", path.display()))
                .with_source(error)
        })
    }

    /// Deletes the state of a stopped sandbox, whose lock `locked` holds, and releases the lock.
    ///
    /// Unknown files abort the removal before anything is deleted. The configuration goes after
    /// the instance's files, so an interrupted removal leaves a sandbox that can be deprovisioned
    /// again, and the lock file goes last, while it is held, so that a transition that waits for
    /// the lock finds no configuration.
    pub(crate) fn remove(&self, sandbox_id: &SandboxId, locked: LockedInstance) -> Result<()> {
        let dir = self.dir(sandbox_id);
        let entries = fs::read_dir(&dir).map_err(io_error(format!(
            "cannot list sandbox state {}",
            dir.display()
        )))?;
        let mut owned = Vec::new();
        for entry in entries {
            let entry = entry.map_err(io_error(format!(
                "cannot list sandbox state {}",
                dir.display()
            )))?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let temporary = name.starts_with('.') && name.ends_with(".tmp");
            if name == RECORD_NAME || name == LOCK_NAME {
                continue;
            }
            if FILE_NAMES.contains(&name.as_ref()) || temporary {
                owned.push(entry.path());
            } else {
                return Err(Error::backend_error(format!(
                    "sandbox state {} contains the unexpected entry {name:?}; remove it and retry",
                    dir.display()
                )));
            }
        }
        for path in owned {
            instance::remove_if_present(&path)?;
        }
        instance::remove_if_present(&dir.join(RECORD_NAME))?;
        instance::remove_if_present(&dir.join(LOCK_NAME))?;
        // The directory goes while the lock is held where it can, and once it is released where
        // the removed lock file stays until its last handle closes.
        if !remove_emptied(&dir)? {
            drop(locked);
            remove_emptied(&dir)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use nvx::ErrorKind;

    use super::*;
    use crate::ErrorCode;

    fn record() -> SandboxRecord {
        SandboxRecord {
            format: STATE_FORMAT,
            backend: BACKEND_KEY.to_owned(),
            network: None,
            filesystem: None,
            host_loopback_forwards: Vec::new(),
            memory_mib: 256,
            workload_uid: 65534,
            workload_gid: 65534,
            create_workload_account: false,
            hostname: "nvx-sandbox".to_owned(),
        }
    }

    /// Returns the names of the entries of `dir`, sorted.
    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn records_round_trip_and_unknown_ids_are_stale() {
        let root = tempfile::tempdir().unwrap();
        let store = StateStore::open(&root.path().join("sandboxes")).unwrap();
        let id = SandboxId::generate().unwrap();
        assert_eq!(store.load(&id).unwrap_err().code(), ErrorCode::StaleId);
        store.create(&id, &record()).unwrap();
        assert_eq!(store.load(&id).unwrap(), record());
        let (locked, loaded) = store.lock_and_load(&id).unwrap();
        assert_eq!(loaded, record());
        assert!(locked.instance().store().runtime().unwrap().is_none());
        assert_eq!(locked.instance().dir(), store.dir(&id));
        // A sandbox's ID is random, so its directory is new.
        let error = store.create(&id, &record()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::BackendError);
        assert!(error.message().contains("exists already"), "{error}");
    }

    #[test]
    fn the_instance_files_and_the_state_are_removed() {
        let root = tempfile::tempdir().unwrap();
        let store = StateStore::open(root.path()).unwrap();
        let id = SandboxId::generate().unwrap();
        store.create(&id, &record()).unwrap();
        let (locked, _) = store.lock_and_load(&id).unwrap();
        let dir = store.dir(&id);
        // A stopped VM leaves its log and outcome report, and an interrupted replacement of a
        // record its temporary file.
        for name in ["openvmm.log", "outcome.json", ".runtime.json.12345.tmp"] {
            fs::write(dir.join(name), b"left").unwrap();
        }
        assert_eq!(store.log_path(&id), dir.join("openvmm.log"));
        store.remove(&id, locked).unwrap();
        assert!(!dir.exists());
        assert!(entries(root.path()).is_empty());
    }

    #[test]
    fn removal_preserves_unknown_files_and_the_sandbox() {
        let root = tempfile::tempdir().unwrap();
        let store = StateStore::open(root.path()).unwrap();
        let id = SandboxId::generate().unwrap();
        store.create(&id, &record()).unwrap();
        fs::write(store.dir(&id).join("keep.txt"), b"user data").unwrap();
        let (locked, _) = store.lock_and_load(&id).unwrap();
        assert!(store.remove(&id, locked).is_err());
        assert!(store.dir(&id).join("keep.txt").exists());
        assert_eq!(store.load(&id).unwrap(), record());
    }

    #[test]
    fn records_of_other_formats_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let store = StateStore::open(root.path()).unwrap();
        let id = SandboxId::generate().unwrap();
        store.create(&id, &record()).unwrap();
        fs::write(store.dir(&id).join(RECORD_NAME), b"{ truncated").unwrap();
        assert_eq!(store.load(&id).unwrap_err().code(), ErrorCode::BackendError);
        let unsupported = |record: &serde_json::Value| {
            fs::write(
                store.dir(&id).join(RECORD_NAME),
                serde_json::to_vec(record).unwrap(),
            )
            .unwrap();
            let error = store.load(&id).unwrap_err();
            assert_eq!(error.code(), ErrorCode::BackendError);
            assert!(error.message().contains("unsupported format"), "{error}");
        };
        let mut legacy = serde_json::to_value(record()).unwrap();
        legacy["format"] = 1.into();
        unsupported(&legacy);
        // Version 2 exported one directory and listed its bind mounts on the kernel command
        // line.
        legacy["format"] = 2.into();
        legacy["filesystem"] = serde_json::json!({
            "root": "/host/work",
            "writable": false,
            "denied": [],
            "deniedIdentities": [],
            "binds": [{ "source": "", "target": "/host/work", "readOnly": true }],
        });
        unsupported(&legacy);
    }

    #[test]
    fn records_of_the_current_format_keep_loading() {
        // A record that the crate wrote before NVX moved to `nvx-rs`, with a file and a
        // directory child, pinned identities, and every optional field.
        let root = tempfile::tempdir().unwrap();
        let store = StateStore::open(root.path()).unwrap();
        let id = SandboxId::generate().unwrap();
        store.create(&id, &record()).unwrap();
        let written = serde_json::json!({
            "format": 3,
            "backend": "openvmm",
            "network": {
                "egress": {
                    "default": "deny",
                    "allow": [{ "to": [{ "cidr": "10.0.0.0/8", "except": ["10.1.0.0/16"] }],
                                "ports": [{ "protocol": "tcp", "port": 443 }] }]
                },
                "ingress": { "default": "deny", "hostLoopback": "deny" }
            },
            "filesystem": {
                "children": [
                    { "root": "/host/work/config.json", "file": true, "writable": false,
                      "hidden": false, "denied": [], "deniedIdentities": [], "allowed": [],
                      "write": [] },
                    { "root": "/host/work/out", "writable": true, "hidden": false,
                      "denied": ["/host/work/out/secret"],
                      "deniedIdentities": [{ "device": 1, "index": 2 }], "allowed": [],
                      "write": [] }
                ],
                "binds": [
                    { "child": 0, "source": "", "target": "/host/work/config.json",
                      "readOnly": true },
                    { "child": 1, "source": "", "target": "/host/work/out", "readOnly": false,
                      "identity": { "device": 1, "index": 3 } }
                ]
            },
            "memoryMib": 512,
            "workloadUid": 1000,
            "workloadGid": 1000,
            "createWorkloadAccount": true,
            "hostname": "nvx-sandbox"
        });
        fs::write(
            store.dir(&id).join(RECORD_NAME),
            serde_json::to_vec(&written).unwrap(),
        )
        .unwrap();
        let loaded = store.load(&id).unwrap();
        let plan = loaded.filesystem.as_ref().unwrap();
        assert!(plan.children[0].file && plan.children[1].writable);
        assert_eq!(plan.binds[1].identity.unwrap().index, 3);
        assert!(loaded.create_workload_account);
        // Writing the record again reproduces it field for field.
        assert_eq!(serde_json::to_value(&loaded).unwrap(), written);
    }

    #[test]
    fn locks_are_exclusive_per_sandbox() {
        let root = tempfile::tempdir().unwrap();
        let store = StateStore::open(root.path()).unwrap();
        let id = SandboxId::generate().unwrap();
        store.create(&id, &record()).unwrap();
        let (locked, loaded) = store.lock_and_load(&id).unwrap();
        assert_eq!(loaded, record());
        // The lock is the lock of the instance's store.
        let contender = Instance::open(store.dir(&id), id.to_string()).unwrap();
        let error = contender.lock(Some(Instant::now())).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::TimedOut, "{error}");
        drop(locked);
        drop(contender.lock(Some(Instant::now())).unwrap());
        assert!(store.dir(&id).join(LOCK_NAME).is_file());
    }

    #[test]
    fn stale_ids_leave_nothing_behind() {
        let root = tempfile::tempdir().unwrap();
        let store = StateStore::open(root.path()).unwrap();
        let never_provisioned = SandboxId::generate().unwrap();
        // An interrupted removal can leave an empty directory, or one with only its lock file,
        // which are stale too.
        let emptied = SandboxId::generate().unwrap();
        fs::create_dir(store.dir(&emptied)).unwrap();
        let locked_only = SandboxId::generate().unwrap();
        Instance::create(store.dir(&locked_only), locked_only.to_string()).unwrap();
        fs::write(store.dir(&locked_only).join(LOCK_NAME), b"").unwrap();
        for id in [&never_provisioned, &emptied, &locked_only] {
            let error = store.lock_and_load(id).unwrap_err();
            assert_eq!(error.code(), ErrorCode::StaleId, "{error}");
        }
        assert!(entries(root.path()).is_empty());

        let id = SandboxId::generate().unwrap();
        store.create(&id, &record()).unwrap();
        drop(store.lock_and_load(&id).unwrap());
        assert_eq!(entries(&store.dir(&id)), [RECORD_NAME, LOCK_NAME]);
    }

    #[test]
    fn state_removed_while_waiting_for_the_lock_leaves_nothing_behind() {
        let root = tempfile::tempdir().unwrap();
        let store = Arc::new(StateStore::open(root.path()).unwrap());
        let id = SandboxId::generate().unwrap();
        store.create(&id, &record()).unwrap();
        let (locked, _) = store.lock_and_load(&id).unwrap();
        let waiter = thread::spawn({
            let (store, id) = (Arc::clone(&store), id.clone());
            move || store.lock_and_load(&id).map(drop)
        });
        // Let the waiter block on the lock, then deprovision as the backend does. A waiter that
        // starts late finds no state and must leave nothing behind either.
        thread::sleep(Duration::from_millis(100));
        store.remove(&id, locked).unwrap();
        let error = waiter.join().unwrap().unwrap_err();
        assert_eq!(error.code(), ErrorCode::StaleId, "{error}");
        assert!(entries(root.path()).is_empty());
    }
}
