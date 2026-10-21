//! The backend must report missing runtime dependencies as `backend_unavailable` instead of
//! failing later or hanging. This runs on hosts without a hypervisor or NVX artifacts.
#![cfg(feature = "openvmm")]

use std::fs;

use aci_edge_sandboxes::nvx::Hypervisor;
use aci_edge_sandboxes::openvmm::OpenVmmConfig;
use aci_edge_sandboxes::{AciEdgeSandbox, ErrorCode, ProvisionRequest};

fn hypervisor() -> Hypervisor {
    Hypervisor::platform_default().unwrap_or(Hypervisor::Kvm)
}

#[test]
fn missing_openvmm_is_reported_as_backend_unavailable() {
    let directory = tempfile::tempdir().unwrap();
    let file = |name: &str| {
        let path = directory.path().join(name);
        fs::write(&path, b"").unwrap();
        path
    };
    let config = OpenVmmConfig::new(
        directory.path().join("missing-openvmm"),
        file("vmlinux"),
        file("initramfs.cpio.gz"),
        hypervisor(),
        directory.path().join("state"),
    );
    let nvx = match AciEdgeSandbox::openvmm(config) {
        Ok(nvx) => nvx,
        // Hosts that cannot run the backend at all reject the configuration up front.
        Err(error) => {
            assert_eq!(error.code(), ErrorCode::BackendUnavailable, "{error}");
            return;
        }
    };
    assert_eq!(
        nvx.probe().unwrap_err().code(),
        ErrorCode::BackendUnavailable
    );
    assert_eq!(
        nvx.provision(&ProvisionRequest::new()).unwrap_err().code(),
        ErrorCode::BackendUnavailable
    );
}

#[test]
fn incompatible_releases_are_reported_as_backend_unavailable() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("SOURCE-MANIFEST.json"),
        r#"{"openvmm":{"microvm_abi_version":1}}"#,
    )
    .unwrap();
    let error = OpenVmmConfig::from_release_dir(directory.path(), hypervisor(), directory.path())
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::BackendUnavailable);
}

#[test]
fn artifacts_named_by_path_need_a_runtime_identity() {
    let directory = tempfile::tempdir().unwrap();
    let file = |name: &str| {
        let path = directory.path().join(name);
        fs::write(&path, b"").unwrap();
        path
    };
    let config = OpenVmmConfig::new(
        file("openvmm"),
        file("vmlinux"),
        file("initramfs.cpio.gz"),
        hypervisor(),
        directory.path().join("state"),
    );
    assert_eq!(config.runtime_identity, None);
    let Ok(nvx) = AciEdgeSandbox::openvmm(config) else {
        // Hosts that cannot run the backend at all reject the configuration up front.
        return;
    };
    // The files exist, but no install metadata names their identity, which the store records
    // with each VM.
    for error in [
        nvx.probe().unwrap_err(),
        nvx.provision(&ProvisionRequest::new()).unwrap_err(),
    ] {
        assert_eq!(error.code(), ErrorCode::BackendUnavailable, "{error}");
        assert!(
            error
                .message()
                .contains("set OpenVmmConfig::runtime_identity"),
            "{error}"
        );
    }
}
