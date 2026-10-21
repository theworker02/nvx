//! The host side of NVX.
//!
//! - [`artifacts`] locates the OpenVMM executable and the guest kernel and initramfs, and every
//!   artifact of the guests that `nvx.py` runs, in a repository checkout or a release.
//! - [`launch`] builds OpenVMM's command line from a [`LaunchSpec`](launch::LaunchSpec): the
//!   managed lifecycle, the workload identity, the kernel command line, and the network and
//!   filesystem options.
//! - [`network`] lowers egress rules, and the egress policy files of `nvx.py`, into OpenVMM's
//!   rules, and [`filesystem`] plans the export of host paths through OpenVMM's aggregate
//!   virtio-fs device.
//! - [`instance`] drives one VM instance in a directory of its own, the [`store`] of its VM,
//!   through the [`supervisor`] and the [`session`]: it starts OpenVMM and waits until the guest
//!   is ready, reattaches to the VM after a restart, runs workloads, and stops the VM.
//!   [`report`] reads OpenVMM's outcome report.
//! - [`store`] keeps the runtime records of a sandbox in a private directory with one lock: the
//!   runtime record of its VM, with the VM's runtime identity, and the journal of its launch,
//!   through which an interrupted launch is recovered without signaling an unverified process.
//! - [`supervisor`] launches a sandbox's VMM through the store's journal, passing it only the
//!   descriptors or handles that the caller allows, stops it through the caller's graceful step,
//!   with a bounded forced step on request, and reattaches to it after the orchestrator restarts
//!   within the same host boot.
//! - [`session`] speaks NVX's control protocol, `NVXC`, over [`nvxs`], which runs the host's role
//!   of the NVXS session for it and for any other application protocol, over a platform
//!   transport. [`platform`] holds the primitives of Linux and Windows hosts. Other hosts compile
//!   but report NVX as unavailable.
//! - [`process`] names processes by verified references: their ID, start time, and host boot,
//!   through which a pidfd or a process handle holds them, so that waits sleep until a process
//!   exits and termination reaches only it, whether or not the caller started it.
//! - [`sandbox`] is the launch profile of `nvx.py sandbox`: a container over read-only EROFS
//!   layers and a private ext4 scratch disk, with live shares, and the checks of its options.
//! - `cli` (feature `cli`) is the `nvx-rs` command-line interface, which `nvx.py` runs for the
//!   steps that it delegates: artifact lookup, egress policy compilation, and sandboxes.
//!
//! The ports of `nvx.py` code keep its Python semantics, which the private `python` module
//! provides.
//!
//! The host modules deny unsafe code. On Unix, the private `sys` module confines the system calls
//! that the standard library lacks behind documented safe functions, which call `nix`. The
//! Windows platform module, and the module of the Win32 calls that Python's filesystem queries
//! make, still call `windows-sys` directly.
#![deny(unsafe_code)]

pub mod artifacts;
#[cfg(feature = "cli")]
pub mod cli;
pub mod filesystem;
pub mod instance;
pub mod launch;
pub mod network;
pub mod nvxs;
pub mod platform;
pub mod process;
pub(crate) mod python;
pub mod report;
pub mod sandbox;
pub mod session;
pub mod store;
pub mod supervisor;
#[cfg(unix)]
#[allow(unsafe_code)]
mod sys;
#[cfg(test)]
mod test_support;
