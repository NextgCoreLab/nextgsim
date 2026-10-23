//! Process table for discovering running nextgsim instances
//!
//! Running gNB and UE instances register themselves in a process table
//! directory (`/tmp/nextgsim.proc-table/`). The CLI uses this to discover
//! available nodes and their command ports.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/lib/app/proc_table.cpp` implementation.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

/// Directory where process table entries are stored
pub const PROC_TABLE_DIR: &str = "/tmp/nextgsim.proc-table/";

/// Version information for compatibility checking
pub const VERSION_MAJOR: u8 = 1;
pub const VERSION_MINOR: u8 = 0;
pub const VERSION_PATCH: u8 = 0;

/// Minimum node name length
pub const MIN_NODE_NAME: usize = 3;

/// Maximum node name length
pub const MAX_NODE_NAME: usize = 30;

/// A process table entry: the one format, shared with the writer
/// (`nextgsim_common::CliServer::register_nodes`), so the two cannot drift.
pub use nextgsim_common::ProcTableEntry;

/// Reads every decodable entry, removing the ones whose process is gone.
///
/// "Gone" includes a PID that is running but is a DIFFERENT process than the
/// one that wrote the entry, told apart by its start time (#320): in a
/// restarted container the dead instance and the new one are both PID 1.
fn live_entries(proc_table_dir: &Path) -> Result<Vec<ProcTableEntry>> {
    let mut live = Vec::new();
    let entries = fs::read_dir(proc_table_dir).context("Failed to read proc table directory")?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let Some(table_entry) = ProcTableEntry::decode(&content) else {
            continue;
        };
        if !nextgsim_common::proc_entry_is_live(&table_entry) {
            let _ = fs::remove_file(&path);
            continue;
        }
        live.push(table_entry);
    }
    Ok(live)
}

/// Of the live, version-compatible entries naming `node_name`, the port of the
/// newest (latest start time; an entry without one counts as oldest). Two live
/// processes claiming one name is a misconfiguration, but the newest is the one
/// an operator just started. Also returns how many were skipped for version.
fn select_node(entries: &[ProcTableEntry], node_name: &str) -> (u16, usize) {
    let mut skipped = 0;
    let mut best: Option<&ProcTableEntry> = None;
    for e in entries
        .iter()
        .filter(|e| e.nodes.iter().any(|n| n == node_name))
    {
        if (e.major, e.minor, e.patch) != (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH) {
            skipped += 1;
            continue;
        }
        if best.is_none_or(|b| e.start_time.unwrap_or(0) >= b.start_time.unwrap_or(0)) {
            best = Some(e);
        }
    }
    (best.map_or(0, |e| e.port), skipped)
}

/// Result of node discovery
#[derive(Debug)]
pub struct DiscoveryResult {
    /// Port number if node was found (0 if not found)
    pub port: u16,
    /// Number of nodes skipped due to version mismatch
    pub skipped_due_to_version: usize,
}

/// Discovers a node by name and returns its command port
///
/// This function scans the process table directory for entries that contain
/// the specified node name. It also cleans up stale entries for processes
/// that are no longer running.
///
/// # Arguments
///
/// * `node_name` - The name of the node to discover
///
/// # Returns
///
/// A `DiscoveryResult` containing the port (0 if not found) and count of
/// version-mismatched nodes that were skipped.
pub fn discover_node(node_name: &str) -> Result<DiscoveryResult> {
    let proc_table_dir = Path::new(PROC_TABLE_DIR);

    if !proc_table_dir.exists() {
        return Ok(DiscoveryResult {
            port: 0,
            skipped_due_to_version: 0,
        });
    }

    let entries = live_entries(proc_table_dir)?;
    let (found_port, skipped_due_to_version) = select_node(&entries, node_name);

    Ok(DiscoveryResult {
        port: found_port,
        skipped_due_to_version,
    })
}

/// Lists all available node names from running instances
///
/// This function scans the process table directory and returns all node names
/// from processes that are still running. Stale entries are cleaned up.
pub fn list_nodes() -> Result<Vec<String>> {
    let proc_table_dir = Path::new(PROC_TABLE_DIR);

    if !proc_table_dir.exists() {
        return Ok(Vec::new());
    }

    let mut nodes: Vec<String> = live_entries(proc_table_dir)?
        .into_iter()
        .flat_map(|e| e.nodes)
        .collect();

    // Sort for consistent output
    nodes.sort();
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(port: u16, nodes: &[&str], start_time: Option<u64>) -> ProcTableEntry {
        ProcTableEntry {
            major: VERSION_MAJOR,
            minor: VERSION_MINOR,
            patch: VERSION_PATCH,
            pid: 1,
            port,
            nodes: nodes.iter().map(|s| (*s).to_string()).collect(),
            start_time,
        }
    }

    #[test]
    fn test_proc_table_entry_decode_invalid() {
        assert!(ProcTableEntry::decode("").is_none());
        assert!(ProcTableEntry::decode("1 2 3").is_none());
        assert!(ProcTableEntry::decode("1 2 3 4 5 2 node1").is_none()); // Missing second node
    }

    /// #320: when two live entries name the node, the newest wins; a legacy entry
    /// with no start time loses to one that has it.
    #[test]
    fn the_newest_entry_for_a_node_wins() {
        let entries = vec![
            entry(1000, &["ue0"], Some(50)),
            entry(2000, &["ue0"], Some(900)),
            entry(3000, &["ue0"], None),
            entry(4000, &["gnb"], Some(5000)),
        ];
        assert_eq!(select_node(&entries, "ue0"), (2000, 0));
        assert_eq!(select_node(&entries, "gnb"), (4000, 0));
        assert_eq!(select_node(&entries, "nope"), (0, 0));
        let mut old = entry(5000, &["ue0"], Some(10_000));
        old.major = 0;
        assert_eq!(select_node(&[old], "ue0"), (0, 1), "version-skipped");
    }
}
