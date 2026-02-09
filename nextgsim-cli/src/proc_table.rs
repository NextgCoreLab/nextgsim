//! Process table for discovering running nextgsim instances
//!
//! Running gNB and UE instances register themselves in a process table
//! directory (`/tmp/nextgsim.proc-table/`). The CLI uses this to discover
//! available nodes and their command ports.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/lib/app/proc_table.cpp` implementation.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Directory where process table entries are stored
pub const PROC_TABLE_DIR: &str = "/tmp/nextgsim.proc-table/";

/// Directory where process information is stored (Linux /proc)
pub const PROCESS_DIR: &str = "/proc/";

/// Version information for compatibility checking
pub const VERSION_MAJOR: u8 = 1;
pub const VERSION_MINOR: u8 = 0;
pub const VERSION_PATCH: u8 = 0;

/// Minimum node name length
pub const MIN_NODE_NAME: usize = 3;

/// Maximum node name length
pub const MAX_NODE_NAME: usize = 30;

/// A process table entry representing a running nextgsim instance
#[derive(Debug, Clone)]
pub struct ProcTableEntry {
    /// Major version number
    pub major: u8,
    /// Minor version number
    pub minor: u8,
    /// Patch version number
    pub patch: u8,
    /// Process ID
    pub pid: u32,
    /// Command port for CLI communication
    pub port: u16,
    /// Node names registered by this process
    pub nodes: Vec<String>,
}

impl ProcTableEntry {
    /// Encodes a process table entry to a string
    #[allow(dead_code)]
    pub fn encode(&self) -> String {
        let mut s = format!(
            "{} {} {} {} {} {}",
            self.major,
            self.minor,
            self.patch,
            self.pid,
            self.port,
            self.nodes.len()
        );
        for node in &self.nodes {
            s.push(' ');
            s.push_str(node);
        }
        s
    }

    /// Decodes a process table entry from a string
    pub fn decode(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() < 6 {
            anyhow::bail!("Invalid proc table entry: too few fields");
        }

        let major: u8 = parts[0].parse().context("Invalid major version")?;
        let minor: u8 = parts[1].parse().context("Invalid minor version")?;
        let patch: u8 = parts[2].parse().context("Invalid patch version")?;
        let pid: u32 = parts[3].parse().context("Invalid PID")?;
        let port: u16 = parts[4].parse().context("Invalid port")?;
        let node_count: usize = parts[5].parse().context("Invalid node count")?;

        if parts.len() < 6 + node_count {
            anyhow::bail!("Invalid proc table entry: missing node names");
        }

        let nodes: Vec<String> = parts[6..6 + node_count]
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        Ok(Self {
            major,
            minor,
            patch,
            pid,
            port,
            nodes,
        })
    }
}

/// Finds all running process IDs by scanning /proc directory
fn find_running_processes() -> HashSet<u32> {
    let mut pids = HashSet::new();

    if let Ok(entries) = fs::read_dir(PROCESS_DIR) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if let Ok(pid) = name.parse::<u32>() {
                    pids.insert(pid);
                }
            }
        }
    }

    pids
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

    // Find all running processes
    let running_processes = find_running_processes();

    let mut found_port: u16 = 0;
    let mut skipped_due_to_version: usize = 0;

    // Read and parse all proc table entries
    let entries = fs::read_dir(proc_table_dir).context("Failed to read proc table directory")?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let table_entry = match ProcTableEntry::decode(&content) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // If process is no longer running, remove the stale entry
        if !running_processes.contains(&table_entry.pid) {
            let _ = fs::remove_file(&path);
            continue;
        }

        // Check if this entry contains the node we're looking for
        for node in &table_entry.nodes {
            if node == node_name {
                // Check version compatibility
                if table_entry.major == VERSION_MAJOR
                    && table_entry.minor == VERSION_MINOR
                    && table_entry.patch == VERSION_PATCH
                {
                    found_port = table_entry.port;
                } else {
                    skipped_due_to_version += 1;
                }
            }
        }
    }

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

    // Find all running processes
    let running_processes = find_running_processes();

    let mut nodes = Vec::new();

    // Read and parse all proc table entries
    let entries = fs::read_dir(proc_table_dir).context("Failed to read proc table directory")?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let table_entry = match ProcTableEntry::decode(&content) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // If process is no longer running, remove the stale entry
        if !running_processes.contains(&table_entry.pid) {
            let _ = fs::remove_file(&path);
            continue;
        }

        // Add all nodes from this entry
        for node in table_entry.nodes {
            nodes.push(node);
        }
    }

    // Sort for consistent output
    nodes.sort();
    Ok(nodes)
}

/// Creates a process table entry for the current process
///
/// This is called by gNB and UE instances to register themselves.
///
/// # Arguments
///
/// * `nodes` - List of node names to register
/// * `cmd_port` - The command port for CLI communication
#[allow(dead_code)]
pub fn create_proc_table(nodes: &[String], cmd_port: u16) -> Result<PathBuf> {
    let proc_table_dir = Path::new(PROC_TABLE_DIR);

    // Create directory if it doesn't exist
    fs::create_dir_all(proc_table_dir).context("Failed to create proc table directory")?;

    // Generate a unique filename
    let pid = std::process::id();
    let filename = format!("{:016x}", {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        nodes.hash(&mut hasher);
        cmd_port.hash(&mut hasher);
        pid.hash(&mut hasher);
        hasher.finish()
    });

    let file_path = proc_table_dir.join(filename);

    // Create the entry
    let entry = ProcTableEntry {
        major: VERSION_MAJOR,
        minor: VERSION_MINOR,
        patch: VERSION_PATCH,
        pid,
        port: cmd_port,
        nodes: nodes.to_vec(),
    };

    // Write the entry
    fs::write(&file_path, entry.encode()).context("Failed to write proc table entry")?;

    Ok(file_path)
}

/// Removes a process table entry
#[allow(dead_code)]
pub fn remove_proc_table(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_file(path).context("Failed to remove proc table entry")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proc_table_entry_encode_decode() {
        let entry = ProcTableEntry {
            major: 1,
            minor: 2,
            patch: 3,
            pid: 12345,
            port: 5000,
            nodes: vec!["gnb1".to_string(), "gnb2".to_string()],
        };

        let encoded = entry.encode();
        let decoded = ProcTableEntry::decode(&encoded).unwrap();

        assert_eq!(decoded.major, 1);
        assert_eq!(decoded.minor, 2);
        assert_eq!(decoded.patch, 3);
        assert_eq!(decoded.pid, 12345);
        assert_eq!(decoded.port, 5000);
        assert_eq!(decoded.nodes, vec!["gnb1", "gnb2"]);
    }

    #[test]
    fn test_proc_table_entry_decode_invalid() {
        assert!(ProcTableEntry::decode("").is_err());
        assert!(ProcTableEntry::decode("1 2 3").is_err());
        assert!(ProcTableEntry::decode("1 2 3 4 5 2 node1").is_err()); // Missing second node
    }
}
