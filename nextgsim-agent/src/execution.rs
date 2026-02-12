//! Intent execution engine
//!
//! Provides the `IntentExecutor` trait and built-in executors that perform
//! real computation for each intent type instead of returning placeholder successes.

use crate::{Intent, IntentType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Execution status with more granularity than a simple bool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentStatus {
    /// The intent was fully executed.
    Success,
    /// The intent was partially executed (some sub-goals achieved).
    PartialSuccess,
    /// The intent failed entirely.
    Failed,
    /// The intent was blocked by a conflict or safety constraint.
    Blocked,
}

/// Rich result returned after intent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentExecutionResult {
    /// ID of the executed intent.
    pub intent_id: String,
    /// Execution status.
    pub status: IntentStatus,
    /// Key-value output data produced by the executor.
    pub output: HashMap<String, String>,
    /// Resources that were read or modified during execution.
    pub affected_resources: Vec<AffectedResource>,
    /// Human-readable message (error detail when status != Success).
    pub message: Option<String>,
    /// Wall-clock duration of execution in microseconds.
    pub execution_time_us: u64,
}

/// A resource that was touched during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedResource {
    /// Resource kind (cell, ue, slice, `qos_flow`, ...).
    pub kind: ResourceKind,
    /// Unique ID of the resource.
    pub id: String,
    /// Whether the resource was only read or also written.
    pub access: ResourceAccess,
}

/// Kinds of network resources.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceKind {
    /// A cell / gNB.
    Cell,
    /// A UE.
    Ue,
    /// A network slice.
    Slice,
    /// A `QoS` flow.
    QosFlow,
    /// A frequency / spectrum resource.
    Spectrum,
    /// Generic named resource.
    Other(String),
}

/// Access mode on a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceAccess {
    /// Read-only access.
    Read,
    /// Read-write access.
    Write,
}

// ---------------------------------------------------------------------------
// State provider trait
// ---------------------------------------------------------------------------

/// Trait for providing network state to executors.
///
/// Implementations supply live data that executors use to produce real results.
pub trait StateProvider: Send + Sync + std::fmt::Debug {
    /// Read a value from the network state store.
    fn get(&self, key: &str) -> Option<String>;
    /// List all keys under a prefix (e.g., `"cell/"` returns `["cell/1", "cell/2"]`).
    fn list_keys(&self, prefix: &str) -> Vec<String>;
    /// Write a value into the state store. Returns `false` if rejected.
    fn set(&self, key: &str, value: &str) -> bool;
}

/// Simple in-memory state provider for testing and simulation.
#[derive(Debug, Default)]
pub struct InMemoryStateProvider {
    data: std::sync::RwLock<HashMap<String, String>>,
}

impl InMemoryStateProvider {
    /// Creates a new empty state provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a value (convenience for test setup).
    pub fn insert(&self, key: impl Into<String>, value: impl Into<String>) {
        if let Ok(mut map) = self.data.write() {
            map.insert(key.into(), value.into());
        }
    }
}

impl StateProvider for InMemoryStateProvider {
    fn get(&self, key: &str) -> Option<String> {
        self.data.read().ok()?.get(key).cloned()
    }

    fn list_keys(&self, prefix: &str) -> Vec<String> {
        self.data
            .read()
            .ok()
            .map(|map| {
                map.keys()
                    .filter(|k| k.starts_with(prefix))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    fn set(&self, key: &str, value: &str) -> bool {
        if let Ok(mut map) = self.data.write() {
            map.insert(key.to_string(), value.to_string());
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Executor trait
// ---------------------------------------------------------------------------

/// Trait that all intent executors implement.
pub trait IntentExecutor: Send + Sync {
    /// Execute an intent and return a rich result.
    fn execute(&self, intent: &Intent, state: &dyn StateProvider) -> IntentExecutionResult;
}

// ---------------------------------------------------------------------------
// Built-in executors
// ---------------------------------------------------------------------------

/// Executor for `IntentType::Query`.
///
/// Reads data from the state provider keyed by the intent's target and parameters.
#[derive(Debug, Default)]
pub struct QueryExecutor;

impl IntentExecutor for QueryExecutor {
    fn execute(&self, intent: &Intent, state: &dyn StateProvider) -> IntentExecutionResult {
        let start = std::time::Instant::now();
        let mut output = HashMap::new();
        let mut affected = Vec::new();

        // Determine which keys to query.
        let keys: Vec<String> = if let Some(ref target) = intent.target {
            // If specific metric keys are requested via parameter "keys" (comma separated)
            if let Some(csv) = intent.parameters.get("keys") {
                csv.split(',')
                    .map(|k| format!("{}/{}", target, k.trim()))
                    .collect()
            } else {
                // List everything under the target prefix.
                state.list_keys(&format!("{target}/"))
            }
        } else if let Some(csv) = intent.parameters.get("keys") {
            csv.split(',').map(|k| k.trim().to_string()).collect()
        } else {
            vec![]
        };

        let mut found = 0u32;
        for key in &keys {
            if let Some(value) = state.get(key) {
                output.insert(key.clone(), value);
                found += 1;
            }
            // Determine resource kind from key prefix.
            let kind = resource_kind_from_key(key);
            affected.push(AffectedResource {
                kind,
                id: key.clone(),
                access: ResourceAccess::Read,
            });
        }

        let status = if keys.is_empty() {
            IntentStatus::Failed
        } else if found == keys.len() as u32 {
            IntentStatus::Success
        } else if found > 0 {
            IntentStatus::PartialSuccess
        } else {
            IntentStatus::Failed
        };

        output.insert("keys_requested".to_string(), keys.len().to_string());
        output.insert("keys_found".to_string(), found.to_string());

        let message = if status == IntentStatus::Failed {
            Some("No data found for the requested keys".to_string())
        } else {
            None
        };

        IntentExecutionResult {
            intent_id: intent.id.clone(),
            status,
            output,
            affected_resources: affected,
            message,
            execution_time_us: start.elapsed().as_micros() as u64,
        }
    }
}

/// Executor for `IntentType::OptimizeResources`.
///
/// Reads current allocations, computes a simple proportional-fair reallocation,
/// and writes the new values back into the state provider.
#[derive(Debug, Default)]
pub struct OptimizeResourcesExecutor;

impl IntentExecutor for OptimizeResourcesExecutor {
    fn execute(&self, intent: &Intent, state: &dyn StateProvider) -> IntentExecutionResult {
        let start = std::time::Instant::now();
        let mut output = HashMap::new();
        let mut affected = Vec::new();

        let target = intent.target.as_deref().unwrap_or("network");

        // Discover cells in scope.
        let cell_keys = state.list_keys(&format!("{target}/cell/"));
        // Fallback: try top-level cell/ prefix.
        let cell_keys = if cell_keys.is_empty() {
            state.list_keys("cell/")
        } else {
            cell_keys
        };

        if cell_keys.is_empty() {
            return IntentExecutionResult {
                intent_id: intent.id.clone(),
                status: IntentStatus::Failed,
                output,
                affected_resources: affected,
                message: Some("No cell resource data found in state".to_string()),
                execution_time_us: start.elapsed().as_micros() as u64,
            };
        }

        // Read current load values and compute total.
        let mut loads: Vec<(String, f64)> = Vec::new();
        let mut total_load: f64 = 0.0;
        for key in &cell_keys {
            let load_key = format!("{key}/load");
            let load: f64 = state
                .get(&load_key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.5);
            loads.push((key.clone(), load));
            total_load += load;
            affected.push(AffectedResource {
                kind: ResourceKind::Cell,
                id: key.clone(),
                access: ResourceAccess::Read,
            });
        }

        // Total available resource budget (parameter or default 100).
        let total_budget: f64 = intent
            .parameters
            .get("total_budget")
            .and_then(|v| v.parse().ok())
            .unwrap_or(100.0);

        // Proportional-fair allocation: higher load -> more resources.
        let avg_load = if loads.is_empty() {
            0.5
        } else {
            total_load / loads.len() as f64
        };

        let mut changes = Vec::new();
        for (cell_key, load) in &loads {
            let weight = if avg_load > 0.0 {
                load / avg_load
            } else {
                1.0
            };
            let allocation = (total_budget / loads.len() as f64) * weight;
            let alloc_key = format!("{cell_key}/allocation");
            state.set(&alloc_key, &format!("{allocation:.2}"));
            changes.push(format!("{cell_key}={allocation:.2}"));
            affected.push(AffectedResource {
                kind: ResourceKind::Cell,
                id: cell_key.clone(),
                access: ResourceAccess::Write,
            });
        }

        output.insert("cells_optimized".to_string(), loads.len().to_string());
        output.insert("total_budget".to_string(), format!("{total_budget:.2}"));
        output.insert("changes".to_string(), changes.join(";"));

        IntentExecutionResult {
            intent_id: intent.id.clone(),
            status: IntentStatus::Success,
            output,
            affected_resources: affected,
            message: None,
            execution_time_us: start.elapsed().as_micros() as u64,
        }
    }
}

/// Executor for `IntentType::TriggerHandover`.
///
/// Reads signal quality data, identifies the best target cell, and writes
/// a handover command into the state.
#[derive(Debug, Default)]
pub struct TriggerHandoverExecutor;

impl IntentExecutor for TriggerHandoverExecutor {
    fn execute(&self, intent: &Intent, state: &dyn StateProvider) -> IntentExecutionResult {
        let start = std::time::Instant::now();
        let mut output = HashMap::new();
        let mut affected = Vec::new();

        let ue_id = match intent.target.as_deref() {
            Some(id) => id,
            None => {
                return IntentExecutionResult {
                    intent_id: intent.id.clone(),
                    status: IntentStatus::Failed,
                    output,
                    affected_resources: affected,
                    message: Some("TriggerHandover requires a target UE ID".to_string()),
                    execution_time_us: start.elapsed().as_micros() as u64,
                };
            }
        };

        // Read source cell from state.
        let source_cell = state
            .get(&format!("ue/{ue_id}/serving_cell"))
            .unwrap_or_else(|| "unknown".to_string());

        affected.push(AffectedResource {
            kind: ResourceKind::Ue,
            id: ue_id.to_string(),
            access: ResourceAccess::Read,
        });

        // Gather candidate cells from RSRP measurements for this UE.
        let rsrp_prefix = format!("ue/{ue_id}/rsrp/");
        let rsrp_keys = state.list_keys(&rsrp_prefix);
        let mut best_cell: Option<String> = None;
        let mut best_rsrp: f64 = f64::NEG_INFINITY;

        for rsrp_key in &rsrp_keys {
            // Extract cell path from the RSRP key (after the prefix).
            let cell_path = rsrp_key.strip_prefix(&rsrp_prefix).unwrap_or(rsrp_key);

            // Skip the source cell.
            if cell_path == source_cell || cell_path.ends_with(&source_cell) {
                continue;
            }

            if let Some(rsrp_str) = state.get(rsrp_key) {
                if let Ok(rsrp) = rsrp_str.parse::<f64>() {
                    if rsrp > best_rsrp {
                        best_rsrp = rsrp;
                        best_cell = Some(cell_path.to_string());
                    }
                }
            }
            affected.push(AffectedResource {
                kind: ResourceKind::Cell,
                id: cell_path.to_string(),
                access: ResourceAccess::Read,
            });
        }

        // If a target cell parameter was explicitly provided, use it.
        let target_cell = intent
            .parameters
            .get("target_cell")
            .cloned()
            .or(best_cell);

        match target_cell {
            Some(tc) => {
                // Write handover command into state.
                let ho_key = format!("ue/{ue_id}/handover_cmd");
                let ho_value = serde_json::json!({
                    "source_cell": source_cell,
                    "target_cell": tc,
                    "ue_id": ue_id,
                    "trigger": "agent_intent",
                    "intent_id": intent.id,
                })
                .to_string();
                state.set(&ho_key, &ho_value);

                affected.push(AffectedResource {
                    kind: ResourceKind::Ue,
                    id: ue_id.to_string(),
                    access: ResourceAccess::Write,
                });

                output.insert("source_cell".to_string(), source_cell);
                output.insert("target_cell".to_string(), tc);
                output.insert("best_rsrp".to_string(), format!("{best_rsrp:.1}"));

                IntentExecutionResult {
                    intent_id: intent.id.clone(),
                    status: IntentStatus::Success,
                    output,
                    affected_resources: affected,
                    message: None,
                    execution_time_us: start.elapsed().as_micros() as u64,
                }
            }
            None => IntentExecutionResult {
                intent_id: intent.id.clone(),
                status: IntentStatus::Failed,
                output,
                affected_resources: affected,
                message: Some("No suitable target cell found for handover".to_string()),
                execution_time_us: start.elapsed().as_micros() as u64,
            },
        }
    }
}

/// Executor for `IntentType::AdjustQos`.
///
/// Reads current `QoS` parameters, computes adjustments within bounds, and
/// writes updated values.
#[derive(Debug, Default)]
pub struct AdjustQosExecutor;

impl IntentExecutor for AdjustQosExecutor {
    fn execute(&self, intent: &Intent, state: &dyn StateProvider) -> IntentExecutionResult {
        let start = std::time::Instant::now();
        let mut output = HashMap::new();
        let mut affected = Vec::new();

        let target = match intent.target.as_deref() {
            Some(t) => t,
            None => {
                return IntentExecutionResult {
                    intent_id: intent.id.clone(),
                    status: IntentStatus::Failed,
                    output,
                    affected_resources: affected,
                    message: Some("AdjustQos requires a target (flow ID or UE ID)".to_string()),
                    execution_time_us: start.elapsed().as_micros() as u64,
                };
            }
        };

        // Read current QoS parameters.
        let current_5qi: u32 = state
            .get(&format!("qos/{target}/5qi"))
            .and_then(|v| v.parse().ok())
            .unwrap_or(9); // default non-GBR

        let current_mbr: f64 = state
            .get(&format!("qos/{target}/mbr_mbps"))
            .and_then(|v| v.parse().ok())
            .unwrap_or(10.0);

        let current_latency: f64 = state
            .get(&format!("qos/{target}/latency_ms"))
            .and_then(|v| v.parse().ok())
            .unwrap_or(100.0);

        affected.push(AffectedResource {
            kind: ResourceKind::QosFlow,
            id: target.to_string(),
            access: ResourceAccess::Read,
        });

        // Requested adjustments from parameters.
        let target_5qi: Option<u32> = intent.parameters.get("target_5qi").and_then(|v| v.parse().ok());
        let mbr_delta_pct: f64 = intent
            .parameters
            .get("mbr_change_pct")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let latency_delta_pct: f64 = intent
            .parameters
            .get("latency_change_pct")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);

        // Compute new values.
        let new_5qi = target_5qi.unwrap_or(current_5qi);
        let new_mbr = (current_mbr * (1.0 + mbr_delta_pct / 100.0)).max(0.1);
        let new_latency = (current_latency * (1.0 + latency_delta_pct / 100.0)).max(1.0);

        // Write back.
        state.set(&format!("qos/{target}/5qi"), &new_5qi.to_string());
        state.set(&format!("qos/{target}/mbr_mbps"), &format!("{new_mbr:.2}"));
        state.set(
            &format!("qos/{target}/latency_ms"),
            &format!("{new_latency:.2}"),
        );

        affected.push(AffectedResource {
            kind: ResourceKind::QosFlow,
            id: target.to_string(),
            access: ResourceAccess::Write,
        });

        output.insert("previous_5qi".to_string(), current_5qi.to_string());
        output.insert("new_5qi".to_string(), new_5qi.to_string());
        output.insert("previous_mbr_mbps".to_string(), format!("{current_mbr:.2}"));
        output.insert("new_mbr_mbps".to_string(), format!("{new_mbr:.2}"));
        output.insert(
            "previous_latency_ms".to_string(),
            format!("{current_latency:.2}"),
        );
        output.insert("new_latency_ms".to_string(), format!("{new_latency:.2}"));

        IntentExecutionResult {
            intent_id: intent.id.clone(),
            status: IntentStatus::Success,
            output,
            affected_resources: affected,
            message: None,
            execution_time_us: start.elapsed().as_micros() as u64,
        }
    }
}

/// Executor for `IntentType::CreateSlice`.
///
/// Generates a full slice configuration and writes it into the state.
#[derive(Debug, Default)]
pub struct CreateSliceExecutor;

impl IntentExecutor for CreateSliceExecutor {
    fn execute(&self, intent: &Intent, state: &dyn StateProvider) -> IntentExecutionResult {
        let start = std::time::Instant::now();
        let mut output = HashMap::new();
        let mut affected = Vec::new();

        let slice_id = intent
            .parameters
            .get("slice_id")
            .cloned()
            .unwrap_or_else(|| format!("slice-{}", &intent.id[..8.min(intent.id.len())]));

        // Check if slice already exists.
        if state.get(&format!("slice/{slice_id}/sst")).is_some() {
            return IntentExecutionResult {
                intent_id: intent.id.clone(),
                status: IntentStatus::Failed,
                output,
                affected_resources: affected,
                message: Some(format!("Slice {slice_id} already exists")),
                execution_time_us: start.elapsed().as_micros() as u64,
            };
        }

        // Parameters.
        let sst: u8 = intent
            .parameters
            .get("sst")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1); // eMBB default
        let sd: String = intent
            .parameters
            .get("sd")
            .cloned()
            .unwrap_or_else(|| "000001".to_string());
        let max_ues: u32 = intent
            .parameters
            .get("max_ues")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000);
        let guaranteed_mbr: f64 = intent
            .parameters
            .get("guaranteed_mbr_mbps")
            .and_then(|v| v.parse().ok())
            .unwrap_or(100.0);
        let isolation: bool = intent
            .parameters
            .get("isolation")
            .map(|v| v == "true")
            .unwrap_or(false);

        // Write slice configuration.
        let prefix = format!("slice/{slice_id}");
        state.set(&format!("{prefix}/sst"), &sst.to_string());
        state.set(&format!("{prefix}/sd"), &sd);
        state.set(&format!("{prefix}/max_ues"), &max_ues.to_string());
        state.set(
            &format!("{prefix}/guaranteed_mbr_mbps"),
            &format!("{guaranteed_mbr:.2}"),
        );
        state.set(&format!("{prefix}/isolation"), &isolation.to_string());
        state.set(&format!("{prefix}/status"), "active");
        state.set(&format!("{prefix}/created_by"), &intent.agent_id.0);

        affected.push(AffectedResource {
            kind: ResourceKind::Slice,
            id: slice_id.clone(),
            access: ResourceAccess::Write,
        });

        output.insert("slice_id".to_string(), slice_id);
        output.insert("sst".to_string(), sst.to_string());
        output.insert("sd".to_string(), sd);
        output.insert("max_ues".to_string(), max_ues.to_string());
        output.insert("guaranteed_mbr_mbps".to_string(), format!("{guaranteed_mbr:.2}"));
        output.insert("isolation".to_string(), isolation.to_string());

        IntentExecutionResult {
            intent_id: intent.id.clone(),
            status: IntentStatus::Success,
            output,
            affected_resources: affected,
            message: None,
            execution_time_us: start.elapsed().as_micros() as u64,
        }
    }
}

/// Executor for `IntentType::ModifySlice`.
///
/// Reads existing slice config, applies parameter changes, and writes back.
#[derive(Debug, Default)]
pub struct ModifySliceExecutor;

impl IntentExecutor for ModifySliceExecutor {
    fn execute(&self, intent: &Intent, state: &dyn StateProvider) -> IntentExecutionResult {
        let start = std::time::Instant::now();
        let mut output = HashMap::new();
        let mut affected = Vec::new();

        let slice_id = match intent.target.as_deref() {
            Some(id) => id,
            None => {
                return IntentExecutionResult {
                    intent_id: intent.id.clone(),
                    status: IntentStatus::Failed,
                    output,
                    affected_resources: affected,
                    message: Some("ModifySlice requires a target slice ID".to_string()),
                    execution_time_us: start.elapsed().as_micros() as u64,
                };
            }
        };

        let prefix = format!("slice/{slice_id}");

        // Check existence.
        if state.get(&format!("{prefix}/sst")).is_none() {
            return IntentExecutionResult {
                intent_id: intent.id.clone(),
                status: IntentStatus::Failed,
                output,
                affected_resources: affected,
                message: Some(format!("Slice {slice_id} does not exist")),
                execution_time_us: start.elapsed().as_micros() as u64,
            };
        }

        affected.push(AffectedResource {
            kind: ResourceKind::Slice,
            id: slice_id.to_string(),
            access: ResourceAccess::Read,
        });

        // Apply each provided parameter.
        let modifiable = ["max_ues", "guaranteed_mbr_mbps", "isolation", "status", "sd"];
        let mut modified_count = 0u32;
        for param_name in &modifiable {
            if let Some(value) = intent.parameters.get(*param_name) {
                let key = format!("{prefix}/{param_name}");
                let old_value = state.get(&key).unwrap_or_default();
                state.set(&key, value);
                output.insert(format!("previous_{param_name}"), old_value);
                output.insert(format!("new_{param_name}"), value.clone());
                modified_count += 1;
            }
        }

        if modified_count == 0 {
            return IntentExecutionResult {
                intent_id: intent.id.clone(),
                status: IntentStatus::Failed,
                output,
                affected_resources: affected,
                message: Some("No modifiable parameters provided".to_string()),
                execution_time_us: start.elapsed().as_micros() as u64,
            };
        }

        affected.push(AffectedResource {
            kind: ResourceKind::Slice,
            id: slice_id.to_string(),
            access: ResourceAccess::Write,
        });

        output.insert("modified_params".to_string(), modified_count.to_string());

        IntentExecutionResult {
            intent_id: intent.id.clone(),
            status: IntentStatus::Success,
            output,
            affected_resources: affected,
            message: None,
            execution_time_us: start.elapsed().as_micros() as u64,
        }
    }
}

// ---------------------------------------------------------------------------
// Executor registry
// ---------------------------------------------------------------------------

/// Registry that maps intent types to their executors.
#[derive(Default)]
pub struct ExecutorRegistry {
    custom_executors: HashMap<String, Box<dyn IntentExecutor>>,
}

impl ExecutorRegistry {
    /// Creates a registry pre-loaded with all built-in executors.
    pub fn new() -> Self {
        Self {
            custom_executors: HashMap::new(),
        }
    }

    /// Registers a custom executor for a `Custom(name)` intent type.
    pub fn register_custom(&mut self, name: impl Into<String>, executor: Box<dyn IntentExecutor>) {
        self.custom_executors.insert(name.into(), executor);
    }

    /// Execute an intent through the appropriate built-in or custom executor.
    pub fn execute(&self, intent: &Intent, state: &dyn StateProvider) -> IntentExecutionResult {
        match &intent.intent_type {
            IntentType::Query => QueryExecutor.execute(intent, state),
            IntentType::OptimizeResources => OptimizeResourcesExecutor.execute(intent, state),
            IntentType::TriggerHandover => TriggerHandoverExecutor.execute(intent, state),
            IntentType::AdjustQos => AdjustQosExecutor.execute(intent, state),
            IntentType::CreateSlice => CreateSliceExecutor.execute(intent, state),
            IntentType::ModifySlice => ModifySliceExecutor.execute(intent, state),
            IntentType::Custom(name) => {
                if let Some(executor) = self.custom_executors.get(name) {
                    executor.execute(intent, state)
                } else {
                    IntentExecutionResult {
                        intent_id: intent.id.clone(),
                        status: IntentStatus::Failed,
                        output: HashMap::new(),
                        affected_resources: vec![],
                        message: Some(format!("No executor registered for custom intent '{name}'")),
                        execution_time_us: 0,
                    }
                }
            }
        }
    }
}

impl std::fmt::Debug for ExecutorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutorRegistry")
            .field("custom_executor_count", &self.custom_executors.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resource_kind_from_key(key: &str) -> ResourceKind {
    if key.starts_with("cell/") || key.contains("/cell/") {
        ResourceKind::Cell
    } else if key.starts_with("ue/") || key.contains("/ue/") {
        ResourceKind::Ue
    } else if key.starts_with("slice/") || key.contains("/slice/") {
        ResourceKind::Slice
    } else if key.starts_with("qos/") || key.contains("/qos/") {
        ResourceKind::QosFlow
    } else {
        ResourceKind::Other(key.split('/').next().unwrap_or("unknown").to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentId;

    fn make_state() -> InMemoryStateProvider {
        let sp = InMemoryStateProvider::new();
        // Seed cell data.
        sp.insert("cell/cell-1/load", "0.8");
        sp.insert("cell/cell-2/load", "0.3");
        sp.insert("cell/cell-3/load", "0.5");
        // Seed UE data.
        sp.insert("ue/ue-42/serving_cell", "cell-1");
        sp.insert("ue/ue-42/rsrp/cell/cell-2", "-85.0");
        sp.insert("ue/ue-42/rsrp/cell/cell-3", "-92.0");
        // Seed QoS data.
        sp.insert("qos/flow-7/5qi", "9");
        sp.insert("qos/flow-7/mbr_mbps", "50.0");
        sp.insert("qos/flow-7/latency_ms", "20.0");
        // Seed a slice.
        sp.insert("slice/existing-slice/sst", "1");
        sp.insert("slice/existing-slice/max_ues", "500");
        sp
    }

    #[test]
    fn test_query_executor_success() {
        let state = make_state();
        let intent = Intent::new(AgentId::new("a1"), IntentType::Query)
            .with_target("cell/cell-1")
            .with_param("keys", "load");

        let result = QueryExecutor.execute(&intent, &state);
        assert_eq!(result.status, IntentStatus::Success);
        assert_eq!(result.output.get("cell/cell-1/load").map(String::as_str), Some("0.8"));
    }

    #[test]
    fn test_query_executor_missing_data() {
        let state = make_state();
        let intent = Intent::new(AgentId::new("a1"), IntentType::Query)
            .with_target("cell/cell-99")
            .with_param("keys", "load");

        let result = QueryExecutor.execute(&intent, &state);
        assert_eq!(result.status, IntentStatus::Failed);
    }

    #[test]
    fn test_optimize_resources_executor() {
        let state = make_state();
        let intent = Intent::new(AgentId::new("a1"), IntentType::OptimizeResources)
            .with_param("total_budget", "90");

        let result = OptimizeResourcesExecutor.execute(&intent, &state);
        assert_eq!(result.status, IntentStatus::Success);
        assert_eq!(result.output.get("cells_optimized").map(String::as_str), Some("3"));
    }

    #[test]
    fn test_trigger_handover_executor() {
        let state = make_state();
        let intent = Intent::new(AgentId::new("a1"), IntentType::TriggerHandover)
            .with_target("ue-42");

        let result = TriggerHandoverExecutor.execute(&intent, &state);
        assert_eq!(result.status, IntentStatus::Success);
        assert_eq!(result.output.get("target_cell").map(String::as_str), Some("cell/cell-2"));
    }

    #[test]
    fn test_adjust_qos_executor() {
        let state = make_state();
        let intent = Intent::new(AgentId::new("a1"), IntentType::AdjustQos)
            .with_target("flow-7")
            .with_param("mbr_change_pct", "20")
            .with_param("latency_change_pct", "-10");

        let result = AdjustQosExecutor.execute(&intent, &state);
        assert_eq!(result.status, IntentStatus::Success);
        assert_eq!(result.output.get("new_mbr_mbps").map(String::as_str), Some("60.00"));
        assert_eq!(result.output.get("new_latency_ms").map(String::as_str), Some("18.00"));
    }

    #[test]
    fn test_create_slice_executor() {
        let state = make_state();
        let intent = Intent::new(AgentId::new("a1"), IntentType::CreateSlice)
            .with_param("slice_id", "new-slice-1")
            .with_param("sst", "2")
            .with_param("max_ues", "200")
            .with_param("guaranteed_mbr_mbps", "50.0");

        let result = CreateSliceExecutor.execute(&intent, &state);
        assert_eq!(result.status, IntentStatus::Success);
        assert_eq!(state.get("slice/new-slice-1/sst"), Some("2".to_string()));
    }

    #[test]
    fn test_create_slice_already_exists() {
        let state = make_state();
        let intent = Intent::new(AgentId::new("a1"), IntentType::CreateSlice)
            .with_param("slice_id", "existing-slice");

        let result = CreateSliceExecutor.execute(&intent, &state);
        assert_eq!(result.status, IntentStatus::Failed);
    }

    #[test]
    fn test_modify_slice_executor() {
        let state = make_state();
        let intent = Intent::new(AgentId::new("a1"), IntentType::ModifySlice)
            .with_target("existing-slice")
            .with_param("max_ues", "1000");

        let result = ModifySliceExecutor.execute(&intent, &state);
        assert_eq!(result.status, IntentStatus::Success);
        assert_eq!(state.get("slice/existing-slice/max_ues"), Some("1000".to_string()));
    }

    #[test]
    fn test_executor_registry() {
        let state = make_state();
        let registry = ExecutorRegistry::new();

        let intent = Intent::new(AgentId::new("a1"), IntentType::Query)
            .with_target("cell/cell-1")
            .with_param("keys", "load");

        let result = registry.execute(&intent, &state);
        assert_eq!(result.status, IntentStatus::Success);
    }
}
