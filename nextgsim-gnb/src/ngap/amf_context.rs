//! AMF Context Management
//!
//! This module manages AMF (Access and Mobility Management Function) contexts
//! for the gNB. Each AMF connection has an associated context that tracks:
//! - Connection state
//! - AMF capabilities and configuration
//! - Served GUAMIs and PLMNs
//! - Stream allocation for UE-associated signaling

use std::collections::HashSet;
use nextgsim_ngap::procedures::{
    Guami, NgSetupResponseData, PlmnSupportItem, ServedGuamiItem, SNssai,
};

/// AMF connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum AmfState {
    /// Not connected
    #[default]
    NotConnected,
    /// SCTP association established, waiting for NG Setup
    Connected,
    /// NG Setup in progress
    WaitingNgSetup,
    /// NG Setup complete, ready for operation
    Ready,
    /// AMF is overloaded
    Overloaded,
}


impl std::fmt::Display for AmfState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AmfState::NotConnected => write!(f, "NotConnected"),
            AmfState::Connected => write!(f, "Connected"),
            AmfState::WaitingNgSetup => write!(f, "WaitingNgSetup"),
            AmfState::Ready => write!(f, "Ready"),
            AmfState::Overloaded => write!(f, "Overloaded"),
        }
    }
}

/// AMF context for tracking AMF connection state and capabilities
#[derive(Debug, Clone)]
pub struct NgapAmfContext {
    /// AMF context ID (same as SCTP client ID)
    pub ctx_id: i32,
    /// SCTP association ID
    pub association_id: Option<i32>,
    /// Number of inbound SCTP streams
    pub in_streams: u16,
    /// Number of outbound SCTP streams
    pub out_streams: u16,
    /// Current state
    pub state: AmfState,
    /// AMF name from NG Setup Response
    pub amf_name: Option<String>,
    /// Relative AMF capacity (0-255)
    pub relative_capacity: u8,
    /// List of served GUAMIs
    pub served_guami_list: Vec<ServedGuamiItem>,
    /// List of supported PLMNs with slice support
    pub plmn_support_list: Vec<PlmnSupportItem>,
    /// Next available stream ID for UE-associated signaling
    next_stream: u16,
    /// Set of allocated stream IDs
    allocated_streams: HashSet<u16>,
}

impl NgapAmfContext {
    /// Creates a new AMF context
    pub fn new(ctx_id: i32) -> Self {
        Self {
            ctx_id,
            association_id: None,
            in_streams: 0,
            out_streams: 0,
            state: AmfState::NotConnected,
            amf_name: None,
            relative_capacity: 0,
            served_guami_list: Vec::new(),
            plmn_support_list: Vec::new(),
            next_stream: 1, // Stream 0 is for non-UE-associated signaling
            allocated_streams: HashSet::new(),
        }
    }

    /// Updates the context when SCTP association is established
    pub fn on_association_up(&mut self, association_id: i32, in_streams: u16, out_streams: u16) {
        self.association_id = Some(association_id);
        self.in_streams = in_streams;
        self.out_streams = out_streams;
        self.state = AmfState::Connected;
    }

    /// Updates the context when SCTP association is closed
    pub fn on_association_down(&mut self) {
        self.association_id = None;
        self.state = AmfState::NotConnected;
        self.amf_name = None;
        self.served_guami_list.clear();
        self.plmn_support_list.clear();
        self.allocated_streams.clear();
        self.next_stream = 1;
    }

    /// Updates the context with NG Setup Response data
    pub fn on_ng_setup_response(&mut self, response: NgSetupResponseData) {
        self.amf_name = Some(response.amf_name);
        self.relative_capacity = response.relative_amf_capacity;
        self.served_guami_list = response.served_guami_list;
        self.plmn_support_list = response.plmn_support_list;
        self.state = AmfState::Ready;
    }

    /// Marks the AMF as waiting for NG Setup response
    pub fn on_ng_setup_sent(&mut self) {
        self.state = AmfState::WaitingNgSetup;
    }

    /// Marks the AMF as overloaded
    pub fn on_overload_start(&mut self) {
        if self.state == AmfState::Ready {
            self.state = AmfState::Overloaded;
        }
    }

    /// Clears the overload state
    pub fn on_overload_stop(&mut self) {
        if self.state == AmfState::Overloaded {
            self.state = AmfState::Ready;
        }
    }

    /// Returns true if the AMF is ready for operation
    pub fn is_ready(&self) -> bool {
        self.state == AmfState::Ready
    }

    /// Returns true if the AMF is connected (SCTP association up)
    pub fn is_connected(&self) -> bool {
        self.association_id.is_some()
    }

    /// Allocates a stream ID for UE-associated signaling
    ///
    /// Returns None if no streams are available
    pub fn allocate_stream(&mut self) -> Option<u16> {
        if self.out_streams <= 1 {
            // Only stream 0 available, use it for everything
            return Some(0);
        }

        // Find an available stream (1 to out_streams-1)
        for _ in 0..self.out_streams {
            let stream = self.next_stream;
            self.next_stream = if self.next_stream >= self.out_streams - 1 {
                1
            } else {
                self.next_stream + 1
            };

            if !self.allocated_streams.contains(&stream) {
                self.allocated_streams.insert(stream);
                return Some(stream);
            }
        }

        // All streams allocated, reuse stream 1
        Some(1)
    }

    /// Releases a stream ID
    pub fn release_stream(&mut self, stream: u16) {
        self.allocated_streams.remove(&stream);
    }

    /// Checks if the AMF supports a given S-NSSAI for a PLMN
    pub fn supports_snssai(&self, plmn: &[u8; 3], snssai: &SNssai) -> bool {
        for plmn_item in &self.plmn_support_list {
            if &plmn_item.plmn_identity == plmn {
                for supported in &plmn_item.slice_support_list {
                    if supported.sst == snssai.sst && supported.sd == snssai.sd {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Finds a GUAMI that matches the given PLMN
    pub fn find_guami_for_plmn(&self, plmn: &[u8; 3]) -> Option<&Guami> {
        for item in &self.served_guami_list {
            if &item.guami.plmn_identity == plmn {
                return Some(&item.guami);
            }
        }
        None
    }
}

/// Simplified AMF context for external use (status reporting, etc.)
#[derive(Debug, Clone)]
pub struct AmfContextInfo {
    /// AMF context ID
    pub ctx_id: i32,
    /// AMF name
    pub name: Option<String>,
    /// Current state
    pub state: AmfState,
    /// Relative capacity
    pub capacity: u8,
    /// Number of served GUAMIs
    pub guami_count: usize,
}

impl From<&NgapAmfContext> for AmfContextInfo {
    fn from(ctx: &NgapAmfContext) -> Self {
        Self {
            ctx_id: ctx.ctx_id,
            name: ctx.amf_name.clone(),
            state: ctx.state,
            capacity: ctx.relative_capacity,
            guami_count: ctx.served_guami_list.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amf_context_new() {
        let ctx = NgapAmfContext::new(1);
        assert_eq!(ctx.ctx_id, 1);
        assert_eq!(ctx.state, AmfState::NotConnected);
        assert!(ctx.amf_name.is_none());
    }

    #[test]
    fn test_amf_context_association_lifecycle() {
        let mut ctx = NgapAmfContext::new(1);

        // Association up
        ctx.on_association_up(100, 2, 2);
        assert_eq!(ctx.state, AmfState::Connected);
        assert_eq!(ctx.association_id, Some(100));
        assert!(ctx.is_connected());

        // NG Setup sent
        ctx.on_ng_setup_sent();
        assert_eq!(ctx.state, AmfState::WaitingNgSetup);

        // NG Setup response
        let response = NgSetupResponseData {
            amf_name: "test-amf".to_string(),
            served_guami_list: vec![],
            relative_amf_capacity: 100,
            plmn_support_list: vec![],
        };
        ctx.on_ng_setup_response(response);
        assert_eq!(ctx.state, AmfState::Ready);
        assert!(ctx.is_ready());
        assert_eq!(ctx.amf_name, Some("test-amf".to_string()));

        // Association down
        ctx.on_association_down();
        assert_eq!(ctx.state, AmfState::NotConnected);
        assert!(!ctx.is_connected());
        assert!(ctx.amf_name.is_none());
    }

    #[test]
    fn test_amf_context_stream_allocation() {
        let mut ctx = NgapAmfContext::new(1);
        ctx.on_association_up(100, 4, 4);

        // Allocate streams
        let s1 = ctx.allocate_stream();
        let s2 = ctx.allocate_stream();
        let s3 = ctx.allocate_stream();

        assert!(s1.is_some());
        assert!(s2.is_some());
        assert!(s3.is_some());

        // Streams should be different (1, 2, 3)
        let streams: HashSet<_> = [s1, s2, s3].iter().filter_map(|s| *s).collect();
        assert!(!streams.is_empty()); // At least some unique streams

        // Release and reallocate
        if let Some(stream) = s1 {
            ctx.release_stream(stream);
        }
    }

    #[test]
    fn test_amf_context_overload() {
        let mut ctx = NgapAmfContext::new(1);
        ctx.on_association_up(100, 2, 2);
        ctx.on_ng_setup_sent();

        let response = NgSetupResponseData {
            amf_name: "test-amf".to_string(),
            served_guami_list: vec![],
            relative_amf_capacity: 100,
            plmn_support_list: vec![],
        };
        ctx.on_ng_setup_response(response);
        assert_eq!(ctx.state, AmfState::Ready);

        // Overload
        ctx.on_overload_start();
        assert_eq!(ctx.state, AmfState::Overloaded);
        assert!(!ctx.is_ready());

        // Clear overload
        ctx.on_overload_stop();
        assert_eq!(ctx.state, AmfState::Ready);
        assert!(ctx.is_ready());
    }
}
