//! Integration test framework for nextgsim
#![allow(missing_docs)]
//!
//! This crate provides test utilities and mock components for integration testing
//! of the nextgsim 5G UE and gNB simulator.
//!
//! # Components
//!
//! - [`mock_amf`] - Mock AMF for testing UE registration and session management
//! - [`test_fixtures`] - Common test fixtures and configuration helpers
//! - [`test_utils`] - Utility functions for test setup and assertions
//!
//! # Test Categories
//!
//! 1. **UE Registration Tests** - Test UE registration with mock AMF
//! 2. **PDU Session Tests** - Test PDU session establishment/modification/release
//! 3. **User Plane Tests** - Test GTP-U data flow
//! 4. **Multi-UE Tests** - Test scenarios with multiple concurrent UEs

pub mod mock_amf;
pub mod test_fixtures;
pub mod test_utils;
pub mod ai_integration;

pub use mock_amf::{MockAmf, MockAmfConfig, MockAmfEvent};
pub use test_fixtures::{TestConfig, TestUeConfig, TestGnbConfig};
pub use test_utils::{
    init_test_logging, wait_for_condition, TestResult,
    DEFAULT_TEST_TIMEOUT, DEFAULT_POLL_INTERVAL,
    TEST_SCTP_PORT_BASE, TEST_GTP_PORT_BASE, TEST_CLI_PORT_BASE,
    get_test_port,
};
