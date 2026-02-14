//! Test fixtures and configuration helpers
//!
//! Provides pre-configured test scenarios and configuration builders.

use nextgsim_common::{Plmn, SNssai, Tai};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

/// Test configuration container
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct TestConfig {
    /// UE configuration
    pub ue: TestUeConfig,
    /// gNB configuration
    pub gnb: TestGnbConfig,
    /// Mock AMF configuration
    pub amf: TestAmfConfig,
}


/// Test UE configuration
#[derive(Debug, Clone)]
pub struct TestUeConfig {
    /// SUPI (IMSI)
    pub supi: String,
    /// Home PLMN
    pub hplmn: Plmn,
    /// Permanent key (K)
    pub key: [u8; 16],
    /// Operator key (OP or OPc)
    pub op: [u8; 16],
    /// Whether OP is OPc
    pub op_is_opc: bool,
    /// Requested NSSAI
    pub nssai: Vec<SNssai>,
    /// gNB search list
    pub gnb_search_list: Vec<SocketAddr>,
}

impl Default for TestUeConfig {
    fn default() -> Self {
        Self {
            supi: "imsi-001010000000001".to_string(),
            hplmn: Plmn::new(1, 1, false),
            key: [0x46, 0x5b, 0x5c, 0xe8, 0xb1, 0x99, 0xb4, 0x9f,
                  0xaa, 0x5f, 0x0a, 0x2e, 0xe2, 0x38, 0xa6, 0xbc],
            op: [0xcd, 0xc2, 0x02, 0xd5, 0x12, 0x3e, 0x20, 0xf6,
                 0x2b, 0x6d, 0x67, 0x6a, 0xc7, 0x2c, 0xb3, 0x18],
            op_is_opc: false,
            nssai: vec![SNssai::new(1)],
            gnb_search_list: vec![
                SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 4997),
            ],
        }
    }
}

impl TestUeConfig {
    /// Create a new test UE config with custom IMSI
    pub fn with_imsi(mut self, imsi: &str) -> Self {
        self.supi = format!("imsi-{imsi}");
        self
    }

    /// Create a new test UE config with custom PLMN
    pub fn with_plmn(mut self, mcc: u16, mnc: u16) -> Self {
        self.hplmn = Plmn::new(mcc, mnc, mnc >= 100);
        self
    }

    /// Add gNB address to search list
    pub fn with_gnb_addr(mut self, addr: SocketAddr) -> Self {
        self.gnb_search_list.push(addr);
        self
    }
}

/// Test gNB configuration
#[derive(Debug, Clone)]
pub struct TestGnbConfig {
    /// gNB ID
    pub gnb_id: u32,
    /// gNB ID length in bits
    pub gnb_id_length: u8,
    /// NR Cell Identity
    pub nci: u64,
    /// TAI
    pub tai: Tai,
    /// Supported S-NSSAIs
    pub nssai: Vec<SNssai>,
    /// AMF addresses
    pub amf_addresses: Vec<SocketAddr>,
    /// Local bind address for NGAP
    pub ngap_bind_addr: SocketAddr,
    /// Local bind address for GTP-U
    pub gtp_bind_addr: SocketAddr,
    /// Local bind address for RLS
    pub rls_bind_addr: SocketAddr,
}

impl Default for TestGnbConfig {
    fn default() -> Self {
        Self {
            gnb_id: 1,
            gnb_id_length: 22,
            nci: 0x000000010,
            tai: Tai::new(Plmn::new(1, 1, false), 1),
            nssai: vec![SNssai::new(1)],
            amf_addresses: vec![
                SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 38412),
            ],
            ngap_bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            gtp_bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 2152),
            rls_bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 4997),
        }
    }
}

impl TestGnbConfig {
    /// Set AMF address
    pub fn with_amf_addr(mut self, addr: SocketAddr) -> Self {
        self.amf_addresses = vec![addr];
        self
    }

    /// Set NGAP bind address
    pub fn with_ngap_bind(mut self, addr: SocketAddr) -> Self {
        self.ngap_bind_addr = addr;
        self
    }

    /// Set GTP-U bind address
    pub fn with_gtp_bind(mut self, addr: SocketAddr) -> Self {
        self.gtp_bind_addr = addr;
        self
    }

    /// Set RLS bind address
    pub fn with_rls_bind(mut self, addr: SocketAddr) -> Self {
        self.rls_bind_addr = addr;
        self
    }
}

/// Test mock AMF configuration
#[derive(Debug, Clone)]
pub struct TestAmfConfig {
    /// AMF name
    pub amf_name: String,
    /// Served GUAMIs
    pub served_guamis: Vec<Guami>,
    /// Supported PLMNs
    pub plmn_support: Vec<PlmnSupport>,
    /// SCTP listen address
    pub sctp_addr: SocketAddr,
    /// Relative AMF capacity
    pub relative_capacity: u8,
}

/// GUAMI (Globally Unique AMF Identifier)
#[derive(Debug, Clone)]
pub struct Guami {
    pub plmn: Plmn,
    pub amf_region_id: u8,
    pub amf_set_id: u16,
    pub amf_pointer: u8,
}

/// PLMN support information
#[derive(Debug, Clone)]
pub struct PlmnSupport {
    pub plmn: Plmn,
    pub nssai: Vec<SNssai>,
}

impl Default for TestAmfConfig {
    fn default() -> Self {
        Self {
            amf_name: "Test-AMF".to_string(),
            served_guamis: vec![Guami {
                plmn: Plmn::new(1, 1, false),
                amf_region_id: 1,
                amf_set_id: 1,
                amf_pointer: 0,
            }],
            plmn_support: vec![PlmnSupport {
                plmn: Plmn::new(1, 1, false),
                nssai: vec![SNssai::new(1)],
            }],
            sctp_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 38412),
            relative_capacity: 255,
        }
    }
}

impl TestAmfConfig {
    /// Set SCTP listen address
    pub fn with_sctp_addr(mut self, addr: SocketAddr) -> Self {
        self.sctp_addr = addr;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TestConfig::default();
        assert_eq!(config.ue.supi, "imsi-001010000000001");
        assert_eq!(config.gnb.gnb_id, 1);
        assert_eq!(config.amf.amf_name, "Test-AMF");
    }

    #[test]
    fn test_ue_config_builder() {
        let config = TestUeConfig::default()
            .with_imsi("001010000000002")
            .with_plmn(310, 260);
        
        assert_eq!(config.supi, "imsi-001010000000002");
        assert_eq!(config.hplmn.mcc, 310);
        assert_eq!(config.hplmn.mnc, 260);
    }

    #[test]
    fn test_gnb_config_builder() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)), 38412);
        let config = TestGnbConfig::default().with_amf_addr(addr);
        
        assert_eq!(config.amf_addresses[0], addr);
    }
}
