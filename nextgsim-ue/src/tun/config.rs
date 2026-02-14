//! TUN interface configuration.
//!
//! This module provides configuration structures for TUN interface creation.

use std::net::Ipv4Addr;

/// Default MTU for TUN interfaces.
pub const DEFAULT_MTU: u16 = 1400;

/// Default TUN interface name prefix.
pub const DEFAULT_NAME_PREFIX: &str = "uesimtun";

/// Configuration for TUN interface creation.
#[derive(Debug, Clone)]
pub struct TunConfig {
    /// Name prefix for the TUN interface (e.g., "uesimtun" -> "uesimtun0", "uesimtun1", etc.)
    pub name_prefix: String,
    /// Specific interface name (if set, overrides `name_prefix`)
    pub name: Option<String>,
    /// IPv4 address to assign to the interface
    pub address: Option<Ipv4Addr>,
    /// Netmask for the interface
    pub netmask: Option<Ipv4Addr>,
    /// MTU (Maximum Transmission Unit)
    pub mtu: u16,
    /// Whether to configure routing for this interface
    pub configure_routing: bool,
    /// PDU Session Identifier (PSI) associated with this TUN interface
    pub psi: Option<u8>,
}

impl Default for TunConfig {
    fn default() -> Self {
        Self {
            name_prefix: DEFAULT_NAME_PREFIX.to_string(),
            name: None,
            address: None,
            netmask: None,
            mtu: DEFAULT_MTU,
            configure_routing: false,
            psi: None,
        }
    }
}

impl TunConfig {
    /// Creates a new TUN configuration with the given name prefix.
    #[must_use]
    pub fn new(name_prefix: impl Into<String>) -> Self {
        Self {
            name_prefix: name_prefix.into(),
            ..Default::default()
        }
    }

    /// Creates a new TUN configuration with a specific interface name.
    #[must_use]
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Default::default()
        }
    }

    /// Returns a builder for constructing TUN configuration.
    #[must_use]
    pub fn builder() -> TunConfigBuilder {
        TunConfigBuilder::default()
    }
}

/// Builder for TUN configuration.
#[derive(Debug, Default)]
pub struct TunConfigBuilder {
    config: TunConfig,
}

impl TunConfigBuilder {
    /// Sets the name prefix for the TUN interface.
    #[must_use]
    pub fn name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.config.name_prefix = prefix.into();
        self
    }

    /// Sets a specific name for the TUN interface.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = Some(name.into());
        self
    }

    /// Sets the IPv4 address for the TUN interface.
    #[must_use]
    pub fn address(mut self, addr: Ipv4Addr) -> Self {
        self.config.address = Some(addr);
        self
    }

    /// Sets the netmask for the TUN interface.
    #[must_use]
    pub fn netmask(mut self, mask: Ipv4Addr) -> Self {
        self.config.netmask = Some(mask);
        self
    }

    /// Sets the MTU for the TUN interface.
    #[must_use]
    pub fn mtu(mut self, mtu: u16) -> Self {
        self.config.mtu = mtu;
        self
    }

    /// Enables routing configuration for the TUN interface.
    #[must_use]
    pub fn configure_routing(mut self, enable: bool) -> Self {
        self.config.configure_routing = enable;
        self
    }

    /// Sets the PDU Session Identifier associated with this TUN interface.
    #[must_use]
    pub fn psi(mut self, psi: u8) -> Self {
        self.config.psi = Some(psi);
        self
    }

    /// Builds the TUN configuration.
    #[must_use]
    pub fn build(self) -> TunConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TunConfig::default();
        assert_eq!(config.name_prefix, DEFAULT_NAME_PREFIX);
        assert!(config.name.is_none());
        assert!(config.address.is_none());
        assert!(config.netmask.is_none());
        assert_eq!(config.mtu, DEFAULT_MTU);
        assert!(!config.configure_routing);
        assert!(config.psi.is_none());
    }

    #[test]
    fn test_config_with_prefix() {
        let config = TunConfig::new("mytun");
        assert_eq!(config.name_prefix, "mytun");
    }

    #[test]
    fn test_config_with_name() {
        let config = TunConfig::with_name("tun0");
        assert_eq!(config.name, Some("tun0".to_string()));
    }

    #[test]
    fn test_builder() {
        let config = TunConfig::builder()
            .name_prefix("test")
            .address(Ipv4Addr::new(10, 45, 0, 2))
            .netmask(Ipv4Addr::new(255, 255, 255, 0))
            .mtu(1500)
            .configure_routing(true)
            .psi(1)
            .build();

        assert_eq!(config.name_prefix, "test");
        assert_eq!(config.address, Some(Ipv4Addr::new(10, 45, 0, 2)));
        assert_eq!(config.netmask, Some(Ipv4Addr::new(255, 255, 255, 0)));
        assert_eq!(config.mtu, 1500);
        assert!(config.configure_routing);
        assert_eq!(config.psi, Some(1));
    }
}
