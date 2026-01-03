//! TUN interface wrapper for async operations.
//!
//! This module provides the main TUN interface implementation using the `tun-rs` crate.

use std::io;
use std::net::Ipv4Addr;
use std::sync::Arc;

use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, error, info, warn};
use tun_rs::AsyncDevice;

use super::config::TunConfig;

/// Maximum buffer size for reading from TUN interface.
/// This should be larger than the MTU to accommodate any overhead.
const MAX_PACKET_SIZE: usize = 65535;

/// Errors that can occur during TUN interface operations.
#[derive(Debug, Error)]
pub enum TunError {
    /// Failed to create TUN device.
    #[error("failed to create TUN device: {0}")]
    CreateFailed(String),

    /// Failed to configure TUN device.
    #[error("failed to configure TUN device: {0}")]
    ConfigureFailed(String),

    /// I/O error during read/write operations.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Interface not configured with required parameters.
    #[error("interface not configured: {0}")]
    NotConfigured(String),

    /// Invalid packet data.
    #[error("invalid packet: {0}")]
    InvalidPacket(String),
}

/// TUN interface wrapper providing async read/write operations.
///
/// This struct wraps the `tun-rs` async device and provides a higher-level
/// interface for reading and writing IP packets.
pub struct TunInterface {
    /// The underlying async TUN device.
    device: AsyncDevice,
    /// The allocated interface name.
    name: String,
    /// Configuration used to create this interface.
    config: TunConfig,
    /// Whether the interface has been configured with IP address.
    configured: bool,
}

impl TunInterface {
    /// Creates a new TUN interface with the given configuration.
    ///
    /// This allocates a new TUN device with a name based on the configuration's
    /// name prefix (e.g., "uesimtun0", "uesimtun1", etc.) or uses the specific
    /// name if provided.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The TUN device cannot be created (e.g., insufficient permissions)
    /// - The requested interface name is already in use
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = TunConfig::new("uesimtun");
    /// let tun = TunInterface::create(config).await?;
    /// println!("Created interface: {}", tun.name());
    /// ```
    pub async fn create(config: TunConfig) -> Result<Self, TunError> {
        let mut tun_config = tun_rs::Configuration::default();

        // Set interface name
        if let Some(ref name) = config.name {
            tun_config.name(name);
            debug!(name = %name, "Creating TUN interface with specific name");
        } else {
            // Use name prefix - tun-rs will append a number
            tun_config.name(&config.name_prefix);
            debug!(prefix = %config.name_prefix, "Creating TUN interface with prefix");
        }

        // Set MTU
        tun_config.mtu(config.mtu);

        // Configure as TUN (not TAP) - layer 3 only
        tun_config.layer(tun_rs::Layer::L3);

        // Don't bring up the interface yet - we'll do that after configuring IP
        tun_config.up();

        // Create the device
        let device = tun_rs::create_as_async(&tun_config)
            .map_err(|e| TunError::CreateFailed(e.to_string()))?;

        // Get the actual allocated name
        let name = device
            .as_ref()
            .name()
            .map_err(|e| TunError::CreateFailed(format!("failed to get interface name: {e}")))?;

        info!(name = %name, mtu = config.mtu, "TUN interface created");

        Ok(Self {
            device,
            name,
            config,
            configured: false,
        })
    }

    /// Creates and configures a TUN interface with IP address.
    ///
    /// This is a convenience method that creates the interface and configures
    /// it with the IP address and netmask from the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if creation or configuration fails.
    pub async fn create_and_configure(config: TunConfig) -> Result<Self, TunError> {
        let mut tun = Self::create(config).await?;
        tun.configure().await?;
        Ok(tun)
    }

    /// Configures the TUN interface with IP address and netmask.
    ///
    /// This must be called after `create()` to assign an IP address to the
    /// interface. The address and netmask are taken from the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No address is configured
    /// - The configuration fails (e.g., invalid address)
    pub async fn configure(&mut self) -> Result<(), TunError> {
        let address = self
            .config
            .address
            .ok_or_else(|| TunError::NotConfigured("IP address not set".to_string()))?;

        let netmask = self
            .config
            .netmask
            .unwrap_or(Ipv4Addr::new(255, 255, 255, 0));

        // Configure IP address using system commands
        // The tun-rs crate doesn't provide direct IP configuration, so we use ip command
        self.configure_ip_address(address, netmask).await?;

        self.configured = true;
        info!(
            name = %self.name,
            address = %address,
            netmask = %netmask,
            "TUN interface configured"
        );

        Ok(())
    }

    /// Configures the IP address using system commands.
    async fn configure_ip_address(
        &self,
        address: Ipv4Addr,
        netmask: Ipv4Addr,
    ) -> Result<(), TunError> {
        // Calculate prefix length from netmask
        let prefix_len = netmask_to_prefix_len(netmask);

        // Use ip command to configure the interface
        let output = tokio::process::Command::new("ip")
            .args([
                "addr",
                "add",
                &format!("{address}/{prefix_len}"),
                "dev",
                &self.name,
            ])
            .output()
            .await
            .map_err(|e| TunError::ConfigureFailed(format!("failed to run ip command: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Ignore "File exists" error - address may already be configured
            if !stderr.contains("File exists") {
                return Err(TunError::ConfigureFailed(format!(
                    "ip addr add failed: {stderr}"
                )));
            }
            warn!(name = %self.name, "IP address already configured");
        }

        // Bring the interface up
        let output = tokio::process::Command::new("ip")
            .args(["link", "set", &self.name, "up"])
            .output()
            .await
            .map_err(|e| TunError::ConfigureFailed(format!("failed to run ip command: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TunError::ConfigureFailed(format!(
                "ip link set up failed: {stderr}"
            )));
        }

        Ok(())
    }

    /// Returns the name of the TUN interface.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the configuration used to create this interface.
    #[must_use]
    pub fn config(&self) -> &TunConfig {
        &self.config
    }

    /// Returns whether the interface has been configured with an IP address.
    #[must_use]
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Returns the PDU Session Identifier associated with this interface.
    #[must_use]
    pub fn psi(&self) -> Option<u8> {
        self.config.psi
    }

    /// Reads an IP packet from the TUN interface.
    ///
    /// This method blocks until a packet is available or an error occurs.
    ///
    /// # Returns
    ///
    /// Returns the raw IP packet data (without any TUN header).
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub async fn read(&mut self) -> Result<Vec<u8>, TunError> {
        let mut buf = vec![0u8; MAX_PACKET_SIZE];
        let n = self.device.read(&mut buf).await?;
        buf.truncate(n);
        Ok(buf)
    }

    /// Reads an IP packet into the provided buffer.
    ///
    /// # Returns
    ///
    /// Returns the number of bytes read.
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub async fn read_buf(&mut self, buf: &mut [u8]) -> Result<usize, TunError> {
        let n = self.device.read(buf).await?;
        Ok(n)
    }

    /// Writes an IP packet to the TUN interface.
    ///
    /// The packet should be a complete IP packet (IPv4 or IPv6).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The write operation fails
    /// - The packet is invalid
    pub async fn write(&mut self, packet: &[u8]) -> Result<usize, TunError> {
        if packet.is_empty() {
            return Err(TunError::InvalidPacket("empty packet".to_string()));
        }

        let n = self.device.write(packet).await?;
        Ok(n)
    }

    /// Splits the TUN interface into separate read and write halves.
    ///
    /// This allows concurrent reading and writing from different tasks.
    #[must_use]
    pub fn split(self) -> (TunReader, TunWriter) {
        let name = self.name.clone();
        let config = self.config.clone();
        let device = Arc::new(tokio::sync::Mutex::new(self.device));

        (
            TunReader {
                device: Arc::clone(&device),
                name: name.clone(),
            },
            TunWriter {
                device,
                name,
                config,
            },
        )
    }
}

/// Read half of a split TUN interface.
pub struct TunReader {
    device: Arc<tokio::sync::Mutex<AsyncDevice>>,
    name: String,
}

impl TunReader {
    /// Reads an IP packet from the TUN interface.
    pub async fn read(&self) -> Result<Vec<u8>, TunError> {
        let mut buf = vec![0u8; MAX_PACKET_SIZE];
        let mut device = self.device.lock().await;
        let n = device.read(&mut buf).await?;
        buf.truncate(n);
        Ok(buf)
    }

    /// Returns the interface name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Write half of a split TUN interface.
pub struct TunWriter {
    device: Arc<tokio::sync::Mutex<AsyncDevice>>,
    name: String,
    config: TunConfig,
}

impl TunWriter {
    /// Writes an IP packet to the TUN interface.
    pub async fn write(&self, packet: &[u8]) -> Result<usize, TunError> {
        if packet.is_empty() {
            return Err(TunError::InvalidPacket("empty packet".to_string()));
        }

        let mut device = self.device.lock().await;
        let n = device.write(packet).await?;
        Ok(n)
    }

    /// Returns the interface name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the PSI associated with this interface.
    #[must_use]
    pub fn psi(&self) -> Option<u8> {
        self.config.psi
    }
}

/// Converts a netmask to prefix length (CIDR notation).
fn netmask_to_prefix_len(netmask: Ipv4Addr) -> u8 {
    let bits = u32::from(netmask);
    bits.count_ones() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_netmask_to_prefix_len() {
        assert_eq!(netmask_to_prefix_len(Ipv4Addr::new(255, 255, 255, 255)), 32);
        assert_eq!(netmask_to_prefix_len(Ipv4Addr::new(255, 255, 255, 0)), 24);
        assert_eq!(netmask_to_prefix_len(Ipv4Addr::new(255, 255, 0, 0)), 16);
        assert_eq!(netmask_to_prefix_len(Ipv4Addr::new(255, 0, 0, 0)), 8);
        assert_eq!(netmask_to_prefix_len(Ipv4Addr::new(0, 0, 0, 0)), 0);
    }

    #[test]
    fn test_tun_error_display() {
        let err = TunError::CreateFailed("permission denied".to_string());
        assert!(err.to_string().contains("permission denied"));

        let err = TunError::NotConfigured("IP address not set".to_string());
        assert!(err.to_string().contains("IP address not set"));
    }
}
