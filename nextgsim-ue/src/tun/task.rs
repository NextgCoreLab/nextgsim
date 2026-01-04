//! TUN task implementation
//!
//! Manages TUN interface lifecycle and IP packet handling for PDU sessions.

use std::collections::HashMap;
use std::net::Ipv4Addr;

use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use nextgsim_common::OctetString;

use super::config::TunConfig;
use super::interface::TunInterface;
use super::packet::{is_valid_ip_packet, IpPacket};

/// TUN task configuration
#[derive(Debug, Clone)]
pub struct TunTaskConfig {
    /// Interface name prefix
    pub name_prefix: String,
    /// Default MTU
    pub mtu: u16,
}

impl Default for TunTaskConfig {
    fn default() -> Self {
        Self {
            name_prefix: "uesimtun".to_string(),
            mtu: 1400,
        }
    }
}

/// Messages for the TUN task
#[derive(Debug)]
pub enum TunMessage {
    /// Create a TUN interface for a PDU session
    CreateInterface {
        /// PDU session ID
        psi: i32,
        /// IPv4 address assigned to the session
        address: Ipv4Addr,
        /// Network mask
        netmask: Ipv4Addr,
    },
    /// Destroy a TUN interface
    DestroyInterface {
        /// PDU session ID
        psi: i32,
    },
    /// Write data to TUN interface (downlink)
    WriteData {
        /// PDU session ID
        psi: i32,
        /// IP packet data
        data: OctetString,
    },
}

/// Messages sent from TUN task to App task
#[derive(Debug, Clone)]
pub enum TunAppMessage {
    /// TUN interface created successfully
    InterfaceCreated {
        /// PDU session ID
        psi: i32,
        /// Interface name
        name: String,
    },
    /// TUN interface destroyed
    InterfaceDestroyed {
        /// PDU session ID
        psi: i32,
    },
    /// Uplink data from TUN interface
    UplinkData {
        /// PDU session ID
        psi: i32,
        /// IP packet data
        data: OctetString,
    },
    /// Error occurred
    Error {
        /// Error message
        message: String,
    },
}

/// TUN task for managing TUN interfaces and IP packet handling
///
/// This task:
/// - Creates/destroys TUN interfaces for PDU sessions
/// - Reads IP packets from TUN interfaces (uplink)
/// - Writes IP packets to TUN interfaces (downlink)
pub struct TunTask {
    /// Task configuration
    config: TunTaskConfig,
    /// Handle to send messages to App task
    app_tx: mpsc::Sender<TunAppMessage>,
    /// Active TUN interfaces indexed by PSI
    interfaces: HashMap<i32, TunInterface>,
}

impl TunTask {
    /// Create a new TUN task
    pub fn new(config: TunTaskConfig, app_tx: mpsc::Sender<TunAppMessage>) -> Self {
        Self {
            config,
            app_tx,
            interfaces: HashMap::new(),
        }
    }

    /// Run the TUN task
    ///
    /// Processes messages and handles TUN interface I/O.
    pub async fn run(&mut self, mut rx: mpsc::Receiver<TunMessage>) {
        info!("TUN task started");

        loop {
            // Use tokio::select! to handle both messages and TUN reads
            tokio::select! {
                // Handle incoming messages
                msg = rx.recv() => {
                    match msg {
                        Some(tun_msg) => {
                            self.handle_message(tun_msg).await;
                        }
                        None => {
                            info!("TUN task channel closed");
                            break;
                        }
                    }
                }
            }
        }

        // Cleanup all interfaces
        self.cleanup().await;
        info!("TUN task stopped");
    }

    /// Handle a TUN message
    async fn handle_message(&mut self, msg: TunMessage) {
        match msg {
            TunMessage::CreateInterface {
                psi,
                address,
                netmask,
            } => {
                self.create_interface(psi, address, netmask).await;
            }
            TunMessage::DestroyInterface { psi } => {
                self.destroy_interface(psi).await;
            }
            TunMessage::WriteData { psi, data } => {
                self.write_data(psi, data.as_ref()).await;
            }
        }
    }

    /// Create a TUN interface for a PDU session
    async fn create_interface(&mut self, psi: i32, address: Ipv4Addr, netmask: Ipv4Addr) {
        // Check if interface already exists
        if self.interfaces.contains_key(&psi) {
            warn!("TUN interface for PSI {} already exists", psi);
            return;
        }

        let config = TunConfig::builder()
            .name_prefix(&self.config.name_prefix)
            .address(address)
            .netmask(netmask)
            .mtu(self.config.mtu)
            .psi(psi as u8)
            .build();

        match TunInterface::create(&config, psi).await {
            Ok(interface) => {
                let name = interface.name().to_string();
                info!(
                    "Created TUN interface {} for PSI {} with address {}/{}",
                    name, psi, address, netmask
                );
                self.interfaces.insert(psi, interface);

                // Notify App task of session establishment
                let _ = self
                    .app_tx
                    .send(TunAppMessage::InterfaceCreated { psi, name })
                    .await;
            }
            Err(e) => {
                error!("Failed to create TUN interface for PSI {}: {}", psi, e);
                let _ = self
                    .app_tx
                    .send(TunAppMessage::Error {
                        message: format!("Failed to create TUN interface: {}", e),
                    })
                    .await;
            }
        }
    }

    /// Destroy a TUN interface
    async fn destroy_interface(&mut self, psi: i32) {
        if let Some(interface) = self.interfaces.remove(&psi) {
            info!(
                "Destroyed TUN interface {} for PSI {}",
                interface.name(),
                psi
            );

            // Notify App task of session release
            let _ = self
                .app_tx
                .send(TunAppMessage::InterfaceDestroyed { psi })
                .await;
        } else {
            warn!("TUN interface for PSI {} not found", psi);
        }
    }

    /// Write data to a TUN interface (downlink)
    async fn write_data(&mut self, psi: i32, data: &[u8]) {
        let interface = match self.interfaces.get(&psi) {
            Some(iface) => iface,
            None => {
                warn!("TUN interface for PSI {} not found, dropping packet", psi);
                return;
            }
        };

        // Validate the IP packet
        if !is_valid_ip_packet(data) {
            warn!("Invalid IP packet for PSI {}, dropping", psi);
            return;
        }

        match interface.write_all(data).await {
            Ok(()) => {
                debug!(
                    "Wrote {} bytes to TUN interface for PSI {}",
                    data.len(),
                    psi
                );
            }
            Err(e) => {
                error!("Failed to write to TUN interface for PSI {}: {}", psi, e);
                let _ = self
                    .app_tx
                    .send(TunAppMessage::Error {
                        message: format!("TUN write error: {}", e),
                    })
                    .await;
            }
        }
    }

    /// Cleanup all TUN interfaces
    async fn cleanup(&mut self) {
        let psis: Vec<i32> = self.interfaces.keys().copied().collect();
        for psi in psis {
            self.destroy_interface(psi).await;
        }
    }
}

/// Spawn a TUN reader task for a specific interface
///
/// This function spawns a background task that continuously reads from
/// the TUN interface and sends packets to the App task.
///
/// # Arguments
/// * `interface` - The TUN interface to read from
/// * `app_tx` - Handle to send messages to the App task
///
/// # Returns
/// A join handle for the spawned task
pub fn spawn_tun_reader(
    interface: TunInterface,
    app_tx: mpsc::Sender<TunAppMessage>,
) -> tokio::task::JoinHandle<()> {
    let psi = interface.psi();
    let name = interface.name().to_string();

    tokio::spawn(async move {
        info!("TUN reader started for interface {} (PSI {})", name, psi);

        loop {
            match interface.recv().await {
                Ok(buf) if buf.is_empty() => {
                    // EOF - interface closed
                    info!("TUN interface {} closed (EOF)", name);
                    break;
                }
                Ok(buf) => {
                    // Validate and parse the packet
                    if let Some(packet) = IpPacket::parse(&buf) {
                        debug!(
                            "Read {:?} packet ({} bytes) from TUN interface for PSI {}",
                            packet.version(),
                            buf.len(),
                            psi
                        );

                        // Send to App task for GTP encapsulation
                        let data = OctetString::from(buf);
                        if let Err(e) = app_tx
                            .send(TunAppMessage::UplinkData { psi, data })
                            .await
                        {
                            error!("Failed to send TUN data to App task: {}", e);
                            break;
                        }
                    } else {
                        warn!("Invalid IP packet from TUN interface for PSI {}", psi);
                    }
                }
                Err(e) => {
                    error!("TUN read error for PSI {}: {}", psi, e);
                    let _ = app_tx
                        .send(TunAppMessage::Error {
                            message: format!("TUN read error: {}", e),
                        })
                        .await;
                    break;
                }
            }
        }

        info!("TUN reader stopped for interface {} (PSI {})", name, psi);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tun_task_config_default() {
        let config = TunTaskConfig::default();
        assert_eq!(config.name_prefix, "uesimtun");
        assert_eq!(config.mtu, 1400);
    }

    #[test]
    fn test_tun_message_variants() {
        let create = TunMessage::CreateInterface {
            psi: 1,
            address: Ipv4Addr::new(10, 45, 0, 1),
            netmask: Ipv4Addr::new(255, 255, 255, 0),
        };
        assert!(matches!(create, TunMessage::CreateInterface { psi: 1, .. }));

        let destroy = TunMessage::DestroyInterface { psi: 1 };
        assert!(matches!(destroy, TunMessage::DestroyInterface { psi: 1 }));

        let write = TunMessage::WriteData {
            psi: 1,
            data: OctetString::from(vec![0x45, 0x00]),
        };
        assert!(matches!(write, TunMessage::WriteData { psi: 1, .. }));
    }

    #[tokio::test]
    async fn test_tun_task_creation() {
        let (tx, _rx) = mpsc::channel(10);
        let config = TunTaskConfig::default();

        let task = TunTask::new(config, tx);
        assert!(task.interfaces.is_empty());
    }
}
