//! Network transport utilities
//!
//! Provides async transport wrappers for UDP communication.

use std::net::SocketAddr;
use tokio::net::UdpSocket;

use crate::Error;

/// Async UDP socket wrapper for RLS and GTP-U protocols.
///
/// Provides a simple interface for binding, sending, and receiving UDP datagrams.
///
/// # Example
///
/// ```ignore
/// use std::net::SocketAddr;
/// use nextgsim_common::UdpTransport;
///
/// async fn example() -> Result<(), nextgsim_common::Error> {
///     let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
///     let transport = UdpTransport::bind(addr).await?;
///     
///     let dest: SocketAddr = "127.0.0.1:5000".parse().unwrap();
///     transport.send_to(b"hello", dest).await?;
///     
///     let (data, src) = transport.recv_from().await?;
///     Ok(())
/// }
/// ```
pub struct UdpTransport {
    socket: UdpSocket,
}

impl UdpTransport {
    /// Binds a UDP socket to the specified address.
    ///
    /// # Arguments
    ///
    /// * `addr` - The socket address to bind to. Use port 0 for automatic port assignment.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket cannot be bound to the specified address.
    pub async fn bind(addr: SocketAddr) -> Result<Self, Error> {
        let socket = UdpSocket::bind(addr).await?;
        Ok(Self { socket })
    }

    /// Sends data to the specified destination address.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to send.
    /// * `addr` - The destination socket address.
    ///
    /// # Errors
    ///
    /// Returns an error if the send operation fails.
    pub async fn send_to(&self, data: &[u8], addr: SocketAddr) -> Result<(), Error> {
        self.socket.send_to(data, addr).await?;
        Ok(())
    }

    /// Receives data from the socket.
    ///
    /// Returns the received data and the source address.
    ///
    /// # Errors
    ///
    /// Returns an error if the receive operation fails.
    pub async fn recv_from(&self) -> Result<(Vec<u8>, SocketAddr), Error> {
        let mut buf = vec![0u8; 65535]; // Max UDP datagram size
        let (len, addr) = self.socket.recv_from(&mut buf).await?;
        buf.truncate(len);
        Ok((buf, addr))
    }

    /// Returns the local address this socket is bound to.
    ///
    /// # Errors
    ///
    /// Returns an error if the local address cannot be retrieved.
    pub fn local_addr(&self) -> Result<SocketAddr, Error> {
        Ok(self.socket.local_addr()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bind_and_local_addr() {
        let addr: SocketAddr = "127.0.0.1:0".parse().expect("valid address");
        let transport = UdpTransport::bind(addr).await.expect("bind should succeed");
        
        let local = transport.local_addr().expect("local_addr should succeed");
        assert_eq!(local.ip(), addr.ip());
        assert_ne!(local.port(), 0); // Port should be assigned
    }

    #[tokio::test]
    async fn test_send_and_recv() {
        // Create two transports
        let addr1: SocketAddr = "127.0.0.1:0".parse().expect("valid address");
        let addr2: SocketAddr = "127.0.0.1:0".parse().expect("valid address");
        
        let transport1 = UdpTransport::bind(addr1).await.expect("bind should succeed");
        let transport2 = UdpTransport::bind(addr2).await.expect("bind should succeed");
        
        let local1 = transport1.local_addr().expect("local_addr should succeed");
        let local2 = transport2.local_addr().expect("local_addr should succeed");
        
        // Send from transport1 to transport2
        let test_data = b"hello, udp!";
        transport1.send_to(test_data, local2).await.expect("send should succeed");
        
        // Receive on transport2
        let (received, src) = transport2.recv_from().await.expect("recv should succeed");
        
        assert_eq!(received, test_data);
        assert_eq!(src, local1);
    }

    #[tokio::test]
    async fn test_large_datagram() {
        let addr1: SocketAddr = "127.0.0.1:0".parse().expect("valid address");
        let addr2: SocketAddr = "127.0.0.1:0".parse().expect("valid address");
        
        let transport1 = UdpTransport::bind(addr1).await.expect("bind should succeed");
        let transport2 = UdpTransport::bind(addr2).await.expect("bind should succeed");
        
        let local2 = transport2.local_addr().expect("local_addr should succeed");
        
        // Send a larger datagram (8KB)
        let test_data: Vec<u8> = (0..8192).map(|i| (i % 256) as u8).collect();
        transport1.send_to(&test_data, local2).await.expect("send should succeed");
        
        let (received, _) = transport2.recv_from().await.expect("recv should succeed");
        assert_eq!(received, test_data);
    }
}
