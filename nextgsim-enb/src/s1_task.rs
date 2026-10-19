//! The S1-MME driver: establishes the SCTP association (TS 36.412 §7: the eNB
//! does), runs S1 Setup over it (TS 36.413 §8.7.3), and re-initiates after a
//! failure, honouring Time To Wait (§8.7.3.3).
//!
//! Transport is behind [`S1Transport`] so the procedure logic runs in default
//! CI against an in-memory peer; production uses kernel SCTP.

use std::future::Future;
use std::net::SocketAddr;
use std::time::Duration;

use nextgsim_s1ap::codec;
use nextgsim_sctp::S1AP_PPID;
use tokio::time::sleep;

use crate::config::EnbConfig;
use crate::s1ap::{build_s1_setup_request, handle_setup_reply, MmeInfo, SetupOutcome};

/// TS 36.412 §7: "a single pair of stream identifiers shall be reserved for the
/// sole use of S1AP elementary procedures that utilize non UE-associated
/// signalling". This eNB reserves stream 0, the conventional choice.
pub const NON_UE_ASSOCIATED_STREAM: u16 = 0;

/// One received SCTP user message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S1Message {
    /// Stream it arrived on.
    pub stream: u16,
    /// PPID it carried (host order).
    pub ppid: u32,
    /// The S1AP PDU octets.
    pub data: Vec<u8>,
}

/// An established S1-MME SCTP association.
pub trait S1Transport: Send {
    /// Send `data` on `stream` with `ppid`.
    fn send(
        &mut self,
        stream: u16,
        data: &[u8],
        ppid: u32,
    ) -> impl Future<Output = std::io::Result<()>> + Send;
    /// Receive the next user message; `None` when the association is gone.
    fn recv(&mut self) -> impl Future<Output = std::io::Result<Option<S1Message>>> + Send;
}

/// Opens associations towards an MME.
pub trait S1Connector: Send {
    /// The association type it produces.
    type Transport: S1Transport;
    /// Establish an association from `local` to `remote`.
    fn connect(
        &mut self,
        local: SocketAddr,
        remote: SocketAddr,
    ) -> impl Future<Output = std::io::Result<Self::Transport>> + Send;
}

/// How one S1 Setup attempt ended.
#[derive(Debug, Clone, PartialEq)]
pub enum AttemptResult {
    /// The S1 interface is operational.
    Operational(MmeInfo),
    /// Re-initiate after this long, on the same association.
    RetryAfter(Duration),
    /// The association is gone; reconnect after this long.
    Reconnect(Duration),
}

/// Run one S1 Setup on an established association: send the request, then
/// consume PDUs until an outcome is decided or the response timer expires.
pub async fn run_s1_setup_attempt<T: S1Transport>(
    cfg: &EnbConfig,
    transport: &mut T,
) -> AttemptResult {
    let retry = Duration::from_secs(cfg.s1_setup_retry_secs);
    let request = match codec::encode_s1ap_pdu(&build_s1_setup_request(cfg)) {
        Ok(b) => b,
        Err(e) => {
            // The config is validated against every type it is encoded into,
            // so this is a defect, not an operational condition.
            log::error!("cannot encode S1 SETUP REQUEST: {e}");
            return AttemptResult::RetryAfter(retry);
        }
    };
    if let Err(e) = transport
        .send(NON_UE_ASSOCIATED_STREAM, &request, S1AP_PPID)
        .await
    {
        log::warn!("S1 SETUP REQUEST send failed: {e}");
        return AttemptResult::Reconnect(retry);
    }
    log::info!("S1 SETUP REQUEST sent (eNB ID {:#x})", cfg.enb_id);

    let deadline =
        tokio::time::Instant::now() + Duration::from_millis(cfg.s1_setup_response_timeout_ms);
    loop {
        let msg = match tokio::time::timeout_at(deadline, transport.recv()).await {
            Err(_) => {
                log::warn!(
                    "no S1 SETUP RESPONSE/FAILURE within {} ms",
                    cfg.s1_setup_response_timeout_ms
                );
                return AttemptResult::RetryAfter(retry);
            }
            Ok(Err(e)) => {
                log::warn!("S1-MME receive failed: {e}");
                return AttemptResult::Reconnect(retry);
            }
            Ok(Ok(None)) => {
                log::warn!("S1-MME association closed by the MME");
                return AttemptResult::Reconnect(retry);
            }
            Ok(Ok(Some(m))) => m,
        };
        if msg.ppid != S1AP_PPID {
            // TS 36.412 §7 assigns PPID 18 to S1AP; anything else is not S1AP.
            log::warn!(
                "discarding {} octets with PPID {} (S1AP is {S1AP_PPID})",
                msg.data.len(),
                msg.ppid
            );
            continue;
        }
        match handle_setup_reply(&msg.data) {
            SetupOutcome::Accepted {
                mme,
                error_indication,
            } => {
                send_error_indication(transport, error_indication).await;
                log::info!(
                    "S1 Setup complete: MME {} (relative capacity {}), {} served GUMMEI item(s)",
                    mme.mme_name.as_deref().unwrap_or("unnamed"),
                    mme.relative_capacity
                        .map_or_else(|| "absent".to_string(), |c| c.to_string()),
                    mme.served_gummeis.len()
                );
                return AttemptResult::Operational(mme);
            }
            SetupOutcome::Failed {
                cause,
                time_to_wait,
                error_indication,
            } => {
                send_error_indication(transport, error_indication).await;
                // §8.7.3.3: "wait at least for the indicated time".
                let wait = time_to_wait.unwrap_or(retry);
                log::warn!(
                    "S1 SETUP FAILURE (cause {cause:?}); re-initiating in {}s",
                    wait.as_secs()
                );
                return AttemptResult::RetryAfter(wait);
            }
            SetupOutcome::UnsuccessfullyTerminated { reason } => {
                // §10.3.4.2/§10.3.5: local error handling, nothing to the MME.
                log::warn!("S1 Setup unsuccessfully terminated: {reason}");
                return AttemptResult::RetryAfter(retry);
            }
            SetupOutcome::NotSetupOutcome {
                description,
                error_indication,
            } => {
                log::warn!("ignoring {description}");
                send_error_indication(transport, error_indication).await;
            }
        }
    }
}

async fn send_error_indication<T: S1Transport>(transport: &mut T, pdu: Option<Vec<u8>>) {
    if let Some(pdu) = pdu {
        // Error Indication is non-UE-associated here (no UE IDs).
        if let Err(e) = transport
            .send(NON_UE_ASSOCIATED_STREAM, &pdu, S1AP_PPID)
            .await
        {
            log::warn!("ERROR INDICATION send failed: {e}");
        } else {
            log::info!("ERROR INDICATION sent");
        }
    }
}

/// Connect to `mme` and run S1 Setup until the interface is operational,
/// reconnecting and re-initiating as §8.7.3.3 and TS 36.412 §7 require.
/// Returns the stored MME data and the live association.
pub async fn establish_s1<C: S1Connector>(
    cfg: &EnbConfig,
    connector: &mut C,
    mme: SocketAddr,
) -> (MmeInfo, C::Transport) {
    let local = SocketAddr::new(cfg.s1ap_ip, 0);
    let retry = Duration::from_secs(cfg.s1_setup_retry_secs);
    loop {
        let mut transport = match connector.connect(local, mme).await {
            Ok(t) => {
                log::info!("S1-MME SCTP association established with {mme}");
                t
            }
            Err(e) => {
                log::warn!(
                    "S1-MME SCTP connect to {mme} failed: {e}; retrying in {}s",
                    retry.as_secs()
                );
                sleep(retry).await;
                continue;
            }
        };
        loop {
            match run_s1_setup_attempt(cfg, &mut transport).await {
                AttemptResult::Operational(info) => return (info, transport),
                AttemptResult::RetryAfter(d) => sleep(d).await,
                AttemptResult::Reconnect(d) => {
                    sleep(d).await;
                    break;
                }
            }
        }
    }
}

/// After S1 Setup, keep the association serviced until it closes. Later
/// increments dispatch UE-associated procedures here; for now inbound PDUs
/// are logged.
pub async fn serve<T: S1Transport>(transport: &mut T) {
    loop {
        match transport.recv().await {
            Ok(Some(m)) => log::info!(
                "S1AP PDU on stream {} ({} octets) not handled in this build",
                m.stream,
                m.data.len()
            ),
            Ok(None) => {
                log::warn!("S1-MME association closed");
                return;
            }
            Err(e) => {
                log::warn!("S1-MME receive failed: {e}");
                return;
            }
        }
    }
}

/// Kernel SCTP (IP protocol 132), the only S1-MME transport a real MME
/// accepts (TS 36.412 §7).
#[cfg(all(target_os = "linux", feature = "kernel-sctp"))]
pub mod kernel {
    use super::{S1Connector, S1Message, S1Transport};
    use nextgsim_sctp::{KernelSctpAssociation, SctpConfig};
    use std::net::SocketAddr;

    fn io(e: nextgsim_sctp::SctpError) -> std::io::Error {
        std::io::Error::other(e.to_string())
    }

    /// A kernel SCTP S1-MME association.
    pub struct KernelS1Transport(pub KernelSctpAssociation);

    impl S1Transport for KernelS1Transport {
        async fn send(&mut self, stream: u16, data: &[u8], ppid: u32) -> std::io::Result<()> {
            self.0.send_with_ppid(stream, data, ppid).await.map_err(io)
        }
        async fn recv(&mut self) -> std::io::Result<Option<S1Message>> {
            Ok(self.0.recv().await.map_err(io)?.map(|m| S1Message {
                stream: m.stream_id,
                ppid: m.ppid,
                data: m.data.to_vec(),
            }))
        }
    }

    /// Connects with [`KernelSctpAssociation`].
    #[derive(Default)]
    pub struct KernelConnector {
        /// SCTP parameters (stream counts and timers).
        pub config: SctpConfig,
    }

    impl S1Connector for KernelConnector {
        type Transport = KernelS1Transport;
        async fn connect(
            &mut self,
            local: SocketAddr,
            remote: SocketAddr,
        ) -> std::io::Result<KernelS1Transport> {
            KernelSctpAssociation::connect(local, remote, &self.config)
                .await
                .map(KernelS1Transport)
                .map_err(io)
        }
    }
}

#[cfg(test)]
mod tests;
