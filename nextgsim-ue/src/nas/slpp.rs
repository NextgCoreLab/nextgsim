//! The UE's SLPP endpoint (TS 38.355, nextgcore #557).
//!
//! The UE answers the LMF's SLPP transactions. They travel in UL and DL NAS
//! TRANSPORT with payload container type "SLPP message container" (`1100`,
//! TS 24.501 §9.11.3.40, §5.4.5.2.2 case c1 and §5.4.5.3.3 case c1). In
//! TS 38.355's terms the UE is Endpoint A, the one that provides capabilities
//! and location information, and the LMF is Endpoint B.
//!
//! This replaces the simulator-defined sidelink ranging report that used to
//! ride an LPP EPDU (nextgcore #542, option 2). The method is SL-RTT
//! (TS 38.305 §8.15.2). The UE reports its own SL Rx-Tx time difference per
//! peer, and the LMF turns a PAIR of them (this UE's and the peer's) into a
//! range. One UE's half is not a range by itself.
//!
//! | Received | Reply | Clause |
//! |---|---|---|
//! | RequestCapabilities | ProvideCapabilities (sl-RTT if asked), endTransaction TRUE | §5.1.5 |
//! | ProvideAssistanceData | none; the assistance data is stored | §5.2 |
//! | RequestAssistanceData | Error: the UE serves no assistance data | §5.4.3 |
//! | RequestLocationInformation | ProvideLocationInformation, sl-RTT measurements or sl-RTT-Error | §5.3, §5.4.3 |
//! | Abort | none; the transaction is aborted | §5.5.3 |
//! | Error | none | §5.4.4 |
//! | undecodable | Error, slppMessageHeaderError or slppMessageBodyError | §5.4.3 |
//!
//! The DL message's Additional information (the LCS correlation identifier)
//! is echoed in the reply's UL NAS TRANSPORT as its routing information.
//!
//! Transport (§4.3): every message the UE sends carries a sequence number
//! (§4.3.2), a duplicate of the last received sequence number is discarded,
//! and a message asking for an acknowledgement gets one (§4.3.3), duplicates
//! included.
//!
//! **Application layer IDs.** TS 38.355 names UEs by their application layer
//! ID (TS 23.287). This simulator has no application layer above ProSe, so it
//! uses each UE's 24-bit ProSe Layer-2 ID (`prose.local_l2_id`, TS 23.304
//! §5.8.2.1) as three big-endian octets, for itself and for every ranged peer.
//! The LMF must use the same mapping to pair a UE's half with its peer's.
//!
//! **Not reported:** SL PRS-RSRP. The ranging pipeline measures no SL-PRS
//! power, and a reported value would be invented.

use nextgsim_nas::ies::ie1::PayloadContainerType;
use nextgsim_nas::messages::mm::UlNasTransport;
use nextgsim_slpp::generated::*;
use nextgsim_slpp::{decode, decode_ie, encode, encode_ie};
use tracing::{debug, info, warn};

/// Tc, the NR basic time unit (TS 38.211 §4.1): 1 / (480000 * 4096) s.
pub const TC_SECONDS: f64 = 1.0 / (480_000.0 * 4096.0);

/// Edge of the absolute SL Rx-Tx reporting range, in Tc (TS 38.133
/// §10.4A.4.1.1: from -985024 Tc to 985024 Tc).
const RX_TX_RANGE_TC: f64 = 985_024.0;

/// Highest reported value for k = 2 (TS 38.133 table 10.4A.4.1.1-1,
/// `SL_RX-TX_TIME_DIFFERENCE_492513`), equal to the ASN.1 `k2` upper bound.
pub const RX_TX_K2_MAX: u32 = 492_513;

/// Speed of light in vacuum, m/s.
const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;

/// `locationInformationType` values that ask for measurements. SL-RTT is
/// UE-assisted here (positioningModes '01'B), so these are the ones answered
/// with measurements (TS 38.355 §6.5, LocationInformationType, root order).
const MEASUREMENT_TYPES: [u8; 10] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19];

/// `errorCause` values of CommonIEsError (TS 38.355 §6.3).
const ERROR_CAUSE_UNDEFINED: u8 = 0;
const ERROR_CAUSE_HEADER: u8 = 1;
const ERROR_CAUSE_BODY: u8 = 2;

/// SL-RTT-LocationInformationError `notAllRequestedMeasurementsPossible`.
const SL_RTT_NOT_ALL_REQUESTED_MEASUREMENTS_POSSIBLE: u8 = 2;

/// LocationFailureCause `requestedMethodNotSupported`.
const REQUESTED_METHOD_NOT_SUPPORTED: u8 = 1;

/// Report value for k = 2 of an SL Rx-Tx time difference given in seconds
/// (TS 38.133 table 10.4A.4.1.1-1): value 0 below -985024 Tc, value 492513 at
/// or above 985024 Tc, and in between the value v with
/// `-985024 + 4(v-1) <= T < -985024 + 4v` (T in Tc). `None` for a non-finite
/// input.
pub fn rx_tx_report_value_k2(seconds: f64) -> Option<u32> {
    if !seconds.is_finite() {
        return None;
    }
    let t_tc = seconds / TC_SECONDS;
    if t_tc < -RX_TX_RANGE_TC {
        return Some(0);
    }
    if t_tc >= RX_TX_RANGE_TC {
        return Some(RX_TX_K2_MAX);
    }
    Some(((t_tc + RX_TX_RANGE_TC) / 4.0).floor() as u32 + 1)
}

/// The three-octet application layer ID this simulator gives a UE with the
/// 24-bit ProSe Layer-2 ID `l2_id` (see the module docs). `None` when the
/// value does not fit 24 bits.
pub fn application_layer_id(l2_id: u64) -> Option<Vec<u8>> {
    let id = u32::try_from(l2_id).ok().filter(|id| *id <= 0x00FF_FFFF)?;
    Some(id.to_be_bytes()[1..].to_vec())
}

/// One SL-RTT measurement: this UE's SL Rx-Tx time difference against one
/// peer (TS 38.215 §5.1.40).
#[derive(Debug, Clone, PartialEq)]
pub struct SlRttMeasurement {
    /// The peer's application layer ID.
    pub peer_application_layer_id: Vec<u8>,
    /// This UE's SL Rx-Tx time difference against the peer, in seconds.
    pub rx_tx_time_difference_s: f64,
}

impl SlRttMeasurement {
    /// The measurement a UE makes against a peer at `distance_m`.
    ///
    /// The simulator transmits both ends' SL-PRS at the same instant, so each
    /// UE's Rx-Tx difference is the one-way flight time and the pair sums to
    /// the round trip (TS 38.305 §8.15.2, single-sided RTT). That simultaneity
    /// is a property of the simulator, not of SL-RTT; a real pair reports
    /// unequal halves with the same sum.
    pub fn from_range(peer_application_layer_id: Vec<u8>, distance_m: f64) -> Self {
        Self {
            peer_application_layer_id,
            rx_tx_time_difference_s: distance_m / SPEED_OF_LIGHT_M_S,
        }
    }
}

/// What the endpoint wants sent after a downlink container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlppUplink {
    /// A complete plain 5GMM UL NAS TRANSPORT PDU, ready for integrity protection.
    Send(Vec<u8>),
    /// Nothing to send.
    Nothing,
}

/// The UE's SLPP endpoint.
#[derive(Debug, Clone, Default)]
pub struct SlppEndpoint {
    /// This UE's application layer ID, set into every message it sends.
    application_layer_id: Option<Vec<u8>>,
    /// The next sequence number this UE sends (§4.3.2).
    next_sequence_number: u8,
    /// The last sequence number received from the LMF (§4.3.2).
    last_received_sequence_number: Option<u8>,
    /// The most recent ProvideAssistanceData IEs, for the ranging layer.
    assistance_data: Option<ProvideAssistanceData_IEs>,
    /// Transactions the LMF aborted (§5.5.3).
    aborted: Vec<u8>,
    /// How many messages this endpoint has sent.
    sent: u32,
}

impl SlppEndpoint {
    /// An endpoint that sends `application_layer_id` as its own.
    pub fn new(application_layer_id: Option<Vec<u8>>) -> Self {
        Self {
            application_layer_id,
            ..Self::default()
        }
    }

    /// Set this UE's application layer ID.
    pub fn set_application_layer_id(&mut self, id: Option<Vec<u8>>) {
        self.application_layer_id = id;
    }

    /// The last assistance data the LMF provided.
    pub fn assistance_data(&self) -> Option<&ProvideAssistanceData_IEs> {
        self.assistance_data.as_ref()
    }

    /// Whether the LMF aborted transaction `tid`.
    pub fn was_aborted(&self, tid: u8) -> bool {
        self.aborted.contains(&tid)
    }

    /// How many SLPP messages this endpoint has sent.
    pub fn messages_sent(&self) -> u32 {
        self.sent
    }

    /// Handle one SLPP container from a DL NAS TRANSPORT and return the UL NAS
    /// TRANSPORT (container type 12) that answers it, if any.
    ///
    /// `additional_info` is the DL message's Additional information IE, the
    /// LCS correlation identifier the AMF set (TS 24.501 §5.4.5.3.2 c1). It is
    /// echoed as the reply's routing information (§5.4.5.2.2 c1), so the AMF
    /// can route the reply to the LMF that asked.
    pub fn handle_nas_container(
        &mut self,
        container: &[u8],
        additional_info: Option<&[u8]>,
        measurements: &[SlRttMeasurement],
    ) -> SlppUplink {
        match self.handle_downlink(container, measurements) {
            Some(slpp) => {
                let mut pdu = Vec::new();
                let mut ul = UlNasTransport::new(PayloadContainerType::SlppMessage, slpp);
                ul.additional_info = additional_info.map(<[u8]>::to_vec);
                ul.encode(&mut pdu);
                info!(
                    "SLPP: sending UL NAS TRANSPORT (container type 12), len={}",
                    pdu.len()
                );
                SlppUplink::Send(pdu)
            }
            None => SlppUplink::Nothing,
        }
    }

    /// Handle one SLPP message and return the SLPP reply, if any.
    pub fn handle_downlink(
        &mut self,
        bytes: &[u8],
        measurements: &[SlRttMeasurement],
    ) -> Option<Vec<u8>> {
        let msg = match decode(bytes) {
            Ok(msg) => msg,
            Err(e) => {
                // §5.4.3: the header did not decode, so neither the transaction
                // nor the message type is known, and the Error carries neither.
                warn!("SLPP: undecodable message ({e:?}); answering slppMessageHeaderError");
                return self.send(self.error(None, ERROR_CAUSE_HEADER));
            }
        };
        let tid = msg.transaction_id.as_ref().map(|t| t.0);

        // §4.3.3: acknowledge every message that asks, duplicates included.
        let ack_for = msg
            .acknowledgement
            .as_ref()
            .filter(|a| a.ack_requested.0)
            .and(msg.sequence_number.as_ref())
            .map(|s| s.0);

        // §4.3.2: a repeat of the last received sequence number is discarded.
        if let Some(seq) = msg.sequence_number.as_ref().map(|s| s.0) {
            if self.last_received_sequence_number == Some(seq) {
                debug!("SLPP: duplicate sequence number {seq} discarded");
                return ack_for.and_then(|n| self.send(self.ack_only(n)));
            }
            self.last_received_sequence_number = Some(seq);
        }

        let reply = match msg.slpp_message_body {
            Some(SLPP_MessageBody::C1(body)) => self.handle_body(body, tid, measurements),
            Some(SLPP_MessageBody::MessageClassExtension(_)) => {
                Some(self.error(tid, ERROR_CAUSE_BODY))
            }
            None => None,
        };
        match (reply, ack_for) {
            (Some(mut reply), Some(n)) => {
                reply.acknowledgement = Some(Acknowledgement {
                    ack_requested: AcknowledgementAckRequested(false),
                    ack_indicator: Some(SequenceNumber(n)),
                });
                self.send(reply)
            }
            (Some(reply), None) => self.send(reply),
            (None, Some(n)) => self.send(self.ack_only(n)),
            (None, None) => None,
        }
    }

    fn handle_body(
        &mut self,
        body: SLPP_MessageBody_c1,
        tid: Option<u8>,
        measurements: &[SlRttMeasurement],
    ) -> Option<SLPP_Message> {
        match body {
            SLPP_MessageBody_c1::RequestCapabilities(req) => Some(match req.critical_extensions {
                RequestCapabilitiesCriticalExtensions::RequestCapabilities(ies) => {
                    self.provide_capabilities(tid, &ies)
                }
                RequestCapabilitiesCriticalExtensions::CriticalExtensionsFuture(_) => {
                    self.error(tid, ERROR_CAUSE_BODY)
                }
            }),
            SLPP_MessageBody_c1::RequestLocationInformation(req) => {
                Some(match req.critical_extensions {
                    RequestLocationInformationCriticalExtensions::RequestLocationInformation(
                        ies,
                    ) => self.provide_location_information(tid, &ies, measurements),
                    RequestLocationInformationCriticalExtensions::CriticalExtensionsFuture(_) => {
                        self.error(tid, ERROR_CAUSE_BODY)
                    }
                })
            }
            SLPP_MessageBody_c1::ProvideAssistanceData(pad) => match pad.critical_extensions {
                ProvideAssistanceDataCriticalExtensions::ProvideAssistanceData(ies) => {
                    info!("SLPP: assistance data received and stored");
                    self.assistance_data = Some(ies);
                    None
                }
                ProvideAssistanceDataCriticalExtensions::CriticalExtensionsFuture(_) => {
                    Some(self.error(tid, ERROR_CAUSE_BODY))
                }
            },
            SLPP_MessageBody_c1::RequestAssistanceData(_) => {
                // The UE is the receiver of assistance data, never its server.
                warn!("SLPP: RequestAssistanceData received; the UE serves none");
                Some(self.error(tid, ERROR_CAUSE_UNDEFINED))
            }
            SLPP_MessageBody_c1::Abort(_) => {
                if let Some(tid) = tid {
                    info!("SLPP: transaction {tid} aborted by the LMF");
                    self.aborted.push(tid);
                }
                None
            }
            // §5.4.2: an Error is never answered with an Error.
            SLPP_MessageBody_c1::Error(_) => {
                warn!("SLPP: Error received for transaction {tid:?}");
                None
            }
            // A message this UE never receives (ProvideCapabilities,
            // ProvideLocationInformation) or a spare: a body the UE cannot act on.
            _ => Some(self.error(tid, ERROR_CAUSE_BODY)),
        }
    }

    /// §5.1.5: capabilities for every requested method this UE supports.
    fn provide_capabilities(&self, tid: Option<u8>, req: &RequestCapabilities_IEs) -> SLPP_Message {
        let sl_rtt = req.sl_rtt_request_capabilities.as_ref().and_then(|_| {
            encode_ie(&sl_rtt_capabilities())
                .ok()
                .map(ProvideCapabilities_IEsSl_RTT_ProvideCapabilities)
        });
        let ies = ProvideCapabilities_IEs {
            common_i_es_provide_capabilities: None,
            common_sl_prs_methods_i_es_provide_capabilities: None,
            sl_ao_a_provide_capabilities: None,
            sl_rtt_provide_capabilities: sl_rtt,
            sl_tdoa_provide_capabilities: None,
            sl_toa_provide_capabilities: None,
            late_non_critical_extension: None,
            non_critical_extension: None,
        };
        self.message(
            tid,
            SLPP_MessageBody_c1::ProvideCapabilities(ProvideCapabilities {
                critical_extensions: ProvideCapabilitiesCriticalExtensions::ProvideCapabilities(
                    ies,
                ),
            }),
        )
    }

    /// §5.3 and §5.4.3: the SL-RTT measurements, or what cannot be provided.
    fn provide_location_information(
        &self,
        tid: Option<u8>,
        req: &RequestLocationInformation_IEs,
        measurements: &[SlRttMeasurement],
    ) -> SLPP_Message {
        let mut ies = ProvideLocationInformation_IEs {
            common_i_es_provide_location_information: None,
            common_sl_prs_methods_i_es_provide_location_information: None,
            sl_ao_a_provide_location_information: None,
            sl_rtt_provide_location_information: None,
            sl_tdoa_provide_location_information: None,
            sl_toa_provide_location_information: None,
            late_non_critical_extension: None,
            non_critical_extension: None,
        };
        let wants_measurements = req
            .common_i_es_request_location_information
            .as_ref()
            .and_then(|c| decode_ie::<CommonIEsRequestLocationInformation>(&c.0).ok())
            .is_none_or(|c| MEASUREMENT_TYPES.contains(&c.location_information_type.0));
        let sl_rtt_requested = req.sl_rtt_request_location_information.as_ref();
        if let (Some(rtt), true) = (sl_rtt_requested, wants_measurements) {
            if decode_ie::<SL_RTT_RequestLocationInformation>(&rtt.0).is_err() {
                return self.error(tid, ERROR_CAUSE_BODY);
            }
            let report = sl_rtt_report(measurements);
            ies.sl_rtt_provide_location_information = encode_ie(&report)
                .ok()
                .map(ProvideLocationInformation_IEsSl_RTT_ProvideLocationInformation);
        } else {
            // Only SL-RTT measurements are supported (UE-assisted).
            let common = CommonIEsProvideLocationInformation {
                location_estimate: None,
                range_and_or_direction: None,
                velocity_estimate: None,
                relative_location_estimate: None,
                location_error: Some(LocationError {
                    locationfailurecause: LocationFailureCause(REQUESTED_METHOD_NOT_SUPPORTED),
                }),
                extensions: CommonIEsProvideLocationInformation_ExtAdditions {
                    location_time_stamp: None,
                },
            };
            ies.common_i_es_provide_location_information = encode_ie(&common)
                .ok()
                .map(ProvideLocationInformation_IEsCommonIEsProvideLocationInformation);
        }
        self.message(
            tid,
            SLPP_MessageBody_c1::ProvideLocationInformation(ProvideLocationInformation {
                critical_extensions:
                    ProvideLocationInformationCriticalExtensions::ProvideLocationInformation(ies),
            }),
        )
    }

    fn error(&self, tid: Option<u8>, cause: u8) -> SLPP_Message {
        let mut msg = self.message(
            tid,
            SLPP_MessageBody_c1::Error(Error::CriticalExtensions(Error_criticalExtensions::Error(
                Error_IEs {
                    common_i_es_error: Some(CommonIEsError {
                        error_cause: CommonIEsErrorErrorCause(cause),
                    }),
                    late_non_critical_extension: None,
                    non_critical_extension: None,
                },
            ))),
        );
        if tid.is_none() {
            msg.transaction_id = None;
        }
        msg
    }

    fn ack_only(&self, acknowledged: u8) -> SLPP_Message {
        SLPP_Message {
            application_layer_id: self
                .application_layer_id
                .clone()
                .map(SLPP_MessageApplicationLayerID),
            transaction_id: None,
            end_transaction: SLPP_MessageEndTransaction(false),
            sequence_number: None,
            session_id: None,
            acknowledgement: Some(Acknowledgement {
                ack_requested: AcknowledgementAckRequested(false),
                ack_indicator: Some(SequenceNumber(acknowledged)),
            }),
            slpp_message_body: None,
            non_critical_extension: None,
        }
    }

    /// A reply that ends the transaction `tid` (§5.1.2, §5.3).
    fn message(&self, tid: Option<u8>, body: SLPP_MessageBody_c1) -> SLPP_Message {
        SLPP_Message {
            application_layer_id: self
                .application_layer_id
                .clone()
                .map(SLPP_MessageApplicationLayerID),
            transaction_id: Some(SLPP_MessageTransactionID(tid.unwrap_or(0))),
            end_transaction: SLPP_MessageEndTransaction(true),
            sequence_number: None,
            session_id: None,
            acknowledgement: None,
            slpp_message_body: Some(SLPP_MessageBody::C1(body)),
            non_critical_extension: None,
        }
    }

    /// Stamp the sequence number (§4.3.2) and encode.
    fn send(&mut self, mut msg: SLPP_Message) -> Option<Vec<u8>> {
        msg.sequence_number = Some(SequenceNumber(self.next_sequence_number));
        match encode(&msg) {
            Ok(bytes) => {
                self.next_sequence_number = self.next_sequence_number.wrapping_add(1);
                self.sent += 1;
                Some(bytes)
            }
            Err(e) => {
                warn!("SLPP: reply could not be encoded: {e:?}");
                None
            }
        }
    }
}

/// This UE's SL-RTT capabilities: UE-assisted only, one band, and one Rx-Tx
/// report per SL-PRS reception without a Tx time stamp (n1).
fn sl_rtt_capabilities() -> SL_RTT_ProvideCapabilities {
    let mut modes = bitvec::vec::BitVec::<u8, bitvec::order::Msb0>::new();
    modes.push(false); // sl-TargetUE-Based (0)
    modes.push(true); // sl-TargetUE-Assisted (1)
    SL_RTT_ProvideCapabilities {
        positioning_modes: PositioningModes(modes),
        ten_ms_unit_response_time: None,
        periodical_reporting: None,
        scheduled_location_request_supported: None,
        location_coordinate_types: None,
        velocity_types: None,
        sl_rtt_capability_band_list: SL_RTT_ProvideCapabilitiesSl_RTT_CapabilityBandList(vec![
            SL_RTT_CapabilityPerBand {
                sl_prs_rx_tx_time_diff_without_tx_time_stamp: Some(
                    SL_RTT_CapabilityPerBandSl_PRS_RxTxTimeDiffWithoutTxTimeStamp(0),
                ),
                sl_prs_rx_tx_time_diff_with_tx_time_stamp: None,
                dummy: None,
                extensions: SL_RTT_CapabilityPerBand_ExtAdditions {
                    measurements_for_multiple_arp_i_ds_rx: None,
                    freq_band_indicator_nr: None,
                },
            },
        ]),
    }
}

/// The SL-RTT report for `measurements`: one per-ARP list of one element per
/// peer, at most maxNrOfUEs (256), or `notAllRequestedMeasurementsPossible`
/// when there is nothing to report.
fn sl_rtt_report(measurements: &[SlRttMeasurement]) -> SL_RTT_ProvideLocationInformation {
    const MAX_NR_OF_UES: usize = 256;
    if measurements.len() > MAX_NR_OF_UES {
        warn!(
            "SLPP: {} peers measured; reporting the first {MAX_NR_OF_UES} (maxNrOfUEs)",
            measurements.len()
        );
    }
    let list: Vec<SL_RTT_MeasElementPerARP_ID_Rx> = measurements
        .iter()
        .take(MAX_NR_OF_UES)
        .filter_map(|m| {
            let value = rx_tx_report_value_k2(m.rx_tx_time_difference_s)?;
            Some(SL_RTT_MeasElementPerARP_ID_Rx(vec![SL_RTT_MeasElement {
                application_layer_id: Some(SL_RTT_MeasElementApplicationLayerID(
                    m.peer_application_layer_id.clone(),
                )),
                los_nlos_indicator: None,
                sl_pos_arp_id_rx: None,
                sl_prs_rx_tx_time_diff_meas_result:
                    SL_PRS_RxTxTimeDiffMeasResult::Single_SL_PRS_RxTxTimeDiff(
                        SL_PRS_RxTxTimeDiffResult {
                            sl_prs_resource_id: None,
                            sl_prs_rx_tx_time_diff_result: Some(
                                SL_PRS_RxTxTimeDiffResultSl_PRS_RxTxTimeDiffResult::K2(
                                    SL_PRS_RxTxTimeDiffResultSl_PRS_RxTxTimeDiffResult_k2(value),
                                ),
                            ),
                            sl_prs_rsrp_result: None,
                            sl_prs_rsrpp_result: None,
                            sl_rtt_additional_path_list: None,
                            sl_time_stamp: None,
                            sl_timing_quality: None,
                            tx_time_info: None,
                        },
                    ),
            }]))
        })
        .collect();
    if list.is_empty() {
        return SL_RTT_ProvideLocationInformation {
            sl_rtt_signal_measurement_information: None,
            sl_rtt_error: Some(SL_RTT_LocationInformationError(
                SL_RTT_NOT_ALL_REQUESTED_MEASUREMENTS_POSSIBLE,
            )),
        };
    }
    SL_RTT_ProvideLocationInformation {
        sl_rtt_signal_measurement_information: Some(SL_RTT_SignalMeasurementInformation {
            sl_rtt_meas_list: SL_RTT_SignalMeasurementInformationSl_RTT_MeasList(list),
        }),
        sl_rtt_error: None,
    }
}

#[cfg(test)]
mod tests;
