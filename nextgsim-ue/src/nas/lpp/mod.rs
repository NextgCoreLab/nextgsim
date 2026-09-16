//! LPP positioning procedures at the UE (3GPP TS 37.355), carried in the LPP
//! payload container of UL/DL NAS TRANSPORT (TS 24.501 §5.4.5.3).
//!
//! Before this (issue #46) a DL NAS Transport carrying an LPP container fell
//! through to the SM orchestrator, which owns only N1 SM information and returned
//! an empty result — so the container was dropped with a `debug!` line and the LMF's
//! transaction timed out with no reply. Nothing in the tree produced a
//! `ProvideCapabilities` or a `ProvideLocationInformation`.
//!
//! ## What this answers
//!
//! - **`RequestCapabilities`** (§5.1.3) → `ProvideCapabilities` naming the E-CID
//!   measurements this UE can actually report.
//! - **`RequestLocationInformation`** (§5.3.3) → `ProvideLocationInformation`
//!   carrying an E-CID measurement report.
//!
//! Both echo the incoming `LPP-TransactionID` and set `endTransaction`, so the
//! server can match the reply to its request and close the transaction.
//!
//! ## What it refuses rather than answers
//!
//! A request for A-GNSS, OTDOA, NR-DL-TDOA or Multi-RTT is **refused at the codec**
//! rather than answered with E-CID: answering a method the server did not ask for
//! looks like a successful positioning fix and is not one. The refusal is logged
//! with its reason.
//!
//! ## What this UE can honestly measure
//!
//! Exactly one of E-CID's three quantities: **RSRP**. The UE's measurement path
//! (`MeasurementManager::update_measurement`) is fed one dBm level per cell by RLS
//! and populates nothing else — `CellMeasResult::rsrq` is declared and never
//! written, and there is no downlink-receive-to-uplink-transmit timing anywhere in
//! the RLS to derive a UE Rx-Tx difference from. So RSRQ and UE Rx-Tx are
//! advertised as **unsupported** and reported as **absent**, even when the server
//! asks for them. The plumbing carries them as `Option`s so that a later RSRQ
//! measurement needs no change here.

mod message;
mod uper;

pub use message::{
    EcidMeasurementBits, EcidSignalMeasurementInformation, Initiator, LppBody, LppMessage,
    LppTransactionId, MeasuredResultsElement, SidelinkRangingMethod, SidelinkRangingReport,
    SidelinkRangingResult,
};
pub use uper::{UperError, UperReader, UperResult, UperWriter};

use nextgsim_nas::ies::ie1::PayloadContainerType;
use nextgsim_nas::messages::mm::UlNasTransport;
use tracing::{debug, info, warn};

/// The serving-cell measurements the UE can put in an E-CID report.
///
/// Supplied by the caller rather than read here, because the LPP layer has no radio
/// of its own: these are values the RRC layer holds, and inventing them would make
/// a report that decodes correctly and means nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServingCellMeasurements {
    /// Physical cell identity of the serving cell.
    ///
    /// The simulator's RLS cell id doubles as the PCI throughout this tree —
    /// `MeasurementManager::update_measurement` stores `pci: cell_id as u32`, and
    /// `rrc::conditional_handover::candidate_cell_id` relies on the same identity.
    pub phys_cell_id: u16,
    /// `arfcnEUTRA`, which TS 37.355 makes a **mandatory** member of
    /// `MeasuredResultsElement`.
    ///
    /// Configured, not measured: this is an NR UE and it has no E-UTRA carrier to
    /// report. See `UeConfig::lpp_arfcn_eutra`.
    pub arfcn: u32,
    /// System frame number, when a frame clock is available.
    pub system_frame_number: Option<u16>,
    /// RSRP as the TS 36.133 §9.1.4 report mapping value (0..97).
    pub rsrp_result: Option<u8>,
    /// RSRQ as the TS 36.133 §9.1.7 report mapping value (0..34).
    ///
    /// Always `None` today: nothing in the UE measures RSRQ (see the module doc).
    pub rsrq_result: Option<u8>,
}

/// RSRP in dBm as the TS 36.133 §9.1.4 `RSRP_LEV` report value (0..97).
///
/// `RSRP_00` is "below -140 dBm", `RSRP_01` is `-140 <= RSRP < -139`, and so on up
/// to `RSRP_97` for "at or above -44 dBm". So the value is `1 + floor(dBm + 140)`,
/// saturated at both ends.
///
/// Saturated rather than refused because a level outside the reportable range is a
/// real measurement of a real signal; 0 and 97 are the clause's own words for "off
/// the bottom" and "off the top", not substitutes for an unknown.
pub fn rsrp_report_value(dbm: i32) -> u8 {
    if dbm < -140 {
        return 0;
    }
    (dbm + 141).clamp(0, 97) as u8
}

/// RSRQ in dB as the TS 36.133 §9.1.7 `RSRQ_LEV` report value (0..34).
///
/// 0.5 dB steps: `RSRQ_00` is "below -19.5 dB", `RSRQ_01` is `-19.5 <= RSRQ < -19`,
/// up to `RSRQ_34` for "at or above -3 dB".
///
/// Unused today — nothing measures RSRQ — and kept beside its RSRP counterpart so
/// that a future RSRQ source has the mapping already written and tested rather than
/// inventing a second one.
pub fn rsrq_report_value(db: f32) -> u8 {
    if db < -19.5 {
        return 0;
    }
    if db >= -3.0 {
        return 34;
    }
    (1 + ((db + 19.5) * 2.0).floor() as i32).clamp(0, 34) as u8
}

/// What the UE should do with a decoded LPP message: the **LPP** level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LppReaction {
    /// Send this LPP PDU back in an UL NAS TRANSPORT LPP container.
    Reply(Vec<u8>),
    /// Nothing to send. The message was understood and needed no answer, or it
    /// asked for something this UE cannot honestly provide.
    Nothing,
}

/// What the UE should send after an LPP container arrived in DL NAS TRANSPORT: the
/// **NAS** level.
///
/// Distinct from [`LppReaction`] because the payloads are different things — one is
/// an LPP PDU, the other the UL NAS TRANSPORT that carries it — and a single type
/// for both would let a caller send the inner PDU as though it were a NAS message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LppUplink {
    /// A complete plain 5GMM UL NAS TRANSPORT PDU, ready for integrity protection.
    Send(Vec<u8>),
    /// Nothing to send.
    Nothing,
}

/// The UE's LPP endpoint.
#[derive(Debug, Clone)]
pub struct LppEndpoint {
    /// The E-CID measurements this UE reports as supported. See the module doc for
    /// why RSRP is the only one.
    supported: EcidMeasurementBits,
    /// How many replies have been sent, so a caller can tell "answered" from
    /// "nothing arrived".
    replies_sent: u32,
}

impl Default for LppEndpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl LppEndpoint {
    /// A new endpoint advertising the measurements this UE can make.
    pub fn new() -> Self {
        Self {
            supported: EcidMeasurementBits {
                rsrp: true,
                // Never advertised: see the module doc. Claiming either would make
                // the LMF request a measurement that comes back absent.
                rsrq: false,
                ue_rx_tx: false,
            },
            replies_sent: 0,
        }
    }

    /// The E-CID measurements this UE advertises.
    pub fn supported(&self) -> EcidMeasurementBits {
        self.supported
    }

    /// How many LPP replies have been produced.
    pub fn replies_sent(&self) -> u32 {
        self.replies_sent
    }

    /// Handle an LPP PDU from a DL NAS TRANSPORT payload container.
    ///
    /// `measurements` are the serving cell's current values, used only for a
    /// location-information request.
    pub fn handle_downlink(
        &mut self,
        pdu: &[u8],
        measurements: &ServingCellMeasurements,
        sidelink_ranging: &[SidelinkRangingResult],
    ) -> LppReaction {
        let request = match LppMessage::decode(pdu) {
            Ok(message) => message,
            Err(e) => {
                // Not answered with an LPP Error: this codec does not produce one,
                // and a malformed reply is worse than silence. Logged at warn so the
                // reason is visible rather than the transaction merely timing out.
                warn!("LPP message from the network could not be decoded ({e}); no reply sent");
                return LppReaction::Nothing;
            }
        };

        let Some(body) = request.body else {
            debug!("LPP message carried no body; nothing to answer");
            return LppReaction::Nothing;
        };

        let reply_body = match body {
            LppBody::RequestCapabilities { ecid_requested } => {
                info!(
                    "LPP RequestCapabilities (E-CID asked for: {ecid_requested}); \
                     answering with rsrp={} rsrq={} ueRxTx={}",
                    self.supported.rsrp, self.supported.rsrq, self.supported.ue_rx_tx
                );
                LppBody::ProvideCapabilities {
                    ecid_supported: self.supported,
                }
            }
            LppBody::RequestLocationInformation { requested } => {
                // Report only what was BOTH asked for and supported. A measurement
                // the server did not request is not part of the answer, and one this
                // UE cannot make would have to be invented.
                let element = MeasuredResultsElement {
                    phys_cell_id: measurements.phys_cell_id,
                    arfcn: measurements.arfcn,
                    system_frame_number: measurements.system_frame_number,
                    rsrp_result: (requested.rsrp && self.supported.rsrp)
                        .then_some(measurements.rsrp_result)
                        .flatten(),
                    rsrq_result: (requested.rsrq && self.supported.rsrq)
                        .then_some(measurements.rsrq_result)
                        .flatten(),
                    // Never reported: see the module doc.
                    ue_rx_tx_time_diff: None,
                };
                info!(
                    "LPP RequestLocationInformation (rsrp={} rsrq={} ueRxTx={}); reporting PCI {} \
                     rsrp={:?} rsrq={:?}",
                    requested.rsrp,
                    requested.rsrq,
                    requested.ue_rx_tx,
                    element.phys_cell_id,
                    element.rsrp_result,
                    element.rsrq_result
                );
                // The sidelink ranging report rides along when this UE holds one
                // (TS 23.586 §5.3.3, issue #137). `MeasuredResultsList` is
                // SIZE(1..32), and so is the ranging list, so an EMPTY set of results
                // must encode as an ABSENT report rather than an empty list -- an
                // empty one encodes and then fails to decode.
                let sidelink = match SidelinkRangingReport::fitting_an_addition(sidelink_ranging) {
                    Some((report, 0)) => {
                        info!(
                            "LPP: reporting {} sidelink range(s) to the LMF",
                            report.results.len()
                        );
                        Some(report)
                    }
                    Some((report, dropped)) => {
                        // Named rather than silent: a truncated report looks exactly
                        // like a complete one at the LMF.
                        warn!(
                            "LPP: reporting {} sidelink range(s) and DROPPING {}; an \
                             extension addition carries at most {} octets",
                            report.results.len(),
                            dropped,
                            SidelinkRangingReport::MAX_ADDITION_OCTETS
                        );
                        Some(report)
                    }
                    None => None,
                };
                LppBody::ProvideLocationInformation {
                    measurements: EcidSignalMeasurementInformation {
                        // The serving cell appears as the PRIMARY cell and as the
                        // single list element, because `MeasuredResultsList` is
                        // SIZE(1..32) and cannot be empty -- so a UE with no
                        // neighbour measurements still has to put something there,
                        // and its own serving cell is the only honest candidate.
                        primary_cell: Some(element.clone()),
                        measured_results: vec![element],
                    },
                    sidelink,
                }
            }
            LppBody::ProvideCapabilities { .. } | LppBody::ProvideLocationInformation { .. } => {
                // A server does not send these to a UE. Understood and not answered.
                debug!("LPP provide-message received from the network; nothing to answer");
                return LppReaction::Nothing;
            }
        };

        let reply = LppMessage {
            // Echo the server's transaction id so it can match the reply; §5.1.3
            // and §5.3.3 both require the response to carry the request's id.
            transaction_id: request.transaction_id,
            end_transaction: true,
            body: Some(reply_body),
        };

        match reply.encode() {
            Ok(bytes) => {
                self.replies_sent += 1;
                LppReaction::Reply(bytes)
            }
            Err(e) => {
                warn!("LPP reply could not be encoded ({e}); no reply sent");
                LppReaction::Nothing
            }
        }
    }

    /// Handle the LPP payload container of a DL NAS TRANSPORT and produce the UL NAS
    /// TRANSPORT that answers it (TS 24.501 §5.4.5.3).
    ///
    /// The NAS framing lives here rather than at the call site so that the container
    /// type and the payload cannot drift apart: an LPP PDU announced as anything but
    /// `PayloadContainerType::LppMessage` would be routed to the wrong handler at the
    /// AMF.
    pub fn handle_nas_container(
        &mut self,
        container: &[u8],
        measurements: &ServingCellMeasurements,
        sidelink_ranging: &[SidelinkRangingResult],
    ) -> LppUplink {
        match self.handle_downlink(container, measurements, sidelink_ranging) {
            LppReaction::Reply(lpp_pdu) => {
                let mut pdu = Vec::new();
                UlNasTransport::new(PayloadContainerType::LppMessage, lpp_pdu).encode(&mut pdu);
                info!(
                    "LPP: sending UL NAS TRANSPORT (container type 3), len={}",
                    pdu.len()
                );
                LppUplink::Send(pdu)
            }
            LppReaction::Nothing => LppUplink::Nothing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurements() -> ServingCellMeasurements {
        ServingCellMeasurements {
            phys_cell_id: 321,
            arfcn: 1850,
            system_frame_number: Some(42),
            rsrp_result: Some(70),
            // Present in the fixture even though the UE never produces one, so the
            // "requested but unsupported" tests below cannot pass merely because
            // there was nothing to report.
            rsrq_result: Some(20),
        }
    }

    /// An LMF-shaped request, encoded exactly as the peer would.
    fn request(body: LppBody, transaction: u8) -> Vec<u8> {
        LppMessage {
            transaction_id: Some(LppTransactionId {
                initiator: Initiator::LocationServer,
                transaction_number: transaction,
            }),
            end_transaction: false,
            body: Some(body),
        }
        .encode()
        .expect("a request encodes")
    }

    fn asked_for(rsrp: bool, rsrq: bool, ue_rx_tx: bool) -> LppBody {
        LppBody::RequestLocationInformation {
            requested: EcidMeasurementBits {
                rsrp,
                rsrq,
                ue_rx_tx,
            },
        }
    }

    // ---- the report-value mappings ----

    #[test]
    fn rsrp_maps_to_the_ts_36_133_report_value() {
        // The clause's own boundaries: RSRP_00 below -140, RSRP_01 at -140,
        // RSRP_97 at or above -44.
        assert_eq!(rsrp_report_value(-141), 0);
        assert_eq!(rsrp_report_value(-200), 0);
        assert_eq!(rsrp_report_value(-140), 1);
        assert_eq!(rsrp_report_value(-139), 2);
        assert_eq!(rsrp_report_value(-45), 96);
        assert_eq!(rsrp_report_value(-44), 97);
        assert_eq!(rsrp_report_value(0), 97, "saturated, not wrapped");
        // A typical camped level, and the value the report fixture uses.
        assert_eq!(rsrp_report_value(-71), 70);
    }

    #[test]
    fn every_rsrp_report_value_is_inside_the_asn1_constraint() {
        // The mapping feeds `write_constrained(.., 0, 97)` directly, so a value
        // outside the range would fail to encode rather than be clamped.
        for dbm in -250..=50 {
            let value = rsrp_report_value(dbm);
            assert!(value <= 97, "{dbm} dBm mapped to {value}");
        }
    }

    #[test]
    fn rsrq_maps_to_the_ts_36_133_report_value_in_half_db_steps() {
        assert_eq!(rsrq_report_value(-20.0), 0);
        assert_eq!(rsrq_report_value(-19.5), 1);
        assert_eq!(rsrq_report_value(-19.0), 2);
        assert_eq!(rsrq_report_value(-18.75), 2, "still inside RSRQ_02");
        assert_eq!(rsrq_report_value(-3.5), 33);
        assert_eq!(rsrq_report_value(-3.0), 34);
        assert_eq!(rsrq_report_value(0.0), 34);
        for tenths in -300..=100 {
            let value = rsrq_report_value(tenths as f32 / 10.0);
            assert!(value <= 34, "{tenths} tenths dB mapped to {value}");
        }
    }

    // ---- the envelope ----

    #[test]
    fn the_envelope_round_trips_with_and_without_a_transaction_id() {
        for transaction_id in [
            Some(LppTransactionId {
                initiator: Initiator::TargetDevice,
                transaction_number: 255,
            }),
            None,
        ] {
            for end_transaction in [true, false] {
                let message = LppMessage {
                    transaction_id,
                    end_transaction,
                    body: None,
                };
                let bytes = message.encode().expect("encodes");
                assert_eq!(LppMessage::decode(&bytes).expect("decodes"), message);
            }
        }
    }

    #[test]
    fn end_transaction_is_written_after_the_preamble_not_inside_it() {
        // The one thing about this envelope that is easy to get wrong: the mandatory
        // BOOLEAN sits between transactionID and sequenceNumber, so with no optional
        // fields at all the whole message is FOUR presence bits plus one flag bit.
        let message = LppMessage {
            transaction_id: None,
            end_transaction: true,
            body: None,
        };
        // 0000 (four absent) then 1 (endTransaction), padded: 0b0000_1000.
        assert_eq!(message.encode().expect("encodes"), vec![0b0000_1000]);

        // And with the flag clear the byte is all zeros -- one octet, not none
        // (X.691 11.1.2).
        let cleared = LppMessage {
            end_transaction: false,
            ..message
        };
        assert_eq!(cleared.encode().expect("encodes"), vec![0b0000_0000]);
    }

    #[test]
    fn every_body_round_trips_through_the_wire_form() {
        // The four bodies this codec produces, each back out of its own bytes: a
        // reply is only useful if the peer's decoder finds what the encoder meant.
        let bodies = [
            LppBody::RequestCapabilities {
                ecid_requested: true,
            },
            LppBody::RequestCapabilities {
                ecid_requested: false,
            },
            LppBody::ProvideCapabilities {
                ecid_supported: EcidMeasurementBits {
                    rsrp: true,
                    rsrq: false,
                    ue_rx_tx: true,
                },
            },
            asked_for(true, true, true),
            LppBody::ProvideLocationInformation {
                measurements: EcidSignalMeasurementInformation {
                    primary_cell: None,
                    measured_results: vec![MeasuredResultsElement {
                        phys_cell_id: 503,
                        arfcn: 65_535,
                        system_frame_number: Some(1023),
                        rsrp_result: Some(97),
                        rsrq_result: Some(34),
                        ue_rx_tx_time_diff: Some(4095),
                    }],
                },
                sidelink: None,
            },
        ];
        for body in bodies {
            let message = LppMessage {
                transaction_id: Some(LppTransactionId {
                    initiator: Initiator::LocationServer,
                    transaction_number: 17,
                }),
                end_transaction: true,
                body: Some(body.clone()),
            };
            let bytes = message.encode().expect("encodes");
            assert_eq!(
                LppMessage::decode(&bytes).expect("decodes").body,
                Some(body)
            );
        }
    }

    #[test]
    fn a_full_list_of_thirty_two_elements_encodes_and_an_empty_one_is_refused() {
        // MeasuredResultsList is SIZE(1..32) at both ends.
        let element = MeasuredResultsElement {
            phys_cell_id: 1,
            arfcn: 1850,
            system_frame_number: None,
            rsrp_result: Some(50),
            rsrq_result: None,
            ue_rx_tx_time_diff: None,
        };
        for count in [1usize, 32] {
            let body = LppBody::ProvideLocationInformation {
                measurements: EcidSignalMeasurementInformation {
                    primary_cell: None,
                    measured_results: vec![element.clone(); count],
                },
                sidelink: None,
            };
            let message = LppMessage {
                transaction_id: None,
                end_transaction: true,
                body: Some(body.clone()),
            };
            let bytes = message.encode().expect("encodes");
            assert_eq!(
                LppMessage::decode(&bytes).expect("decodes").body,
                Some(body),
                "{count} element(s)"
            );
        }
        for count in [0usize, 33] {
            let message = LppMessage {
                transaction_id: None,
                end_transaction: true,
                body: Some(LppBody::ProvideLocationInformation {
                    measurements: EcidSignalMeasurementInformation {
                        primary_cell: None,
                        measured_results: vec![element.clone(); count],
                    },
                    sidelink: None,
                }),
            };
            assert_eq!(
                message.encode(),
                Err(UperError::InvalidLength(count)),
                "{count} element(s) is outside SIZE(1..32)"
            );
        }
    }

    // ---- byte agreement with the peer ----

    /// The LMF's own hand-derived reference vector, copied from
    /// `nextgcore-asn1c::lpp::message`'s `test_request_location_information_reference_hex`
    /// together with its bit-by-bit derivation.
    ///
    /// `LPP-Message { transactionID { locationServer, 0 }, endTransaction TRUE,
    /// body c1: requestLocationInformation: r9 { ecid { requestedMeasurements
    /// '11000'B } } }` — 39 bits padded to 5 octets.
    const LMF_REQUEST_LOCATION_INFORMATION: &[u8] = &[0x90, 0x01, 0x20, 0x09, 0x30];

    #[test]
    fn the_peers_reference_request_decodes_to_what_it_says_it_is() {
        // This is the ONLY evidence available that the two codecs agree: they are
        // separate products, so nothing links them and a round trip through this
        // module alone would pass however wrong both were.
        let decoded = LppMessage::decode(LMF_REQUEST_LOCATION_INFORMATION).expect("decodes");
        assert_eq!(
            decoded.transaction_id,
            Some(LppTransactionId {
                initiator: Initiator::LocationServer,
                transaction_number: 0,
            })
        );
        assert!(decoded.end_transaction);
        assert_eq!(
            decoded.body,
            // '11000'B is rsrpReq(0) + rsrqReq(1). The peer sends five bits; this
            // codec models three, and the two are the same named-bit set because a
            // bit past the string's length is absent, which for a NamedBitList means
            // clear.
            Some(asked_for(true, true, false)),
        );
    }

    #[test]
    fn the_peers_reference_request_gets_a_reply_with_the_hand_derived_bytes() {
        // Derived the same way the peer's vector was, bit by bit, because "it round
        // trips" cannot catch a layout both sides get wrong:
        //   preamble (non-ext, [tid,seq,ack,body]):            1 0 0 1
        //   transactionID: SEQ ext 0, initiator ext 0 + 0,
        //                  transactionNumber 0 (8 bits):       0 0 0  0000 0000
        //   endTransaction TRUE:                               1
        //   body: outer CHOICE c1 (0 of 2):                    0
        //         c1 CHOICE provideLocationInformation (5/16): 0 1 0 1
        //         criticalExtensions c1 (0 of 2):              0
        //         c1 r9 (0 of 4):                              0 0
        //         r9-IEs ext 0 + [0,0,0,ecid=1,0]:             0  0 0 0 1 0
        //           ECID-ProvideLocationInformation ext 0
        //             + [sigMeas=1, ecid-Error=0]:             0  1 0
        //             ECID-SignalMeasurementInformation ext 0
        //               + [primaryCell=1]:                     0  1
        //               MeasuredResultsElement ext 0 + [cgi=0,
        //                 sfn=1, rsrp=1, rsrq=0, ueRxTx=0]:    0  0 1 1 0 0
        //                 physCellId 321 (9 bits):             1 0100 0001
        //                 arfcnEUTRA 1850 (16 bits):           0000 0111 0011 1010
        //                 systemFrameNumber 42 (10 bits):      00 0010 1010
        //                 rsrp-Result 70 (7 bits):             100 0110
        //               measuredResultsList length 1 (5 bits): 0 0000
        //               MeasuredResultsElement (the same one again)
        let mut endpoint = LppEndpoint::new();
        let LppReaction::Reply(reply) =
            endpoint.handle_downlink(LMF_REQUEST_LOCATION_INFORMATION, &measurements(), &[])
        else {
            panic!("the peer's own request must be answered");
        };

        let mut expected = UperWriter::new();
        expected.write_sequence_preamble(None, &[true, false, false, true]);
        expected.write_sequence_preamble(Some(false), &[]);
        expected.write_extensible_enumerated(0, 1).expect("root");
        expected.write_constrained(0, 0, 255).expect("in range");
        expected.write_bit(true); // endTransaction
        expected.write_choice_index(0, 2).expect("c1");
        expected.write_choice_index(5, 16).expect("provideLocInfo");
        expected.write_choice_index(0, 2).expect("critExt c1");
        expected.write_choice_index(0, 4).expect("r9");
        expected.write_sequence_preamble(Some(false), &[false, false, false, true, false]);
        expected.write_sequence_preamble(Some(false), &[true, false]);
        expected.write_sequence_preamble(Some(false), &[true]);
        let element = |w: &mut UperWriter| {
            w.write_sequence_preamble(Some(false), &[false, true, true, false, false]);
            w.write_constrained(321, 0, 503).expect("pci");
            w.write_constrained(1850, 0, 65_535).expect("arfcn");
            let sfn: Vec<bool> = (0..10).rev().map(|i| (42u16 >> i) & 1 == 1).collect();
            w.write_bit_string(&sfn, 10, 10).expect("sfn");
            w.write_constrained(70, 0, 97).expect("rsrp");
        };
        element(&mut expected); // primaryCellMeasuredResults
        expected.write_constrained(1, 1, 32).expect("list length");
        element(&mut expected); // the single measuredResultsList entry
        assert_eq!(reply, expected.into_bytes());

        // RSRQ was requested and is absent, so the reply is NOT symmetric with the
        // request -- which is the point of the derivation above.
        let element = match LppMessage::decode(&reply).expect("decodes").body {
            Some(LppBody::ProvideLocationInformation {
                measurements: report,
                sidelink: None,
            }) => report.primary_cell.expect("primary"),
            other => panic!("expected ProvideLocationInformation, got {other:?}"),
        };
        assert_eq!(element.rsrp_result, Some(70));
        assert_eq!(element.rsrq_result, None);
    }

    #[test]
    fn a_common_ies_capability_request_is_accepted_and_a_location_one_is_refused() {
        // nextgcore's LMF encodes `commonIEsRequestCapabilities`, an EMPTY extensible
        // SEQUENCE. Refusing it would fail a capability transfer over a field with no
        // content -- while the location-information commonIEs does carry content this
        // codec does not model, so reading past it would misalign every later field.
        let mut w = UperWriter::new();
        w.write_sequence_preamble(None, &[false, false, false, true]);
        w.write_bit(false); // endTransaction
        w.write_choice_index(0, 2).expect("c1");
        w.write_choice_index(0, 16).expect("requestCapabilities");
        w.write_choice_index(0, 2).expect("criticalExtensions c1");
        w.write_choice_index(0, 4).expect("r9");
        // Root presence bits with BOTH commonIEs and ecid set.
        w.write_sequence_preamble(Some(false), &[true, false, false, true, false]);
        w.write_sequence_preamble(Some(false), &[]); // CommonIEsRequestCapabilities
        w.write_sequence_preamble(Some(false), &[]); // ECID-RequestCapabilities
        let accepted = w.into_bytes();
        assert_eq!(
            LppMessage::decode(&accepted).expect("decodes").body,
            Some(LppBody::RequestCapabilities {
                ecid_requested: true
            })
        );

        let mut w = UperWriter::new();
        w.write_sequence_preamble(None, &[false, false, false, true]);
        w.write_bit(false);
        w.write_choice_index(0, 2).expect("c1");
        w.write_choice_index(4, 16)
            .expect("requestLocationInformation");
        w.write_choice_index(0, 2).expect("criticalExtensions c1");
        w.write_choice_index(0, 4).expect("r9");
        w.write_sequence_preamble(Some(false), &[true, false, false, true, false]);
        let refused = w.into_bytes();
        assert_eq!(
            LppMessage::decode(&refused),
            Err(UperError::Unsupported(
                "LPP commonIEs for location information"
            ))
        );
    }

    // ---- capability transfer (§5.1.3) ----

    #[test]
    fn a_request_capabilities_is_answered_with_provide_capabilities() {
        let mut endpoint = LppEndpoint::new();
        let pdu = request(
            LppBody::RequestCapabilities {
                ecid_requested: true,
            },
            7,
        );

        let LppReaction::Reply(reply) = endpoint.handle_downlink(&pdu, &measurements(), &[]) else {
            panic!("a capability request must be answered");
        };
        let decoded = LppMessage::decode(&reply).expect("the reply decodes");
        assert_eq!(
            decoded.transaction_id.map(|t| t.transaction_number),
            Some(7),
            "the reply must echo the request's transaction number"
        );
        assert_eq!(
            decoded.transaction_id.map(|t| t.initiator),
            Some(Initiator::LocationServer),
            "the transaction stays the server's; the UE did not open it"
        );
        assert!(decoded.end_transaction, "the transaction is complete");
        match decoded.body {
            Some(LppBody::ProvideCapabilities { ecid_supported }) => {
                assert!(ecid_supported.rsrp, "RSRP is measured, so it is claimed");
                assert!(
                    !ecid_supported.rsrq,
                    "nothing in the UE populates CellMeasResult::rsrq"
                );
                assert!(
                    !ecid_supported.ue_rx_tx,
                    "UE Rx-Tx needs timing this simulator does not measure"
                );
            }
            other => panic!("expected ProvideCapabilities, got {other:?}"),
        }
        assert_eq!(endpoint.replies_sent(), 1);
    }

    #[test]
    fn the_advertised_capabilities_are_the_ones_the_ue_then_reports() {
        // The two halves of the transaction must not disagree: a claimed measurement
        // that comes back absent is worse than one never claimed.
        let mut endpoint = LppEndpoint::new();
        let LppReaction::Reply(capability_reply) = endpoint.handle_downlink(
            &request(
                LppBody::RequestCapabilities {
                    ecid_requested: true,
                },
                1,
            ),
            &measurements(),
            &[],
        ) else {
            panic!("a capability request is answered");
        };
        let claimed = match LppMessage::decode(&capability_reply).expect("decodes").body {
            Some(LppBody::ProvideCapabilities { ecid_supported }) => ecid_supported,
            other => panic!("expected ProvideCapabilities, got {other:?}"),
        };

        // Now ask for everything and see what actually arrives.
        let LppReaction::Reply(reply) = endpoint.handle_downlink(
            &request(asked_for(true, true, true), 2),
            &measurements(),
            &[],
        ) else {
            panic!("a location request is answered");
        };
        let element = match LppMessage::decode(&reply).expect("decodes").body {
            Some(LppBody::ProvideLocationInformation {
                measurements: r, ..
            }) => r.primary_cell.expect("primary"),
            other => panic!("expected ProvideLocationInformation, got {other:?}"),
        };
        assert_eq!(element.rsrp_result.is_some(), claimed.rsrp);
        assert_eq!(element.rsrq_result.is_some(), claimed.rsrq);
        assert_eq!(element.ue_rx_tx_time_diff.is_some(), claimed.ue_rx_tx);
    }

    // ---- location information transfer (§5.3.3) ----

    #[test]
    fn a_request_location_information_is_answered_with_a_measurement_report() {
        let mut endpoint = LppEndpoint::new();
        let pdu = request(asked_for(true, false, false), 3);

        let LppReaction::Reply(reply) = endpoint.handle_downlink(&pdu, &measurements(), &[]) else {
            panic!("a location request must be answered");
        };
        let decoded = LppMessage::decode(&reply).expect("the reply decodes");
        assert_eq!(
            decoded.transaction_id.map(|t| t.transaction_number),
            Some(3)
        );
        assert!(decoded.end_transaction);
        match decoded.body {
            Some(LppBody::ProvideLocationInformation {
                measurements: report,
                ..
            }) => {
                let element = report
                    .primary_cell
                    .clone()
                    .expect("the serving cell is the primary measurement");
                assert_eq!(element.phys_cell_id, 321);
                assert_eq!(element.arfcn, 1850);
                assert_eq!(element.system_frame_number, Some(42));
                assert_eq!(element.rsrp_result, Some(70));
                assert_eq!(
                    report.measured_results.len(),
                    1,
                    "MeasuredResultsList is SIZE(1..32), so it cannot be empty"
                );
                assert_eq!(
                    report.measured_results.first(),
                    report.primary_cell.as_ref(),
                    "the one list element is the serving cell, the same one"
                );
            }
            other => panic!("expected ProvideLocationInformation, got {other:?}"),
        }
    }

    #[test]
    fn only_the_requested_measurements_are_reported() {
        // A measurement the server did not ask for is not part of the answer, so a
        // request naming nothing gets a cell identity and no levels.
        let mut endpoint = LppEndpoint::new();
        for (rsrp_requested, expected) in [(true, Some(70)), (false, None)] {
            let pdu = request(asked_for(rsrp_requested, false, false), 1);
            let LppReaction::Reply(reply) = endpoint.handle_downlink(&pdu, &measurements(), &[])
            else {
                panic!("answered");
            };
            match LppMessage::decode(&reply).expect("decodes").body {
                Some(LppBody::ProvideLocationInformation {
                    measurements: report,
                    ..
                }) => {
                    let element = report.primary_cell.expect("primary");
                    assert_eq!(
                        element.rsrp_result, expected,
                        "rsrp asked: {rsrp_requested}"
                    );
                    // The cell identity is unconditional either way.
                    assert_eq!(element.phys_cell_id, 321);
                }
                other => panic!("expected ProvideLocationInformation, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_unmeasurable_quantity_is_absent_even_when_requested_and_available() {
        // The server asks for RSRQ and UE Rx-Tx; the fixture even HAS an RSRQ value.
        // The UE does not measure either, so both must be absent rather than passed
        // through as though they had been measured.
        let mut endpoint = LppEndpoint::new();
        let pdu = request(asked_for(false, true, true), 1);
        let LppReaction::Reply(reply) = endpoint.handle_downlink(&pdu, &measurements(), &[]) else {
            panic!("answered");
        };
        match LppMessage::decode(&reply).expect("decodes").body {
            Some(LppBody::ProvideLocationInformation {
                measurements: report,
                ..
            }) => {
                let element = report.primary_cell.expect("primary");
                assert_eq!(
                    element.rsrq_result, None,
                    "requested and available in the fixture, but not a measurement \
                     this UE makes"
                );
                assert_eq!(element.ue_rx_tx_time_diff, None);
                assert_eq!(element.rsrp_result, None, "not requested either");
                // The report is still well-formed: the cell identity is always there.
                assert_eq!(element.phys_cell_id, 321);
            }
            other => panic!("expected ProvideLocationInformation, got {other:?}"),
        }
    }

    #[test]
    fn a_measurement_the_ue_does_not_have_is_absent_rather_than_zero() {
        // Zero is a REAL RSRP report value (RSRP_00 is "below -140 dBm"), so
        // substituting it for "unknown" would report an implausibly weak signal as
        // a measurement.
        let mut endpoint = LppEndpoint::new();
        let unknown = ServingCellMeasurements {
            rsrp_result: None,
            ..measurements()
        };
        let pdu = request(asked_for(true, true, false), 1);
        let LppReaction::Reply(reply) = endpoint.handle_downlink(&pdu, &unknown, &[]) else {
            panic!("answered");
        };
        match LppMessage::decode(&reply).expect("decodes").body {
            Some(LppBody::ProvideLocationInformation {
                measurements: report,
                ..
            }) => {
                let element = report.primary_cell.expect("primary");
                assert_eq!(element.rsrp_result, None);
                assert_eq!(element.rsrq_result, None);
            }
            other => panic!("expected ProvideLocationInformation, got {other:?}"),
        }
    }

    #[test]
    fn a_pci_outside_the_asn1_range_is_reported_rather_than_encoded_wrongly() {
        // physCellId is INTEGER(0..503). A cell id above that cannot be encoded, and
        // truncating it would name a DIFFERENT cell in the report.
        let mut endpoint = LppEndpoint::new();
        let pdu = request(asked_for(true, false, false), 1);
        let out_of_range = ServingCellMeasurements {
            phys_cell_id: 504,
            ..measurements()
        };
        assert_eq!(
            endpoint.handle_downlink(&pdu, &out_of_range, &[]),
            LppReaction::Nothing
        );
        assert_eq!(endpoint.replies_sent(), 0);
        // And 503 itself is fine, so the bound is the constraint's and not an
        // off-by-one of mine.
        let at_bound = ServingCellMeasurements {
            phys_cell_id: 503,
            ..measurements()
        };
        assert!(matches!(
            endpoint.handle_downlink(&pdu, &at_bound, &[]),
            LppReaction::Reply(_)
        ));
    }

    // ---- what is refused ----

    #[test]
    fn a_malformed_pdu_produces_no_reply_rather_than_a_malformed_one() {
        let mut endpoint = LppEndpoint::new();
        for pdu in [vec![], vec![0xFF], vec![0xFF; 4], vec![0x00; 8]] {
            // Either it fails to decode, or it decodes as something with no answer.
            // Neither may panic, and neither may produce a reply that claims a fix.
            match endpoint.handle_downlink(&pdu, &measurements(), &[]) {
                LppReaction::Nothing => {}
                LppReaction::Reply(reply) => {
                    // If it did decode to a request, the reply must at least be
                    // well-formed -- a malformed reply is worse than silence.
                    LppMessage::decode(&reply).expect("any reply must be decodable");
                }
            }
        }
    }

    #[test]
    fn a_provide_message_from_the_network_is_not_answered() {
        let mut endpoint = LppEndpoint::new();
        let pdu = request(
            LppBody::ProvideCapabilities {
                ecid_supported: EcidMeasurementBits::default(),
            },
            9,
        );
        assert_eq!(
            endpoint.handle_downlink(&pdu, &measurements(), &[]),
            LppReaction::Nothing
        );
        assert_eq!(endpoint.replies_sent(), 0);
    }

    #[test]
    fn a_body_less_message_is_not_answered() {
        let mut endpoint = LppEndpoint::new();
        let pdu = LppMessage {
            transaction_id: None,
            end_transaction: false,
            body: None,
        }
        .encode()
        .expect("encodes");
        assert_eq!(
            endpoint.handle_downlink(&pdu, &measurements(), &[]),
            LppReaction::Nothing
        );
    }

    /// A request naming one root positioning method, by its index in
    /// `[commonIEs, a-gnss, otdoa, ecid, epdu]`, built by hand because this codec
    /// cannot produce one.
    fn request_naming_method(c1_index: usize, method_index: usize) -> Vec<u8> {
        let mut w = UperWriter::new();
        w.write_sequence_preamble(None, &[true, false, false, true]);
        // transactionID: extensible SEQUENCE preamble, initiator, number.
        w.write_sequence_preamble(Some(false), &[]);
        w.write_extensible_enumerated(0, 1).expect("root");
        w.write_constrained(4, 0, 255).expect("in range");
        w.write_bit(false); // endTransaction
        w.write_choice_index(0, 2).expect("c1");
        w.write_choice_index(c1_index, 16)
            .expect("a c1 alternative");
        w.write_choice_index(0, 2).expect("criticalExtensions c1");
        w.write_choice_index(0, 4).expect("r9");
        let mut presence = [false; 5];
        presence[method_index] = true;
        w.write_sequence_preamble(Some(false), &presence);
        w.into_bytes()
    }

    #[test]
    fn a_request_for_another_positioning_method_is_refused_not_answered_with_ecid() {
        // Answering a method the server did not ask for looks like a successful fix
        // and is not one.
        //
        // BOTH message types, because they fail differently and the revert harness
        // caught it: a location request naming another method already has no E-CID
        // body to read, so the absent-body guard refuses it even without the
        // method check -- while a CAPABILITY request legally has no E-CID body
        // (`ecid_requested: false`), so only the method check stands between an
        // A-GNSS capability enquiry and an E-CID answer to it.
        for c1_index in [
            0, // requestCapabilities
            4, // requestLocationInformation
        ] {
            for method_index in [1usize, 2, 4] {
                let pdu = request_naming_method(c1_index, method_index);
                let mut endpoint = LppEndpoint::new();
                assert_eq!(
                    endpoint.handle_downlink(&pdu, &measurements(), &[]),
                    LppReaction::Nothing,
                    "c1 alternative {c1_index} naming root member {method_index} must not \
                     be answered with an E-CID report"
                );
                assert_eq!(endpoint.replies_sent(), 0);
            }
        }
    }

    #[test]
    fn a_measured_results_element_carrying_a_cell_global_id_is_refused() {
        // This codec never SETS cellGlobalId, so no round trip reaches the field --
        // the revert harness proved that removing its guard changed nothing any test
        // could see. A peer that does set it must be refused rather than have every
        // later field read at the wrong offset, which would report a different cell
        // at a different level.
        let mut w = UperWriter::new();
        w.write_sequence_preamble(None, &[false, false, false, true]);
        w.write_bit(true); // endTransaction
        w.write_choice_index(0, 2).expect("c1");
        w.write_choice_index(5, 16)
            .expect("provideLocationInformation");
        w.write_choice_index(0, 2).expect("criticalExtensions c1");
        w.write_choice_index(0, 4).expect("r9");
        w.write_sequence_preamble(Some(false), &[false, false, false, true, false]);
        w.write_sequence_preamble(Some(false), &[true, false]); // sigMeas present
        w.write_sequence_preamble(Some(false), &[true]); // primaryCell present
                                                         // MeasuredResultsElement with cellGlobalId PRESENT.
        w.write_sequence_preamble(Some(false), &[true, false, false, false, false]);
        w.write_constrained(1, 0, 503).expect("pci");
        let pdu = w.into_bytes();

        assert_eq!(
            LppMessage::decode(&pdu),
            Err(UperError::Unsupported(
                "MeasuredResultsElement cellGlobalId"
            ))
        );
    }

    #[test]
    fn a_request_naming_no_method_at_all_is_refused() {
        // Nothing to answer: reporting E-CID anyway would answer a question the
        // server did not ask.
        let mut w = UperWriter::new();
        w.write_sequence_preamble(None, &[false, false, false, true]);
        w.write_bit(false);
        w.write_choice_index(0, 2).expect("c1");
        w.write_choice_index(4, 16)
            .expect("requestLocationInformation");
        w.write_choice_index(0, 2).expect("criticalExtensions c1");
        w.write_choice_index(0, 4).expect("r9");
        w.write_sequence_preamble(Some(false), &[false; 5]);
        let pdu = w.into_bytes();
        let mut endpoint = LppEndpoint::new();
        assert_eq!(
            endpoint.handle_downlink(&pdu, &measurements(), &[]),
            LppReaction::Nothing
        );
    }

    #[test]
    fn every_transaction_number_survives_the_echo() {
        // The transaction number is how the server matches a reply to its request,
        // so an off-by-one anywhere in the envelope would strand the transaction.
        let mut endpoint = LppEndpoint::new();
        for number in [0u8, 1, 127, 128, 254, 255] {
            for body in [
                LppBody::RequestCapabilities {
                    ecid_requested: true,
                },
                asked_for(true, false, false),
            ] {
                let pdu = request(body, number);
                let LppReaction::Reply(reply) =
                    endpoint.handle_downlink(&pdu, &measurements(), &[])
                else {
                    panic!("answered");
                };
                assert_eq!(
                    LppMessage::decode(&reply)
                        .expect("decodes")
                        .transaction_id
                        .map(|t| t.transaction_number),
                    Some(number)
                );
            }
        }
    }

    // ---- the NAS envelope (TS 24.501 §5.4.5.3) ----

    #[test]
    fn the_reply_is_a_ul_nas_transport_with_the_lpp_container_type() {
        // Announcing an LPP PDU as any other container type would have the AMF route
        // it to the wrong consumer, which is the bug this issue is about in reverse.
        let mut endpoint = LppEndpoint::new();
        let LppUplink::Send(pdu) =
            endpoint.handle_nas_container(LMF_REQUEST_LOCATION_INFORMATION, &measurements(), &[])
        else {
            panic!("the peer's request must produce an uplink");
        };

        // Plain 5GMM UL NAS TRANSPORT: EPD 0x7E, security header 0x00, message type
        // 0x67, then the payload container type.
        assert_eq!(&pdu[..3], &[0x7E, 0x00, 0x67]);
        assert_eq!(
            pdu[3],
            u8::from(PayloadContainerType::LppMessage),
            "the container type must be LPP (0b0011)"
        );

        // And the container really holds the LPP reply, decoded back out of the NAS
        // message rather than compared against what went in.
        let ul = UlNasTransport::decode(&mut &pdu[3..]).expect("the UL NAS TRANSPORT decodes");
        assert_eq!(ul.payload_container_type, PayloadContainerType::LppMessage);
        let inner = LppMessage::decode(&ul.payload_container).expect("the LPP PDU decodes");
        assert_eq!(inner.transaction_id.map(|t| t.transaction_number), Some(0));
        assert!(inner.end_transaction);
        assert!(matches!(
            inner.body,
            Some(LppBody::ProvideLocationInformation { .. })
        ));
    }

    #[test]
    fn nothing_to_answer_produces_no_uplink_at_all() {
        // Not an empty UL NAS TRANSPORT: an LPP container with no LPP message in it
        // would be a protocol error at the LMF.
        let mut endpoint = LppEndpoint::new();
        assert_eq!(
            endpoint.handle_nas_container(&[0xFF; 3], &measurements(), &[]),
            LppUplink::Nothing
        );
    }

    #[test]
    fn a_request_with_no_transaction_id_gets_a_reply_with_none_either() {
        // Inventing one would tie the reply to a transaction the server never opened.
        let mut endpoint = LppEndpoint::new();
        let pdu = LppMessage {
            transaction_id: None,
            end_transaction: false,
            body: Some(asked_for(true, false, false)),
        }
        .encode()
        .expect("encodes");
        let LppReaction::Reply(reply) = endpoint.handle_downlink(&pdu, &measurements(), &[]) else {
            panic!("answered");
        };
        assert_eq!(
            LppMessage::decode(&reply).expect("decodes").transaction_id,
            None
        );
    }

    // ---- the sidelink ranging report (issue #137) ----

    /// One measured range, in the units and bounds the IE carries.
    fn ranging_result() -> SidelinkRangingResult {
        SidelinkRangingResult {
            peer_layer2_id: 0x00A5_A5A5,
            // 50.00 m: the same geometry #136's SL-PRS stimulus test uses.
            range_cm: 5_000,
            accuracy_cm: 1,
            method: SidelinkRangingMethod::CarrierPhase,
            measurement_count: 3,
        }
    }

    /// The report's open-type payload, derived bit by bit rather than captured from
    /// this encoder -- a self round trip cannot catch a layout both halves get wrong,
    /// and nextgcore's LMF has to read these bytes:
    ///
    /// ```text
    ///   SidelinkRangingReport SEQUENCE, extensible, no additions:   0
    ///   results count 1, SIZE(1..32) -> 5 bits of (1-1):            0 0000
    ///   SidelinkRangingResult SEQUENCE, extensible, no additions:   0
    ///   peerLayer2Id 0xA5A5A5, INTEGER(0..16777215) -> 24 bits:     1010 0101 1010 0101 1010 0101
    ///   rangeCm 5000, INTEGER(0..1000000) -> 20 bits:               0000 0001 0011 1000 1000
    ///   accuracyCm 1, INTEGER(0..65535) -> 16 bits:                 0000 0000 0000 0001
    ///   method: extensible ENUMERATED, root value:                  0
    ///          index 1 (carrierPhase), root max 1 -> 1 bit:         1
    ///   measurementCount 3, INTEGER(0..65535) -> 16 bits:           0000 0000 0000 0011
    ///                                                              = 85 bits, padded to 11 octets
    /// ```
    const GOLDEN_SIDELINK_RANGING_REPORT: &[u8] = &[
        0x01, 0x4B, 0x4B, 0x4A, 0x02, 0x71, 0x00, 0x00, 0x28, 0x00, 0x18,
    ];

    #[test]
    fn the_sidelink_ranging_report_matches_its_hand_derived_bytes() {
        let report = SidelinkRangingReport {
            results: vec![ranging_result()],
        };
        assert_eq!(
            report.encode().expect("encodes"),
            GOLDEN_SIDELINK_RANGING_REPORT
        );
        assert_eq!(
            SidelinkRangingReport::decode(GOLDEN_SIDELINK_RANGING_REPORT).expect("decodes"),
            report,
            "and the hand-derived bytes decode back to the same report"
        );
    }

    /// A report with sidelink results survives the whole message: it rides an
    /// extension addition of the `-r9` IEs, and comes back out of a decode.
    #[test]
    fn a_provide_location_information_carries_the_ranging_report_through_a_round_trip() {
        let mut endpoint = LppEndpoint::new();
        let results = [ranging_result()];
        let LppReaction::Reply(reply) =
            endpoint.handle_downlink(LMF_REQUEST_LOCATION_INFORMATION, &measurements(), &results)
        else {
            panic!("the peer's request must be answered");
        };

        match LppMessage::decode(&reply).expect("decodes").body {
            Some(LppBody::ProvideLocationInformation { sidelink, .. }) => {
                assert_eq!(
                    sidelink,
                    Some(SidelinkRangingReport {
                        results: vec![ranging_result()],
                    }),
                    "the ranging report must survive the extension addition"
                );
            }
            other => panic!("expected ProvideLocationInformation, got {other:?}"),
        }
    }

    /// With no ranging results the encoding is what it was before sidelink existed:
    /// the extension bit stays CLEAR, so nextgcore's LMF -- which skips additions it
    /// does not know -- sees an unchanged message rather than one it has to tolerate.
    ///
    /// The 40 pre-existing tests in this module are the other half of this guard:
    /// they assert the exact reply bytes and still pass.
    #[test]
    fn a_reply_with_no_ranging_results_sets_no_extension_bit() {
        let mut endpoint = LppEndpoint::new();
        let LppReaction::Reply(without) =
            endpoint.handle_downlink(LMF_REQUEST_LOCATION_INFORMATION, &measurements(), &[])
        else {
            panic!("answered");
        };
        let LppReaction::Reply(with) = endpoint.handle_downlink(
            LMF_REQUEST_LOCATION_INFORMATION,
            &measurements(),
            &[ranging_result()],
        ) else {
            panic!("answered");
        };

        assert!(
            with.len() > without.len(),
            "the addition must add octets; without={without:02X?} with={with:02X?}"
        );
        // The two differ from the -r9 preamble onward, which is where the extension
        // bit lives -- so the difference is not merely appended bytes.
        assert_ne!(
            with[..without.len()],
            without[..],
            "the extension bit sits INSIDE the message, so the shorter encoding \
             cannot be a prefix of the longer one"
        );
        assert!(
            matches!(
                LppMessage::decode(&without).expect("decodes").body,
                Some(LppBody::ProvideLocationInformation { sidelink: None, .. })
            ),
            "and it decodes back as carrying no report"
        );
    }

    /// An EMPTY result set must encode as an ABSENT report, not an empty list:
    /// `SEQUENCE (SIZE(1..32))` cannot be empty, and this codec's encoder would
    /// refuse it -- but a caller that built one anyway would put bytes on the wire
    /// that fail to decode at the LMF.
    #[test]
    fn an_empty_ranging_result_set_is_reported_as_absent_rather_than_as_an_empty_list() {
        assert_eq!(
            SidelinkRangingReport { results: vec![] }.encode(),
            Err(UperError::InvalidLength(0)),
            "the encoder must refuse an empty list"
        );

        let mut endpoint = LppEndpoint::new();
        let LppReaction::Reply(reply) =
            endpoint.handle_downlink(LMF_REQUEST_LOCATION_INFORMATION, &measurements(), &[])
        else {
            panic!("answered");
        };
        assert!(
            matches!(
                LppMessage::decode(&reply).expect("decodes").body,
                Some(LppBody::ProvideLocationInformation { sidelink: None, .. })
            ),
            "and the endpoint must leave the report absent rather than build one"
        );
    }

    /// More results than the IE can carry are TRUNCATED with a warning, not encoded:
    /// 33 entries encode to bytes that then fail to decode, and the sender would see
    /// no error at all.
    #[test]
    fn more_results_than_the_ie_can_carry_are_truncated_to_its_bound() {
        let mut endpoint = LppEndpoint::new();
        let many: Vec<SidelinkRangingResult> = (0..40)
            .map(|index| SidelinkRangingResult {
                peer_layer2_id: index,
                ..ranging_result()
            })
            .collect();
        let LppReaction::Reply(reply) =
            endpoint.handle_downlink(LMF_REQUEST_LOCATION_INFORMATION, &measurements(), &many)
        else {
            panic!("answered");
        };
        match LppMessage::decode(&reply).expect("decodes").body {
            Some(LppBody::ProvideLocationInformation {
                sidelink: Some(report),
                ..
            }) => {
                let encoded = report.encode().expect("the kept report encodes");
                assert!(
                    encoded.len() <= SidelinkRangingReport::MAX_ADDITION_OCTETS,
                    "the kept report must fit an addition: {} octets",
                    encoded.len()
                );
                assert!(
                    report.results.len() < SidelinkRangingReport::MAX_RESULTS,
                    "the binding bound is the 127-octet open type, NOT SIZE(1..32): \
                     {} results kept",
                    report.results.len()
                );
                assert_eq!(
                    report.results[0].peer_layer2_id, 0,
                    "and it is the FIRST results that are kept"
                );
            }
            other => panic!("expected a report, got {other:?}"),
        }
    }
}
