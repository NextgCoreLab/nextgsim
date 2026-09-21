//! PC5-S signalling: the wire messages of PC5 unicast link establishment,
//! direct discovery and UE-to-UE relay (TS 24.554, TS 23.304).
//!
//! # Why this module exists (issue #141)
//!
//! Before this, `SidelinkMessage::EstablishPc5Link` set `Pc5LinkState::Establishing`
//! and then `Pc5LinkState::Active` on the next line, with only a `debug!` between
//! them. There was no wire format at all — no `Direct Communication Request`, no
//! `Direct Communication Accept`, and nothing for a peer UE to answer. Discovery was
//! a local `discovery_active: bool`. So the PC5 surface modelled the *shapes* of
//! TS 23.304 without ever putting a byte between two UEs.
//!
//! This module is the missing wire format. It is deliberately **not** ASN.1: PC5-S is
//! a NAS-layer protocol specified in TS 24.554 as octet-aligned type/length/value
//! messages with a one-octet message type, exactly like the 5GMM/5GSM messages
//! `nextgsim-nas` already encodes by hand — not like the RRC IEs of TS 38.331, which
//! the generated codec owns. Encoding it through the RRC codec would be a category
//! error; `SidelinkUEInformation` and `sl-Config` are the RRC half and live in
//! `nextgsim-rrc::procedures::sidelink_ue_information`.
//!
//! # The message set, and why exactly these
//!
//! TS 24.554 §6.1.2 defines a dozen PC5 signalling messages. Implemented here are the
//! ones the issue's acceptance criteria name, and no more:
//!
//! | Message | Clause | Why |
//! |---|---|---|
//! | `DIRECT COMMUNICATION REQUEST` | §6.1.2.1 | Step 3 of TS 23.304 §6.4.3.1 |
//! | `DIRECT COMMUNICATION ACCEPT` | §6.1.2.2 | Step 5 of TS 23.304 §6.4.3.1 |
//! | `DIRECT COMMUNICATION REJECT` | §6.1.2.3 | The negative answer to step 5 — without it a refusal is indistinguishable from a lost message |
//! | `DIRECT COMMUNICATION RELEASE` | §6.1.2.4 | The link has to be releasable over the air too, or `ReleasePc5Link` stays local |
//! | `DIRECT DISCOVERY ANNOUNCEMENT` | §7 / TS 23.304 §6.3.1.2 | Model A announce |
//! | `DIRECT DISCOVERY SOLICITATION` | §7 / TS 23.304 §6.3.1.3 | Model B solicit |
//! | `DIRECT DISCOVERY RESPONSE` | §7 / TS 23.304 §6.3.1.3 | Model B response |
//!
//! Not implemented, and not stubbed: link modification (§6.1.2.5), link identifier
//! update (§6.1.2.7), and keep-alive (§6.1.2.9). Nothing in this tree drives them, and
//! a message type with no sender is the defect this issue exists to remove.
//!
//! # Security: what is carried and what is not
//!
//! TS 23.304 §6.4.3.1 step 4 interposes a security establishment procedure between the
//! request and the accept, defined in TS 33.503. That procedure is **not** implemented,
//! and the `Security Information` IE is carried as an opaque octet string rather than
//! being given invented semantics. What *is* enforced is the observable consequence the
//! spec pins in step 4: "Upon receiving the security establishment procedure messages,
//! UE-1 obtains the peer UE's Layer-2 ID for future communication". So the responder's
//! Layer-2 ID is carried in the accept and the initiator adopts it — see
//! [`Pc5LinkContext::peer_l2_id`]. A real TS 33.503 exchange would change which bytes
//! prove that, not whether it must happen.

use std::fmt;

/// The PC5 signalling message type, octet 1 of every PC5-S message
/// (TS 24.554 §6.1.2, §8.2 "Message type").
///
/// The discriminant values are this simulator's own assignment within the
/// spare/reserved range rather than TS 24.554's own table: the published table is not
/// in the vendored spec text (`../../6g_docs/specs/` carries TS 23.304 but not
/// TS 24.554), and inventing a value that *collided* with a real one would be worse
/// than picking an unambiguously local one. Both ends of every exchange in this tree
/// read this same enum, so the values are self-consistent; a future agent with the
/// real table needs to change only this `match` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Pc5SMessageType {
    /// `DIRECT COMMUNICATION REQUEST` (TS 24.554 §6.1.2.1).
    DirectCommunicationRequest = 0x01,
    /// `DIRECT COMMUNICATION ACCEPT` (TS 24.554 §6.1.2.2).
    DirectCommunicationAccept = 0x02,
    /// `DIRECT COMMUNICATION REJECT` (TS 24.554 §6.1.2.3).
    DirectCommunicationReject = 0x03,
    /// `DIRECT COMMUNICATION RELEASE` (TS 24.554 §6.1.2.4).
    DirectCommunicationRelease = 0x04,
    /// Model A announcement (TS 23.304 §6.3.1.2 step 3a).
    DirectDiscoveryAnnouncement = 0x11,
    /// Model B solicitation (TS 23.304 §6.3.1.3).
    DirectDiscoverySolicitation = 0x12,
    /// Model B response (TS 23.304 §6.3.1.3).
    DirectDiscoveryResponse = 0x13,
}

impl Pc5SMessageType {
    /// The octet this type encodes to.
    pub fn to_octet(self) -> u8 {
        self as u8
    }

    /// The type an octet decodes to, or `None` for a type this UE does not implement.
    ///
    /// `None` rather than an error variant per unknown octet: TS 24.554 §6.1.2 has
    /// message types this module deliberately does not carry (link modification,
    /// keep-alive), and a peer that sends one is not malformed — it is using a
    /// procedure this UE has not implemented. The caller logs and ignores it.
    pub fn from_octet(octet: u8) -> Option<Self> {
        match octet {
            0x01 => Some(Self::DirectCommunicationRequest),
            0x02 => Some(Self::DirectCommunicationAccept),
            0x03 => Some(Self::DirectCommunicationReject),
            0x04 => Some(Self::DirectCommunicationRelease),
            0x11 => Some(Self::DirectDiscoveryAnnouncement),
            0x12 => Some(Self::DirectDiscoverySolicitation),
            0x13 => Some(Self::DirectDiscoveryResponse),
            _ => None,
        }
    }
}

/// Why a PC5-S message could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pc5SError {
    /// The buffer ended before a field did.
    Truncated {
        /// The field being read when the bytes ran out.
        field: &'static str,
        /// Octets needed.
        needed: usize,
        /// Octets available.
        available: usize,
    },
    /// Octet 1 is not a message type this UE implements.
    UnknownMessageType(u8),
    /// The message type octet is a PC5-S type, but not the one the caller asked to
    /// decode.
    UnexpectedMessageType {
        /// What the caller expected.
        expected: Pc5SMessageType,
        /// What octet 1 actually held.
        actual: Pc5SMessageType,
    },
    /// A variable-length IE declared a length the message cannot hold.
    BadLength {
        /// The IE.
        field: &'static str,
        /// The declared length.
        declared: usize,
        /// The octets actually remaining.
        remaining: usize,
    },
}

impl fmt::Display for Pc5SError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated {
                field,
                needed,
                available,
            } => write!(
                f,
                "PC5-S message truncated reading {field}: need {needed} octet(s), have {available}"
            ),
            Self::UnknownMessageType(octet) => {
                write!(f, "unimplemented PC5-S message type 0x{octet:02X}")
            }
            Self::UnexpectedMessageType { expected, actual } => {
                write!(f, "expected PC5-S {expected:?}, got {actual:?}")
            }
            Self::BadLength {
                field,
                declared,
                remaining,
            } => write!(
                f,
                "PC5-S IE {field} declares {declared} octet(s) but only {remaining} remain"
            ),
        }
    }
}

impl std::error::Error for Pc5SError {}

/// A ProSe Layer-2 ID: the 24-bit PC5 source/destination address
/// (TS 23.304 §5.8.2.1, "the Layer-2 ID is 24 bits").
///
/// A distinct type rather than a bare `u32` because 24 bits in a 32-bit carrier is
/// exactly the shape that silently truncates: [`ProseL2Id::new`] masks to 24 bits at
/// construction, so a value that round-trips through the wire cannot differ from the
/// value a caller thought it set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProseL2Id(u32);

impl ProseL2Id {
    /// The 24-bit mask a Layer-2 ID is held to.
    const MASK: u32 = 0x00FF_FFFF;

    /// A Layer-2 ID from its low 24 bits. Bits above 24 are discarded.
    pub fn new(value: u32) -> Self {
        Self(value & Self::MASK)
    }

    /// The ID as a `u32`, always `<= 0xFFFFFF`.
    pub fn value(self) -> u32 {
        self.0
    }

    /// The three octets this ID encodes to, most significant first.
    fn to_octets(self) -> [u8; 3] {
        [
            ((self.0 >> 16) & 0xFF) as u8,
            ((self.0 >> 8) & 0xFF) as u8,
            (self.0 & 0xFF) as u8,
        ]
    }

    /// Reads three octets, most significant first.
    fn from_octets(octets: [u8; 3]) -> Self {
        Self((u32::from(octets[0]) << 16) | (u32::from(octets[1]) << 8) | u32::from(octets[2]))
    }
}

impl fmt::Display for ProseL2Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:06X}", self.0)
    }
}

/// A Relay Service Code: the 24-bit identifier of a 5G ProSe relay service
/// (TS 23.304 §5.4.2, "the RSC ... identifies a connectivity service the 5G ProSe
/// UE-to-Network Relay provides").
///
/// 24 bits for the same reason as [`ProseL2Id`], and masked at construction for the
/// same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelayServiceCode(u32);

impl RelayServiceCode {
    /// A relay service code from its low 24 bits.
    pub fn new(value: u32) -> Self {
        Self(value & ProseL2Id::MASK)
    }

    /// The code as a `u32`, always `<= 0xFFFFFF`.
    pub fn value(self) -> u32 {
        self.0
    }
}

impl fmt::Display for RelayServiceCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:06X}", self.0)
    }
}

/// The reason a `DIRECT COMMUNICATION REJECT` refuses a link
/// (TS 24.554 §6.1.2.3, `PC5 signalling protocol cause`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Pc5RejectCause {
    /// The target user info in the request is not this UE (TS 23.304 §6.4.3.1 step
    /// 5a: the accept is sent "if the Application Layer ID for UE-2 matches").
    TargetUserMismatch = 0x01,
    /// This UE is not interested in the announced ProSe service (step 5b).
    ProseServiceNotInterested = 0x02,
    /// This UE cannot serve the requested relay service code.
    RelayServiceNotSupported = 0x03,
    /// This UE has no resources for another unicast link.
    InsufficientResources = 0x04,
}

impl Pc5RejectCause {
    /// The octet this cause encodes to.
    pub fn to_octet(self) -> u8 {
        self as u8
    }

    /// The cause an octet decodes to.
    ///
    /// An unrecognised cause maps to [`Pc5RejectCause::InsufficientResources`] rather
    /// than failing the decode: the message is still a valid rejection, and losing the
    /// rejection because its reason is unfamiliar would leave the initiator waiting on
    /// a link the peer has already refused.
    pub fn from_octet(octet: u8) -> Self {
        match octet {
            0x01 => Self::TargetUserMismatch,
            0x02 => Self::ProseServiceNotInterested,
            0x03 => Self::RelayServiceNotSupported,
            _ => Self::InsufficientResources,
        }
    }
}

/// Which PC5 cast type a message is sent with (TS 23.304 §5.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pc5CastType {
    /// One named peer. The destination Layer-2 ID is that peer's.
    Unicast,
    /// Every UE monitoring the discovery destination Layer-2 ID
    /// (TS 23.304 §5.8.2.4).
    Broadcast,
}

/// `DIRECT COMMUNICATION REQUEST` (TS 24.554 §6.1.2.1), step 3 of
/// TS 23.304 §6.4.3.1.
///
/// # Layout
///
/// ```text
/// octet 1     message type (0x01)
/// octets 2-4  Source User Info: the initiating UE's Layer-2 ID
/// octet 5     flags: bit 0 = Target User Info present, bit 1 = RSC present
/// octets ...  Target User Info (3 octets, if bit 0)
/// octets ...  Relay Service Code (3 octets, if bit 1)
/// octets ...  ProSe Service Info: 1 length octet then that many octets
/// octets ...  Security Information: 1 length octet then that many octets
/// ```
///
/// The two optional IEs are flagged rather than tagged because the presence of Target
/// User Info is *semantic* here, not merely an optimisation: TS 23.304 §6.4.3.1 step
/// 5a/5b make the whole responder behaviour turn on it — with it, only the named UE
/// answers; without it, every UE interested in the announced service answers. A
/// receiver therefore has to be able to tell "absent" from "present and zero", which
/// is why `Option<ProseL2Id>` and not a sentinel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectCommunicationRequest {
    /// `Source User Info`: the initiating UE's Layer-2 ID (TS 23.304 §6.4.3.1 step 3).
    pub source_l2_id: ProseL2Id,
    /// `Target User Info`, when the application layer named a target UE.
    ///
    /// `Some` selects UE-oriented establishment (step 5a): only that UE accepts.
    /// `None` selects ProSe-service-oriented establishment (step 5b): every UE
    /// interested in `prose_service_info` accepts.
    pub target_l2_id: Option<ProseL2Id>,
    /// The Relay Service Code, when this request is for relay connectivity
    /// (TS 23.304 §6.4.3.1 note, §5.4.2).
    pub relay_service_code: Option<RelayServiceCode>,
    /// `ProSe Service Info`: "the information about the ProSe identifier(s)
    /// requesting Layer-2 link establishment" (step 3).
    pub prose_service_info: Vec<u8>,
    /// `Security Information`: "the information for the establishment of security"
    /// (step 3).
    ///
    /// Opaque here. TS 23.304's NOTE 1 defers its content to TS 33.503, which this
    /// module does not implement, so giving these octets invented structure would
    /// claim a security procedure that does not run. Carried so that the IE exists on
    /// the wire and a TS 33.503 implementation has somewhere to put its payload.
    pub security_info: Vec<u8>,
}

/// Bit 0 of the request flags octet: `Target User Info` follows.
const FLAG_TARGET_USER_INFO: u8 = 0b0000_0001;
/// Bit 1 of the request flags octet: `Relay Service Code` follows.
const FLAG_RELAY_SERVICE_CODE: u8 = 0b0000_0010;

impl DirectCommunicationRequest {
    /// Encodes the message, message-type octet included.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.prose_service_info.len());
        out.push(Pc5SMessageType::DirectCommunicationRequest.to_octet());
        out.extend_from_slice(&self.source_l2_id.to_octets());

        let mut flags = 0u8;
        if self.target_l2_id.is_some() {
            flags |= FLAG_TARGET_USER_INFO;
        }
        if self.relay_service_code.is_some() {
            flags |= FLAG_RELAY_SERVICE_CODE;
        }
        out.push(flags);

        if let Some(target) = self.target_l2_id {
            out.extend_from_slice(&target.to_octets());
        }
        if let Some(rsc) = self.relay_service_code {
            out.extend_from_slice(&ProseL2Id(rsc.value()).to_octets());
        }
        push_lv(&mut out, &self.prose_service_info);
        push_lv(&mut out, &self.security_info);
        out
    }

    /// Decodes the message, message-type octet included.
    pub fn decode(bytes: &[u8]) -> Result<Self, Pc5SError> {
        let mut r = Reader::new(bytes);
        r.expect_type(Pc5SMessageType::DirectCommunicationRequest)?;
        let source_l2_id = r.l2_id("sourceUserInfo")?;
        let flags = r.octet("flags")?;
        let target_l2_id = if flags & FLAG_TARGET_USER_INFO != 0 {
            Some(r.l2_id("targetUserInfo")?)
        } else {
            None
        };
        let relay_service_code = if flags & FLAG_RELAY_SERVICE_CODE != 0 {
            Some(RelayServiceCode::new(r.l2_id("relayServiceCode")?.value()))
        } else {
            None
        };
        let prose_service_info = r.lv("proseServiceInfo")?;
        let security_info = r.lv("securityInformation")?;
        Ok(Self {
            source_l2_id,
            target_l2_id,
            relay_service_code,
            prose_service_info,
            security_info,
        })
    }
}

/// `DIRECT COMMUNICATION ACCEPT` (TS 24.554 §6.1.2.2), step 5 of
/// TS 23.304 §6.4.3.1.
///
/// # Layout
///
/// ```text
/// octet 1     message type (0x02)
/// octets 2-4  Source User Info: the accepting UE's Layer-2 ID
/// octet 5     PFI, the PC5 QoS Flow Identifier
/// octet 6     flags: bit 0 = this UE is acting as a relay for the requested RSC
/// ```
///
/// `Source User Info` is what makes the accept load-bearing rather than a bare ack:
/// TS 23.304 §6.4.3.1 step 4 says the initiator "obtains the peer UE's Layer-2 ID for
/// future communication" from the responder's messages, and step 5 lists `Source User
/// Info` as a member of the accept. So an initiator that has processed an accept knows
/// which Layer-2 ID to address — and that is the value
/// [`Pc5LinkContext::peer_l2_id`] holds and the round-trip tests assert, because it is
/// reachable *only* by having decoded a real accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectCommunicationAccept {
    /// `Source User Info`: "Application Layer ID of the UE sending the Direct
    /// Communication Accept message" (TS 23.304 §6.4.3.1 step 5), as a Layer-2 ID.
    pub source_l2_id: ProseL2Id,
    /// `QoS Info`: the PC5 QoS Flow Identifier for the flow this link carries
    /// (TS 23.304 §5.6.1, step 5's "the PFI and the corresponding PC5 QoS
    /// parameters").
    ///
    /// The PFI alone, not the whole PC5 QoS parameter set: the PQI/MFBR/GFBR
    /// parameters have no consumer in this tree — no PC5 scheduler reads them — and an
    /// IE nothing reads is the defect this issue removes. The PFI *is* read: it keys
    /// the accepted link's bearer.
    pub pfi: u8,
    /// Whether the accepting UE is acting as a relay for the requested Relay Service
    /// Code (TS 23.304 §6.4.3.1, the relay case of step 5).
    pub acting_as_relay: bool,
}

/// Bit 0 of the accept flags octet: the responder accepted as a relay.
const FLAG_ACTING_AS_RELAY: u8 = 0b0000_0001;

impl DirectCommunicationAccept {
    /// Encodes the message, message-type octet included.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(6);
        out.push(Pc5SMessageType::DirectCommunicationAccept.to_octet());
        out.extend_from_slice(&self.source_l2_id.to_octets());
        out.push(self.pfi);
        out.push(if self.acting_as_relay {
            FLAG_ACTING_AS_RELAY
        } else {
            0
        });
        out
    }

    /// Decodes the message, message-type octet included.
    pub fn decode(bytes: &[u8]) -> Result<Self, Pc5SError> {
        let mut r = Reader::new(bytes);
        r.expect_type(Pc5SMessageType::DirectCommunicationAccept)?;
        let source_l2_id = r.l2_id("sourceUserInfo")?;
        let pfi = r.octet("pfi")?;
        let flags = r.octet("flags")?;
        Ok(Self {
            source_l2_id,
            pfi,
            acting_as_relay: flags & FLAG_ACTING_AS_RELAY != 0,
        })
    }
}

/// `DIRECT COMMUNICATION REJECT` (TS 24.554 §6.1.2.3): the negative answer to a
/// request.
///
/// ```text
/// octet 1     message type (0x03)
/// octets 2-4  Source User Info: the rejecting UE's Layer-2 ID
/// octet 5     PC5 signalling protocol cause
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectCommunicationReject {
    /// The rejecting UE's Layer-2 ID.
    pub source_l2_id: ProseL2Id,
    /// Why the link was refused.
    pub cause: Pc5RejectCause,
}

impl DirectCommunicationReject {
    /// Encodes the message, message-type octet included.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(5);
        out.push(Pc5SMessageType::DirectCommunicationReject.to_octet());
        out.extend_from_slice(&self.source_l2_id.to_octets());
        out.push(self.cause.to_octet());
        out
    }

    /// Decodes the message, message-type octet included.
    pub fn decode(bytes: &[u8]) -> Result<Self, Pc5SError> {
        let mut r = Reader::new(bytes);
        r.expect_type(Pc5SMessageType::DirectCommunicationReject)?;
        let source_l2_id = r.l2_id("sourceUserInfo")?;
        let cause = Pc5RejectCause::from_octet(r.octet("cause")?);
        Ok(Self {
            source_l2_id,
            cause,
        })
    }
}

/// `DIRECT COMMUNICATION RELEASE` (TS 24.554 §6.1.2.4): tear the unicast link down.
///
/// ```text
/// octet 1     message type (0x04)
/// octets 2-4  Source User Info: the releasing UE's Layer-2 ID
/// ```
///
/// No cause octet. TS 24.554 defines one, but nothing in this tree distinguishes
/// release reasons — the receiver's action is the same for all of them — and an IE
/// whose every value produces identical behaviour is exactly the kind of decorative
/// field this issue removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectCommunicationRelease {
    /// The releasing UE's Layer-2 ID.
    pub source_l2_id: ProseL2Id,
}

impl DirectCommunicationRelease {
    /// Encodes the message, message-type octet included.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4);
        out.push(Pc5SMessageType::DirectCommunicationRelease.to_octet());
        out.extend_from_slice(&self.source_l2_id.to_octets());
        out
    }

    /// Decodes the message, message-type octet included.
    pub fn decode(bytes: &[u8]) -> Result<Self, Pc5SError> {
        let mut r = Reader::new(bytes);
        r.expect_type(Pc5SMessageType::DirectCommunicationRelease)?;
        Ok(Self {
            source_l2_id: r.l2_id("sourceUserInfo")?,
        })
    }
}

/// A direct discovery message: Model A announcement, or Model B solicitation or
/// response (TS 23.304 §6.3.1.2, §6.3.1.3).
///
/// One type for all three because the three carry the same IEs and differ only in
/// which message type octet they bear and who is expected to answer:
///
/// * **Model A announcement** — broadcast, unsolicited. TS 23.304 §6.3.1.2 step 3a:
///   the authorised announcer "starts announcing on PC5 interface". A monitor that
///   matches its filter learns the announcer and answers nothing.
/// * **Model B solicitation** — broadcast. §6.3.1.3: the discoverer asks who is out
///   there matching a ProSe application code.
/// * **Model B response** — unicast back to the solicitor. §6.3.1.3: the discoveree
///   answers with its own code.
///
/// ```text
/// octet 1     message type (0x11 announce / 0x12 solicit / 0x13 response)
/// octets 2-4  Source User Info: the sending UE's Layer-2 ID
/// octets 5-8  ProSe Application Code (big-endian u32)
/// octet 9     flags: bit 0 = Relay Service Code present
/// octets ...  Relay Service Code (3 octets, if bit 0)
/// ```
///
/// The Relay Service Code rides the discovery message because that is how a remote UE
/// finds a relay at all: TS 23.304 §6.3.2 relay discovery has the relay announce the
/// RSC it serves, and without it `SetRelayMode` could only ever be set by local
/// configuration — which is precisely the unreachable-handler defect this issue
/// names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectDiscoveryMessage {
    /// Which of the three discovery messages this is.
    pub message_type: Pc5SMessageType,
    /// The sending UE's Layer-2 ID.
    pub source_l2_id: ProseL2Id,
    /// `ProSe Application Code` (TS 23.304 §5.8.1): what a monitor's discovery filter
    /// is matched against.
    pub prose_app_code: u32,
    /// The Relay Service Code this UE serves, when it is announcing as a relay
    /// (TS 23.304 §6.3.2).
    pub relay_service_code: Option<RelayServiceCode>,
}

/// Bit 0 of the discovery flags octet: `Relay Service Code` follows.
const FLAG_DISCOVERY_RSC: u8 = 0b0000_0001;

impl DirectDiscoveryMessage {
    /// A Model A announcement (TS 23.304 §6.3.1.2 step 3a).
    pub fn announcement(source_l2_id: ProseL2Id, prose_app_code: u32) -> Self {
        Self {
            message_type: Pc5SMessageType::DirectDiscoveryAnnouncement,
            source_l2_id,
            prose_app_code,
            relay_service_code: None,
        }
    }

    /// A Model B solicitation (TS 23.304 §6.3.1.3).
    pub fn solicitation(source_l2_id: ProseL2Id, prose_app_code: u32) -> Self {
        Self {
            message_type: Pc5SMessageType::DirectDiscoverySolicitation,
            source_l2_id,
            prose_app_code,
            relay_service_code: None,
        }
    }

    /// A Model B response (TS 23.304 §6.3.1.3).
    pub fn response(source_l2_id: ProseL2Id, prose_app_code: u32) -> Self {
        Self {
            message_type: Pc5SMessageType::DirectDiscoveryResponse,
            source_l2_id,
            prose_app_code,
            relay_service_code: None,
        }
    }

    /// Declares the Relay Service Code this UE serves (TS 23.304 §6.3.2).
    pub fn with_relay_service_code(mut self, rsc: RelayServiceCode) -> Self {
        self.relay_service_code = Some(rsc);
        self
    }

    /// Whether this message expects an answer.
    ///
    /// Only a Model B solicitation does. A Model A announcement is unsolicited and a
    /// Model B response closes the exchange — which is the entire behavioural
    /// difference between the two models, so it is a named method rather than a
    /// comparison spelled out at each call site.
    pub fn expects_response(&self) -> bool {
        self.message_type == Pc5SMessageType::DirectDiscoverySolicitation
    }

    /// Encodes the message, message-type octet included.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12);
        out.push(self.message_type.to_octet());
        out.extend_from_slice(&self.source_l2_id.to_octets());
        out.extend_from_slice(&self.prose_app_code.to_be_bytes());
        out.push(if self.relay_service_code.is_some() {
            FLAG_DISCOVERY_RSC
        } else {
            0
        });
        if let Some(rsc) = self.relay_service_code {
            out.extend_from_slice(&ProseL2Id(rsc.value()).to_octets());
        }
        out
    }

    /// Decodes any of the three discovery messages, message-type octet included.
    pub fn decode(bytes: &[u8]) -> Result<Self, Pc5SError> {
        let mut r = Reader::new(bytes);
        let message_type = r.message_type()?;
        if !matches!(
            message_type,
            Pc5SMessageType::DirectDiscoveryAnnouncement
                | Pc5SMessageType::DirectDiscoverySolicitation
                | Pc5SMessageType::DirectDiscoveryResponse
        ) {
            return Err(Pc5SError::UnexpectedMessageType {
                expected: Pc5SMessageType::DirectDiscoveryAnnouncement,
                actual: message_type,
            });
        }
        let source_l2_id = r.l2_id("sourceUserInfo")?;
        let prose_app_code = r.u32_be("proseApplicationCode")?;
        let flags = r.octet("flags")?;
        let relay_service_code = if flags & FLAG_DISCOVERY_RSC != 0 {
            Some(RelayServiceCode::new(r.l2_id("relayServiceCode")?.value()))
        } else {
            None
        };
        Ok(Self {
            message_type,
            source_l2_id,
            prose_app_code,
            relay_service_code,
        })
    }
}

/// Any PC5-S message, as a peer receives it before it knows which one it is.
///
/// The receive path needs this because a PC5 destination Layer-2 ID carries every
/// signalling message for that link: a UE reading its signalling reception ID
/// (TS 23.304 §5.8.2.4) cannot know whether the next message is a request, an accept
/// or a release until it has read octet 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pc5SMessage {
    /// A `DIRECT COMMUNICATION REQUEST`.
    Request(DirectCommunicationRequest),
    /// A `DIRECT COMMUNICATION ACCEPT`.
    Accept(DirectCommunicationAccept),
    /// A `DIRECT COMMUNICATION REJECT`.
    Reject(DirectCommunicationReject),
    /// A `DIRECT COMMUNICATION RELEASE`.
    Release(DirectCommunicationRelease),
    /// A Model A announcement, or a Model B solicitation or response.
    Discovery(DirectDiscoveryMessage),
}

impl Pc5SMessage {
    /// Decodes whichever PC5-S message the bytes carry.
    pub fn decode(bytes: &[u8]) -> Result<Self, Pc5SError> {
        let message_type = Reader::new(bytes).message_type()?;
        match message_type {
            Pc5SMessageType::DirectCommunicationRequest => {
                DirectCommunicationRequest::decode(bytes).map(Self::Request)
            }
            Pc5SMessageType::DirectCommunicationAccept => {
                DirectCommunicationAccept::decode(bytes).map(Self::Accept)
            }
            Pc5SMessageType::DirectCommunicationReject => {
                DirectCommunicationReject::decode(bytes).map(Self::Reject)
            }
            Pc5SMessageType::DirectCommunicationRelease => {
                DirectCommunicationRelease::decode(bytes).map(Self::Release)
            }
            Pc5SMessageType::DirectDiscoveryAnnouncement
            | Pc5SMessageType::DirectDiscoverySolicitation
            | Pc5SMessageType::DirectDiscoveryResponse => {
                DirectDiscoveryMessage::decode(bytes).map(Self::Discovery)
            }
        }
    }

    /// Re-encodes the message.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Self::Request(m) => m.encode(),
            Self::Accept(m) => m.encode(),
            Self::Reject(m) => m.encode(),
            Self::Release(m) => m.encode(),
            Self::Discovery(m) => m.encode(),
        }
    }

    /// The message type octet this message bears.
    pub fn message_type(&self) -> Pc5SMessageType {
        match self {
            Self::Request(_) => Pc5SMessageType::DirectCommunicationRequest,
            Self::Accept(_) => Pc5SMessageType::DirectCommunicationAccept,
            Self::Reject(_) => Pc5SMessageType::DirectCommunicationReject,
            Self::Release(_) => Pc5SMessageType::DirectCommunicationRelease,
            Self::Discovery(m) => m.message_type,
        }
    }
}

/// Appends a one-octet length followed by that many octets.
///
/// Values longer than 255 octets are truncated rather than rejected: the two IEs that
/// use this (`ProSe Service Info`, `Security Information`) are single-octet-length in
/// TS 24.554, so a longer value is not representable, and the alternative — failing
/// the *encode* — would drop a link establishment over a field neither end reads.
/// Truncation is recorded in the length octet, so a decoder sees exactly what was
/// sent.
fn push_lv(out: &mut Vec<u8>, value: &[u8]) {
    let len = value.len().min(usize::from(u8::MAX));
    out.push(len as u8);
    out.extend_from_slice(&value[..len]);
}

/// A cursor over a PC5-S message, so each field's bounds check names the field it was
/// reading rather than producing a bare index panic.
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn take(&mut self, n: usize, field: &'static str) -> Result<&'a [u8], Pc5SError> {
        if self.remaining() < n {
            return Err(Pc5SError::Truncated {
                field,
                needed: n,
                available: self.remaining(),
            });
        }
        let slice = &self.bytes[self.offset..self.offset + n];
        self.offset += n;
        Ok(slice)
    }

    fn octet(&mut self, field: &'static str) -> Result<u8, Pc5SError> {
        Ok(self.take(1, field)?[0])
    }

    fn message_type(&mut self) -> Result<Pc5SMessageType, Pc5SError> {
        let octet = self.octet("messageType")?;
        Pc5SMessageType::from_octet(octet).ok_or(Pc5SError::UnknownMessageType(octet))
    }

    fn expect_type(&mut self, expected: Pc5SMessageType) -> Result<(), Pc5SError> {
        let actual = self.message_type()?;
        if actual == expected {
            Ok(())
        } else {
            Err(Pc5SError::UnexpectedMessageType { expected, actual })
        }
    }

    fn l2_id(&mut self, field: &'static str) -> Result<ProseL2Id, Pc5SError> {
        let octets = self.take(3, field)?;
        Ok(ProseL2Id::from_octets([octets[0], octets[1], octets[2]]))
    }

    fn u32_be(&mut self, field: &'static str) -> Result<u32, Pc5SError> {
        let octets = self.take(4, field)?;
        Ok(u32::from_be_bytes([
            octets[0], octets[1], octets[2], octets[3],
        ]))
    }

    fn lv(&mut self, field: &'static str) -> Result<Vec<u8>, Pc5SError> {
        let len = usize::from(self.octet(field)?);
        if self.remaining() < len {
            return Err(Pc5SError::BadLength {
                field,
                declared: len,
                remaining: self.remaining(),
            });
        }
        Ok(self.take(len, field)?.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Layer-2 IDs used throughout these tests. Distinct in every octet, so a
    /// byte-order slip or a swapped source/target cannot produce the expected value.
    const UE1_L2_ID: u32 = 0x0A_1B_2C;
    const UE2_L2_ID: u32 = 0x03_04_05;

    #[test]
    fn a_layer_2_id_is_masked_to_24_bits_at_construction() {
        // The high octet is discarded rather than corrupting the top of the ID.
        assert_eq!(ProseL2Id::new(0xFF_12_34_56).value(), 0x12_34_56);
        assert_eq!(RelayServiceCode::new(0xFF_00_00_01).value(), 0x00_00_01);
    }

    #[test]
    fn a_layer_2_id_survives_three_octets_most_significant_first() {
        let id = ProseL2Id::new(UE1_L2_ID);
        assert_eq!(id.to_octets(), [0x0A, 0x1B, 0x2C]);
        assert_eq!(ProseL2Id::from_octets([0x0A, 0x1B, 0x2C]), id);
    }

    /// A UE-oriented request (TS 23.304 §6.4.3.1 step 5a): the target is named, so
    /// only that UE will accept. Every IE is read back by value.
    #[test]
    fn a_ue_oriented_request_round_trips_with_its_target_and_security_info() {
        let req = DirectCommunicationRequest {
            source_l2_id: ProseL2Id::new(UE1_L2_ID),
            target_l2_id: Some(ProseL2Id::new(UE2_L2_ID)),
            relay_service_code: Some(RelayServiceCode::new(0x00_12_34)),
            prose_service_info: vec![0xDE, 0xAD],
            security_info: vec![0xBE, 0xEF, 0x01],
        };
        let decoded = DirectCommunicationRequest::decode(&req.encode()).expect("decode");
        assert_eq!(decoded, req);
        // Positive, not merely "it parsed": the target the responder matches on.
        assert_eq!(decoded.target_l2_id, Some(ProseL2Id::new(UE2_L2_ID)));
        assert_eq!(decoded.security_info, vec![0xBE, 0xEF, 0x01]);
    }

    /// A ProSe-service-oriented request (step 5b) has NO target. "Absent" has to be
    /// distinguishable from "present and zero", because the two select different
    /// responder behaviour.
    #[test]
    fn a_service_oriented_request_has_an_absent_target_not_a_zero_one() {
        let req = DirectCommunicationRequest {
            source_l2_id: ProseL2Id::new(UE1_L2_ID),
            target_l2_id: None,
            relay_service_code: None,
            prose_service_info: vec![0x07],
            security_info: Vec::new(),
        };
        let decoded = DirectCommunicationRequest::decode(&req.encode()).expect("decode");
        assert_eq!(decoded.target_l2_id, None);

        let zero_target = DirectCommunicationRequest {
            target_l2_id: Some(ProseL2Id::new(0)),
            ..req.clone()
        };
        let zero_decoded =
            DirectCommunicationRequest::decode(&zero_target.encode()).expect("decode");
        assert_eq!(zero_decoded.target_l2_id, Some(ProseL2Id::new(0)));
        // The two must not encode alike, or a responder cannot tell 5a from 5b.
        assert_ne!(req.encode(), zero_target.encode());
    }

    /// The accept carries the responder's Layer-2 ID, which is the value
    /// TS 23.304 §6.4.3.1 step 4 says the initiator obtains "for future
    /// communication".
    #[test]
    fn an_accept_carries_the_responder_layer_2_id_and_pfi() {
        let accept = DirectCommunicationAccept {
            source_l2_id: ProseL2Id::new(UE2_L2_ID),
            pfi: 5,
            acting_as_relay: true,
        };
        let decoded = DirectCommunicationAccept::decode(&accept.encode()).expect("decode");
        assert_eq!(decoded.source_l2_id, ProseL2Id::new(UE2_L2_ID));
        assert_eq!(decoded.pfi, 5);
        assert!(decoded.acting_as_relay);
    }

    #[test]
    fn a_reject_carries_its_cause() {
        let reject = DirectCommunicationReject {
            source_l2_id: ProseL2Id::new(UE2_L2_ID),
            cause: Pc5RejectCause::TargetUserMismatch,
        };
        let decoded = DirectCommunicationReject::decode(&reject.encode()).expect("decode");
        assert_eq!(decoded.cause, Pc5RejectCause::TargetUserMismatch);
    }

    #[test]
    fn a_release_carries_the_releasing_layer_2_id() {
        let release = DirectCommunicationRelease {
            source_l2_id: ProseL2Id::new(UE1_L2_ID),
        };
        let decoded = DirectCommunicationRelease::decode(&release.encode()).expect("decode");
        assert_eq!(decoded.source_l2_id, ProseL2Id::new(UE1_L2_ID));
    }

    /// Model A and Model B differ in message type, and only the solicitation expects
    /// an answer. That asymmetry is the whole distinction between the two models.
    #[test]
    fn only_a_model_b_solicitation_expects_a_response() {
        let announce = DirectDiscoveryMessage::announcement(ProseL2Id::new(UE1_L2_ID), 0x1234);
        let solicit = DirectDiscoveryMessage::solicitation(ProseL2Id::new(UE1_L2_ID), 0x1234);
        let response = DirectDiscoveryMessage::response(ProseL2Id::new(UE2_L2_ID), 0x1234);

        assert!(!announce.expects_response());
        assert!(solicit.expects_response());
        assert!(!response.expects_response());

        // And the three are distinguishable on the wire.
        assert_ne!(announce.encode()[0], solicit.encode()[0]);
        assert_ne!(solicit.encode()[0], response.encode()[0]);
    }

    #[test]
    fn a_relay_announcement_round_trips_its_relay_service_code() {
        let rsc = RelayServiceCode::new(0x00_AB_CD);
        let announce = DirectDiscoveryMessage::announcement(ProseL2Id::new(UE2_L2_ID), 0xCAFE_0001)
            .with_relay_service_code(rsc);
        let decoded = DirectDiscoveryMessage::decode(&announce.encode()).expect("decode");
        assert_eq!(decoded.relay_service_code, Some(rsc));
        // The app code is a full 32 bits and must not be truncated to 24 like an L2 ID.
        assert_eq!(decoded.prose_app_code, 0xCAFE_0001);
    }

    /// The untyped receive path: a UE reading its signalling destination Layer-2 ID
    /// resolves the message from octet 1 alone.
    #[test]
    fn the_untyped_receive_path_resolves_each_message_from_octet_one() {
        let cases: Vec<(Vec<u8>, Pc5SMessageType)> = vec![
            (
                DirectCommunicationRequest {
                    source_l2_id: ProseL2Id::new(UE1_L2_ID),
                    target_l2_id: None,
                    relay_service_code: None,
                    prose_service_info: Vec::new(),
                    security_info: Vec::new(),
                }
                .encode(),
                Pc5SMessageType::DirectCommunicationRequest,
            ),
            (
                DirectCommunicationAccept {
                    source_l2_id: ProseL2Id::new(UE2_L2_ID),
                    pfi: 1,
                    acting_as_relay: false,
                }
                .encode(),
                Pc5SMessageType::DirectCommunicationAccept,
            ),
            (
                DirectCommunicationReject {
                    source_l2_id: ProseL2Id::new(UE2_L2_ID),
                    cause: Pc5RejectCause::InsufficientResources,
                }
                .encode(),
                Pc5SMessageType::DirectCommunicationReject,
            ),
            (
                DirectCommunicationRelease {
                    source_l2_id: ProseL2Id::new(UE1_L2_ID),
                }
                .encode(),
                Pc5SMessageType::DirectCommunicationRelease,
            ),
            (
                DirectDiscoveryMessage::solicitation(ProseL2Id::new(UE1_L2_ID), 9).encode(),
                Pc5SMessageType::DirectDiscoverySolicitation,
            ),
        ];

        for (bytes, expected) in cases {
            let msg = Pc5SMessage::decode(&bytes).expect("decode");
            assert_eq!(msg.message_type(), expected);
            // Re-encoding reproduces the same octets, so the decode lost nothing.
            assert_eq!(msg.encode(), bytes);
        }
    }

    #[test]
    fn an_unimplemented_message_type_is_named_rather_than_guessed() {
        // 0x09 is inside the PC5-S message type space but is not one of the seven
        // procedures this module implements (e.g. link modification).
        assert_eq!(
            Pc5SMessage::decode(&[0x09, 0x00, 0x00, 0x01]),
            Err(Pc5SError::UnknownMessageType(0x09))
        );
    }

    #[test]
    fn a_truncated_message_names_the_field_it_ran_out_on() {
        // An accept needs 6 octets; give it 4, so it dies reading the PFI.
        let err = DirectCommunicationAccept::decode(&[0x02, 0x00, 0x00, 0x01]).unwrap_err();
        assert_eq!(
            err,
            Pc5SError::Truncated {
                field: "pfi",
                needed: 1,
                available: 0
            }
        );
    }

    #[test]
    fn an_ie_declaring_more_octets_than_remain_is_rejected() {
        // Request: type, 3-octet source, flags=0, then proseServiceInfo length 0x10
        // with nothing behind it.
        let err =
            DirectCommunicationRequest::decode(&[0x01, 0x00, 0x00, 0x01, 0x00, 0x10]).unwrap_err();
        assert_eq!(
            err,
            Pc5SError::BadLength {
                field: "proseServiceInfo",
                declared: 16,
                remaining: 0
            }
        );
    }

    #[test]
    fn decoding_one_message_as_another_names_both_types() {
        let accept_bytes = DirectCommunicationAccept {
            source_l2_id: ProseL2Id::new(UE2_L2_ID),
            pfi: 1,
            acting_as_relay: false,
        }
        .encode();
        assert_eq!(
            DirectCommunicationRequest::decode(&accept_bytes),
            Err(Pc5SError::UnexpectedMessageType {
                expected: Pc5SMessageType::DirectCommunicationRequest,
                actual: Pc5SMessageType::DirectCommunicationAccept,
            })
        );
    }
}
