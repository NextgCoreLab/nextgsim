//! LPP messages the UE decodes and produces (3GPP TS 37.355), UNALIGNED PER.
//!
//! The subset issue #46 asks for: the `LPP-Message` envelope, the capability
//! transfer pair (§5.1.3) and the location-information transfer pair (§5.3.3) for
//! **E-CID**. Everything else in the `c1` CHOICE is decoded far enough to be
//! recognised and then reported as unsupported, rather than mistaken for something
//! it is not.
//!
//! ## Why E-CID and not A-GNSS
//!
//! #46 accepts either. E-CID is the one a simulator can answer *honestly*: its
//! measurements are the serving cell's PCI, RSRP and RSRQ, which this UE genuinely
//! has. A-GNSS would need a GNSS receiver model, so every field would be invented.
//!
//! ## Byte agreement with the LMF
//!
//! Every layout below mirrors `nextgcore`'s `nextgcore-asn1c::lpp`, which is the
//! peer that sends these. The two are separate products, so this is a second
//! implementation of one wire format by necessity rather than duplication within a
//! repo — and each layout comment names the clause that fixes it, because two
//! implementations agreeing *wrongly* is the failure this cannot detect on its own.

use super::uper::{UperError, UperReader, UperResult, UperWriter};

/// `Initiator ::= ENUMERATED { locationServer, targetDevice, ... }` — extensible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Initiator {
    /// The location server (LMF) opened the transaction.
    LocationServer,
    /// The target device (this UE) opened it.
    TargetDevice,
}

impl Initiator {
    const ROOT_MAX: i64 = 1;

    fn write(self, w: &mut UperWriter) -> UperResult<()> {
        w.write_extensible_enumerated(
            match self {
                Self::LocationServer => 0,
                Self::TargetDevice => 1,
            },
            Self::ROOT_MAX,
        )
    }

    fn read(r: &mut UperReader<'_>) -> UperResult<Self> {
        match r.read_extensible_enumerated(Self::ROOT_MAX)? {
            0 => Ok(Self::LocationServer),
            1 => Ok(Self::TargetDevice),
            _ => Err(UperError::Unsupported("Initiator value")),
        }
    }
}

/// `LPP-TransactionID ::= SEQUENCE { initiator, transactionNumber, ... }`.
///
/// Extensible with no OPTIONAL root members, so its preamble is one extension bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LppTransactionId {
    /// Which side opened the transaction.
    pub initiator: Initiator,
    /// `TransactionNumber ::= INTEGER (0..255)`.
    pub transaction_number: u8,
}

impl LppTransactionId {
    fn write(&self, w: &mut UperWriter) -> UperResult<()> {
        w.write_sequence_preamble(Some(false), &[]);
        self.initiator.write(w)?;
        w.write_constrained(i64::from(self.transaction_number), 0, 255)
    }

    fn read(r: &mut UperReader<'_>) -> UperResult<Self> {
        let (additions, _) = r.read_sequence_preamble(true, 0)?;
        let initiator = Initiator::read(r)?;
        let transaction_number = r.read_constrained(0, 255)? as u8;
        if additions {
            r.skip_extension_additions()?;
        }
        Ok(Self {
            initiator,
            transaction_number,
        })
    }
}

/// The `c1` alternatives of `LPP-MessageBody` this UE understands, by their
/// TS 37.355 index.
///
/// The indices are the spec's own: 0/1 are the capability pair, 4/5 the
/// location-information pair. They are named as constants because an off-by-one
/// here turns a capability request into an assistance-data request, which decodes
/// without complaint and answers the wrong question.
mod c1 {
    pub const ALTERNATIVES: usize = 16;
    pub const REQUEST_CAPABILITIES: usize = 0;
    pub const PROVIDE_CAPABILITIES: usize = 1;
    pub const REQUEST_LOCATION_INFORMATION: usize = 4;
    pub const PROVIDE_LOCATION_INFORMATION: usize = 5;
}

/// One `MeasuredResultsElement` of an E-CID measurement report.
///
/// `cellGlobalId` is always absent, matching the peer: it is a
/// `CellGlobalIdEUTRA-AndUTRA`, and this UE has no E-UTRA/UTRA identity to put in
/// it. Absent is the honest encoding; a fabricated one would name a cell that does
/// not exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasuredResultsElement {
    /// `physCellId ::= INTEGER (0..503)`.
    pub phys_cell_id: u16,
    /// `arfcnEUTRA ::= INTEGER (0..65535)`.
    pub arfcn: u32,
    /// `systemFrameNumber ::= BIT STRING (SIZE(10))`.
    pub system_frame_number: Option<u16>,
    /// `rsrp-Result ::= INTEGER (0..97)`.
    pub rsrp_result: Option<u8>,
    /// `rsrq-Result ::= INTEGER (0..34)`.
    pub rsrq_result: Option<u8>,
    /// `ue-RxTxTimeDiff ::= INTEGER (0..4095)`.
    pub ue_rx_tx_time_diff: Option<u16>,
}

impl MeasuredResultsElement {
    const SFN_BITS: usize = 10;

    fn write(&self, w: &mut UperWriter) -> UperResult<()> {
        w.write_sequence_preamble(
            Some(false),
            &[
                false, // cellGlobalId: deliberately absent
                self.system_frame_number.is_some(),
                self.rsrp_result.is_some(),
                self.rsrq_result.is_some(),
                self.ue_rx_tx_time_diff.is_some(),
            ],
        );
        w.write_constrained(i64::from(self.phys_cell_id), 0, 503)?;
        w.write_constrained(i64::from(self.arfcn), 0, 65_535)?;
        if let Some(sfn) = self.system_frame_number {
            let bits: Vec<bool> = (0..Self::SFN_BITS)
                .rev()
                .map(|index| (sfn >> index) & 1 == 1)
                .collect();
            w.write_bit_string(&bits, Self::SFN_BITS, Self::SFN_BITS)?;
        }
        if let Some(rsrp) = self.rsrp_result {
            w.write_constrained(i64::from(rsrp), 0, 97)?;
        }
        if let Some(rsrq) = self.rsrq_result {
            w.write_constrained(i64::from(rsrq), 0, 34)?;
        }
        if let Some(rxtx) = self.ue_rx_tx_time_diff {
            w.write_constrained(i64::from(rxtx), 0, 4095)?;
        }
        Ok(())
    }

    fn read(r: &mut UperReader<'_>) -> UperResult<Self> {
        let (additions, opts) = r.read_sequence_preamble(true, 5)?;
        if opts[0] {
            // A peer that sent cellGlobalId is refused rather than having its
            // remaining fields read at the wrong offsets.
            return Err(UperError::Unsupported(
                "MeasuredResultsElement cellGlobalId",
            ));
        }
        let phys_cell_id = r.read_constrained(0, 503)? as u16;
        let arfcn = r.read_constrained(0, 65_535)? as u32;
        let system_frame_number = if opts[1] {
            let bits = r.read_bit_string(Self::SFN_BITS, Self::SFN_BITS)?;
            Some(bits.iter().fold(0u16, |acc, &b| (acc << 1) | u16::from(b)))
        } else {
            None
        };
        let rsrp_result = if opts[2] {
            Some(r.read_constrained(0, 97)? as u8)
        } else {
            None
        };
        let rsrq_result = if opts[3] {
            Some(r.read_constrained(0, 34)? as u8)
        } else {
            None
        };
        let ue_rx_tx_time_diff = if opts[4] {
            Some(r.read_constrained(0, 4095)? as u16)
        } else {
            None
        };
        if additions {
            r.skip_extension_additions()?;
        }
        Ok(Self {
            phys_cell_id,
            arfcn,
            system_frame_number,
            rsrp_result,
            rsrq_result,
            ue_rx_tx_time_diff,
        })
    }
}

/// `ECID-SignalMeasurementInformation`: the primary cell's measurements plus a
/// `MeasuredResultsList ::= SEQUENCE (SIZE(1..32))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EcidSignalMeasurementInformation {
    /// The serving cell's measurements.
    pub primary_cell: Option<MeasuredResultsElement>,
    /// Neighbour measurements; 1..32 elements on the wire.
    pub measured_results: Vec<MeasuredResultsElement>,
}

impl EcidSignalMeasurementInformation {
    fn write(&self, w: &mut UperWriter) -> UperResult<()> {
        w.write_sequence_preamble(Some(false), &[self.primary_cell.is_some()]);
        if let Some(primary) = &self.primary_cell {
            primary.write(w)?;
        }
        // The list's own size constraint starts at ONE, so an empty report cannot be
        // encoded at all -- reported rather than sent as a zero length the peer
        // would read as something else.
        if self.measured_results.is_empty() || self.measured_results.len() > 32 {
            return Err(UperError::InvalidLength(self.measured_results.len()));
        }
        w.write_constrained(self.measured_results.len() as i64, 1, 32)?;
        for element in &self.measured_results {
            element.write(w)?;
        }
        Ok(())
    }

    fn read(r: &mut UperReader<'_>) -> UperResult<Self> {
        let (additions, opts) = r.read_sequence_preamble(true, 1)?;
        let primary_cell = if opts[0] {
            Some(MeasuredResultsElement::read(r)?)
        } else {
            None
        };
        let count = r.read_constrained(1, 32)? as usize;
        let mut measured_results = Vec::with_capacity(count);
        for _ in 0..count {
            measured_results.push(MeasuredResultsElement::read(r)?);
        }
        if additions {
            r.skip_extension_additions()?;
        }
        Ok(Self {
            primary_cell,
            measured_results,
        })
    }
}

/// The `requestedMeasurements` / `ecid-MeasSupported` bit map, `SIZE(1..8)`.
///
/// Bit 0 is RSRP, bit 1 RSRQ, bit 2 UE Rx-Tx. Modelled as booleans rather than a
/// raw byte so a caller cannot silently set a bit it does not mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EcidMeasurementBits {
    /// RSRP requested or supported.
    pub rsrp: bool,
    /// RSRQ requested or supported.
    pub rsrq: bool,
    /// UE Rx-Tx time difference requested or supported.
    pub ue_rx_tx: bool,
}

impl EcidMeasurementBits {
    /// The three modelled bits, which is a legal `SIZE(1..8)` length.
    fn as_bits(self) -> Vec<bool> {
        vec![self.rsrp, self.rsrq, self.ue_rx_tx]
    }

    fn from_bits(bits: &[bool]) -> Self {
        Self {
            rsrp: bits.first().copied().unwrap_or(false),
            rsrq: bits.get(1).copied().unwrap_or(false),
            ue_rx_tx: bits.get(2).copied().unwrap_or(false),
        }
    }

    fn write(self, w: &mut UperWriter) -> UperResult<()> {
        w.write_sequence_preamble(Some(false), &[]);
        w.write_bit_string(&self.as_bits(), 1, 8)
    }

    fn read(r: &mut UperReader<'_>) -> UperResult<Self> {
        let (additions, _) = r.read_sequence_preamble(true, 0)?;
        let bits = r.read_bit_string(1, 8)?;
        if additions {
            r.skip_extension_additions()?;
        }
        Ok(Self::from_bits(&bits))
    }
}

/// What a present `commonIEs` root member means for a given `-r9` body.
///
/// The distinction is not cosmetic: the two shapes have different *content*, so
/// treating one as the other misaligns every field after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommonIes {
    /// `CommonIEsRequestCapabilities` / `CommonIEsProvideCapabilities` are **empty**
    /// extensible SEQUENCEs, so a present one is a single extension bit and can be
    /// consumed without interpreting anything.
    EmptySequence,
    /// `CommonIEsRequestLocationInformation` / `...ProvideLocationInformation` carry
    /// real content (location-information type, QoS, reporting criteria) that this
    /// codec does not model, so a present one must be refused.
    Unmodelled,
}

/// The five root presence bits of an `-r9` body:
/// `[commonIEs, a-gnss, otdoa, ecid, epdu]`.
///
/// Only the E-CID slot is ever set on encode; a peer that sets A-GNSS, OTDOA or an
/// EPDU is reported rather than answered.
fn write_r9_body_preamble(w: &mut UperWriter, ecid_present: bool) {
    w.write_sequence_preamble(Some(false), &[false, false, false, ecid_present, false]);
}

/// Read an `-r9` body preamble, consuming a `commonIEs` this codec can skip.
///
/// Returns `(additions_follow, ecid_present)`.
fn read_r9_body_preamble(r: &mut UperReader<'_>, common: CommonIes) -> UperResult<(bool, bool)> {
    let (additions, opts) = r.read_sequence_preamble(true, 5)?;
    if opts[1] || opts[2] || opts[4] {
        // A-GNSS, OTDOA or an EPDU. Each has its own content, so continuing would
        // read later fields at the wrong offsets.
        return Err(UperError::Unsupported(
            "LPP body for a positioning method other than E-CID",
        ));
    }
    if opts[0] {
        match common {
            // The peer (nextgcore's LMF) does encode this member, and refusing it
            // would fail a capability transfer over a field with no content.
            CommonIes::EmptySequence => {
                let (inner_additions, _) = r.read_sequence_preamble(true, 0)?;
                if inner_additions {
                    r.skip_extension_additions()?;
                }
            }
            CommonIes::Unmodelled => {
                return Err(UperError::Unsupported(
                    "LPP commonIEs for location information",
                ));
            }
        }
    }
    Ok((additions, opts[3]))
}

/// The `criticalExtensions` wrapper every LPP body carries: a 2-alternative CHOICE
/// (`c1` / `criticalExtensionsFuture`) then a 4-alternative one selecting the `-r9`
/// variant.
fn write_critical_extensions(w: &mut UperWriter) -> UperResult<()> {
    w.write_choice_index(0, 2)?;
    w.write_choice_index(0, 4)
}

fn read_critical_extensions(r: &mut UperReader<'_>) -> UperResult<()> {
    if r.read_choice_index(2)? != 0 {
        return Err(UperError::Unsupported("criticalExtensionsFuture"));
    }
    if r.read_choice_index(4)? != 0 {
        return Err(UperError::Unsupported(
            "a non-r9 criticalExtensions variant",
        ));
    }
    Ok(())
}

/// How a sidelink range was measured (issue #137).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidelinkRangingMethod {
    /// Round-trip time over SL-PRS. Metre-level.
    Rtt,
    /// Multi-frequency carrier phase, ambiguity resolved against the RTT.
    /// Centimetre-level.
    CarrierPhase,
}

impl SidelinkRangingMethod {
    /// Root index in the extensible ENUMERATED.
    const ROOT_MAX: i64 = 1;

    fn index(self) -> i64 {
        match self {
            Self::Rtt => 0,
            Self::CarrierPhase => 1,
        }
    }

    fn from_index(index: i64) -> UperResult<Self> {
        match index {
            0 => Ok(Self::Rtt),
            1 => Ok(Self::CarrierPhase),
            _ => Err(UperError::Unsupported("sidelink ranging method")),
        }
    }
}

/// One measured range to one sidelink peer (TS 23.586 §5.3.3, issue #137).
///
/// ## Why the units are centimetres
///
/// The carrier-phase estimate this carries is centimetre-level, so a range in
/// whole metres would throw away the precision the whole widelane combination
/// exists to produce, and an LMF could not tell an RTT fix from a carrier-phase one
/// by looking at the number. Encoding a float was rejected: PER has no IEEE-754
/// primitive here, and a scaled integer is what every other distance IE in this
/// tree uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SidelinkRangingResult {
    /// The peer's PC5 Layer-2 identity, `INTEGER (0..16777215)` — 24 bits, which is
    /// what TS 23.303 §8.2 gives a Layer-2 ID. A UE id wider than that is refused
    /// rather than truncated onto another peer's identity.
    pub peer_layer2_id: u32,
    /// Range in centimetres, `INTEGER (0..1000000)`: 0..10 km at 1 cm resolution.
    pub range_cm: u32,
    /// Reported accuracy in centimetres, `INTEGER (0..65535)`.
    pub accuracy_cm: u16,
    /// How it was measured.
    pub method: SidelinkRangingMethod,
    /// How many measurements the estimate rests on, `INTEGER (0..65535)`.
    ///
    /// Carried because it is the difference between a range from one occasion and
    /// one from fifty, and an LMF weighing two reports has nothing else to go on.
    pub measurement_count: u16,
}

impl SidelinkRangingResult {
    const PEER_ID_MAX: i64 = 16_777_215;
    const RANGE_CM_MAX: i64 = 1_000_000;
    const ACCURACY_CM_MAX: i64 = 65_535;
    const COUNT_MAX: i64 = 65_535;

    fn write(&self, w: &mut UperWriter) -> UperResult<()> {
        w.write_sequence_preamble(Some(false), &[]);
        w.write_constrained(i64::from(self.peer_layer2_id), 0, Self::PEER_ID_MAX)?;
        w.write_constrained(i64::from(self.range_cm), 0, Self::RANGE_CM_MAX)?;
        w.write_constrained(i64::from(self.accuracy_cm), 0, Self::ACCURACY_CM_MAX)?;
        w.write_extensible_enumerated(self.method.index(), SidelinkRangingMethod::ROOT_MAX)?;
        w.write_constrained(i64::from(self.measurement_count), 0, Self::COUNT_MAX)
    }

    fn read(r: &mut UperReader<'_>) -> UperResult<Self> {
        let (additions, _) = r.read_sequence_preamble(true, 0)?;
        let peer_layer2_id = r.read_constrained(0, Self::PEER_ID_MAX)? as u32;
        let range_cm = r.read_constrained(0, Self::RANGE_CM_MAX)? as u32;
        let accuracy_cm = r.read_constrained(0, Self::ACCURACY_CM_MAX)? as u16;
        let method = SidelinkRangingMethod::from_index(
            r.read_extensible_enumerated(SidelinkRangingMethod::ROOT_MAX)?,
        )?;
        let measurement_count = r.read_constrained(0, Self::COUNT_MAX)? as u16;
        if additions {
            r.skip_extension_additions()?;
        }
        Ok(Self {
            peer_layer2_id,
            range_cm,
            accuracy_cm,
            method,
            measurement_count,
        })
    }
}

/// Which extension-addition group of `ProvideLocationInformation-r9-IEs` carries the
/// sidelink ranging report: the FOURTH, index 3.
///
/// TS 37.355 declares three groups on that sequence — G1(r13), G2(r16), G3(r19) — so
/// a simulator-defined IE must sit past them rather than squat on one. G2 in
/// particular is where the peer LMF reads nr-Multi-RTT and nr-DL-TDOA; G1 and G3 it
/// ignores, which is exactly why putting the report in G1 would have looked like it
/// worked from this end and delivered nothing.
const SIDELINK_ADDITION_GROUP: usize = 3;

/// The sidelink ranging report a UE sends to the LMF (issue #137).
///
/// Carried as an **extension addition** of `ProvideLocationInformation-r9-IEs`,
/// which is where a Rel-18 IE belongs in a Rel-9 root sequence: a root member
/// cannot be added without changing the byte layout of every existing LPP message,
/// and this codec's peer is nextgcore's LMF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidelinkRangingReport {
    /// One entry per ranged peer, `SEQUENCE (SIZE(1..32))`.
    pub results: Vec<SidelinkRangingResult>,
}

impl SidelinkRangingReport {
    const MIN_RESULTS: usize = 1;
    /// `SEQUENCE (SIZE(1..32))`. Public because the caller building a report has to
    /// bound its own list to it: 33 results encode to bytes that then fail to
    /// decode, and the sender would see no error.
    pub const MAX_RESULTS: usize = 32;

    /// Encode as a standalone open-type payload: a complete UPER encoding padded to
    /// whole octets, which is what an extension addition is.
    pub fn encode(&self) -> UperResult<Vec<u8>> {
        if self.results.len() < Self::MIN_RESULTS || self.results.len() > Self::MAX_RESULTS {
            return Err(UperError::InvalidLength(self.results.len()));
        }
        let mut w = UperWriter::new();
        w.write_sequence_preamble(Some(false), &[]);
        w.write_constrained(
            self.results.len() as i64,
            Self::MIN_RESULTS as i64,
            Self::MAX_RESULTS as i64,
        )?;
        for result in &self.results {
            result.write(&mut w)?;
        }
        Ok(w.into_bytes())
    }

    /// The most octets an extension addition can carry: an open type's length
    /// determinant is written in the SHORT form here (one octet, values below 128),
    /// which is all the matching reader can read.
    pub const MAX_ADDITION_OCTETS: usize = 0x7F;

    /// Build a report from `results`, dropping trailing entries until it fits an
    /// extension addition, and say how many were dropped.
    ///
    /// The bound is found by ENCODING rather than by arithmetic over the field
    /// widths, so it cannot drift from the layout: a result gained a field, and a
    /// hand-computed capacity would silently become wrong while still looking
    /// derived. Roughly 12 results fit today, which is well under the ASN.1
    /// `SIZE(1..32)` — so the size bound is NOT the binding one, and a caller that
    /// only checked it would emit an addition the peer cannot read.
    ///
    /// Returns `None` for an empty input, and for the case where even one result does
    /// not fit — which cannot happen at the current field widths but is not asserted
    /// away, because that is exactly what changes when a field is added.
    pub fn fitting_an_addition(results: &[SidelinkRangingResult]) -> Option<(Self, usize)> {
        let mut kept = results.len().min(Self::MAX_RESULTS).min(results.len());
        while kept > 0 {
            let candidate = Self {
                results: results[..kept].to_vec(),
            };
            match candidate.encode() {
                Ok(bytes) if bytes.len() <= Self::MAX_ADDITION_OCTETS => {
                    return Some((candidate, results.len() - kept));
                }
                _ => kept -= 1,
            }
        }
        None
    }

    /// Decode an open-type payload produced by [`Self::encode`].
    pub fn decode(bytes: &[u8]) -> UperResult<Self> {
        let mut r = UperReader::new(bytes);
        let (additions, _) = r.read_sequence_preamble(true, 0)?;
        let count =
            r.read_constrained(Self::MIN_RESULTS as i64, Self::MAX_RESULTS as i64)? as usize;
        let mut results = Vec::with_capacity(count);
        for _ in 0..count {
            results.push(SidelinkRangingResult::read(&mut r)?);
        }
        if additions {
            r.skip_extension_additions()?;
        }
        Ok(Self { results })
    }
}

/// The LPP message bodies this UE handles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LppBody {
    /// `RequestCapabilities` (§5.1.3), with the E-CID request present or not.
    RequestCapabilities {
        /// Whether the server asked about E-CID specifically.
        ecid_requested: bool,
    },
    /// `ProvideCapabilities` (§5.1.3).
    ProvideCapabilities {
        /// The E-CID measurements this UE supports.
        ecid_supported: EcidMeasurementBits,
    },
    /// `RequestLocationInformation` (§5.3.3).
    RequestLocationInformation {
        /// The E-CID measurements the server asked for.
        requested: EcidMeasurementBits,
    },
    /// `ProvideLocationInformation` (§5.3.3).
    ProvideLocationInformation {
        /// The E-CID measurement report. Always present: a UE that ranges over
        /// sidelink also has a serving cell, so reporting both is accurate rather
        /// than padding.
        measurements: EcidSignalMeasurementInformation,
        /// The sidelink ranging report, when this UE has ranges to report
        /// (issue #137). `None` leaves the encoding byte-for-byte what it was
        /// before sidelink existed -- the extension bit stays clear.
        sidelink: Option<SidelinkRangingReport>,
    },
}

impl LppBody {
    fn write(&self, w: &mut UperWriter) -> UperResult<()> {
        // LPP-MessageBody CHOICE: c1 (index 0 of 2), then the c1 index.
        w.write_choice_index(0, 2)?;
        match self {
            Self::RequestCapabilities { ecid_requested } => {
                w.write_choice_index(c1::REQUEST_CAPABILITIES, c1::ALTERNATIVES)?;
                write_critical_extensions(w)?;
                write_r9_body_preamble(w, *ecid_requested);
                if *ecid_requested {
                    // ECID-RequestCapabilities is an empty extensible SEQUENCE.
                    w.write_sequence_preamble(Some(false), &[]);
                }
                Ok(())
            }
            Self::ProvideCapabilities { ecid_supported } => {
                w.write_choice_index(c1::PROVIDE_CAPABILITIES, c1::ALTERNATIVES)?;
                write_critical_extensions(w)?;
                write_r9_body_preamble(w, true);
                ecid_supported.write(w)
            }
            Self::RequestLocationInformation { requested } => {
                w.write_choice_index(c1::REQUEST_LOCATION_INFORMATION, c1::ALTERNATIVES)?;
                write_critical_extensions(w)?;
                write_r9_body_preamble(w, true);
                requested.write(w)
            }
            Self::ProvideLocationInformation {
                measurements,
                sidelink,
            } => {
                w.write_choice_index(c1::PROVIDE_LOCATION_INFORMATION, c1::ALTERNATIVES)?;
                write_critical_extensions(w)?;
                // The extension bit is set only when there is an addition to carry,
                // so a report without sidelink results encodes exactly as it did
                // before this existed.
                w.write_sequence_preamble(
                    Some(sidelink.is_some()),
                    &[false, false, false, true, false],
                );
                // ECID-ProvideLocationInformation: [signalMeasurementInformation,
                // ecid-Error], and only the measurements are produced here.
                w.write_sequence_preamble(Some(false), &[true, false]);
                measurements.write(w)?;
                // Additions come after every root member, which is why this is here
                // and not beside the preamble.
                //
                // The report is addition GROUP 4, not the first addition. TS 37.355
                // declares three groups on this sequence -- G1(r13), G2(r16),
                // G3(r19) -- and the peer LMF reads nr-Multi-RTT and nr-DL-TDOA out
                // of G2 and *ignores* G1 and G3 (see nextgcore-asn1c
                // `lpp::ecid::ProvideLocationInformationR9`). Writing the report as
                // the first addition would put it in G1, where the LMF would drop it
                // silently, and would also claim a group the spec defines.
                if let Some(report) = sidelink {
                    let mut groups = vec![None; SIDELINK_ADDITION_GROUP];
                    groups.push(Some(report.encode()?));
                    w.write_extension_additions(&groups)?;
                }
                Ok(())
            }
        }
    }

    fn read(r: &mut UperReader<'_>) -> UperResult<Self> {
        if r.read_choice_index(2)? != 0 {
            return Err(UperError::Unsupported("messageClassExtension"));
        }
        let index = r.read_choice_index(c1::ALTERNATIVES)?;
        match index {
            c1::REQUEST_CAPABILITIES => {
                read_critical_extensions(r)?;
                let (additions, ecid) = read_r9_body_preamble(r, CommonIes::EmptySequence)?;
                if ecid {
                    let (inner_additions, _) = r.read_sequence_preamble(true, 0)?;
                    if inner_additions {
                        r.skip_extension_additions()?;
                    }
                }
                if additions {
                    r.skip_extension_additions()?;
                }
                Ok(Self::RequestCapabilities {
                    ecid_requested: ecid,
                })
            }
            c1::PROVIDE_CAPABILITIES => {
                read_critical_extensions(r)?;
                let (additions, ecid) = read_r9_body_preamble(r, CommonIes::EmptySequence)?;
                let ecid_supported = if ecid {
                    EcidMeasurementBits::read(r)?
                } else {
                    EcidMeasurementBits::default()
                };
                if additions {
                    r.skip_extension_additions()?;
                }
                Ok(Self::ProvideCapabilities { ecid_supported })
            }
            c1::REQUEST_LOCATION_INFORMATION => {
                read_critical_extensions(r)?;
                let (additions, ecid) = read_r9_body_preamble(r, CommonIes::Unmodelled)?;
                let requested = if ecid {
                    EcidMeasurementBits::read(r)?
                } else {
                    // A request naming no method has nothing to answer. Reported
                    // rather than answered with a default, which would report
                    // measurements the server did not ask for.
                    return Err(UperError::Unsupported(
                        "RequestLocationInformation with no E-CID body",
                    ));
                };
                if additions {
                    r.skip_extension_additions()?;
                }
                Ok(Self::RequestLocationInformation { requested })
            }
            c1::PROVIDE_LOCATION_INFORMATION => {
                read_critical_extensions(r)?;
                let (additions, ecid) = read_r9_body_preamble(r, CommonIes::Unmodelled)?;
                if !ecid {
                    return Err(UperError::Unsupported(
                        "ProvideLocationInformation with no E-CID body",
                    ));
                }
                let (inner_additions, opts) = r.read_sequence_preamble(true, 2)?;
                let measurements = if opts[0] {
                    EcidSignalMeasurementInformation::read(r)?
                } else {
                    return Err(UperError::Unsupported(
                        "ProvideLocationInformation without signal measurements",
                    ));
                };
                if inner_additions {
                    r.skip_extension_additions()?;
                }
                // The report rides addition GROUP 4 (see the encoder). Groups are
                // read POSITIONALLY and a peer may send fewer -- the trailing-absent
                // ones are trimmed (X.691 18.8) -- so a message carrying only the
                // spec's G1..G3 decodes to no report rather than to an error.
                let sidelink = if additions {
                    match r
                        .read_extension_additions()?
                        .into_iter()
                        .nth(SIDELINK_ADDITION_GROUP)
                        .flatten()
                    {
                        Some(bytes) => Some(SidelinkRangingReport::decode(&bytes)?),
                        None => None,
                    }
                } else {
                    None
                };
                Ok(Self::ProvideLocationInformation {
                    measurements,
                    sidelink,
                })
            }
            other => Err(match other {
                2 | 3 => UperError::Unsupported("LPP assistance-data transfer"),
                6 => UperError::Unsupported("LPP abort"),
                7 => UperError::Unsupported("LPP error"),
                _ => UperError::Unsupported("a spare LPP c1 alternative"),
            }),
        }
    }
}

/// `LPP-Message ::= SEQUENCE { transactionID OPTIONAL, endTransaction,
/// sequenceNumber OPTIONAL, acknowledgement OPTIONAL, lpp-MessageBody OPTIONAL }`
/// — **non-extensible**.
///
/// The four presence bits come in declaration order, and `endTransaction` is
/// written **after** the preamble, between `transactionID` and `sequenceNumber`.
/// That ordering is the one thing about this envelope easy to get wrong, and it
/// shifts every later field by a bit when it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LppMessage {
    /// The transaction this message belongs to.
    pub transaction_id: Option<LppTransactionId>,
    /// Whether this message ends the transaction.
    pub end_transaction: bool,
    /// The body, when one is carried.
    pub body: Option<LppBody>,
}

impl LppMessage {
    /// Encode to UPER octets.
    ///
    /// # Errors
    /// [`UperError`] when a field is outside its constraint or the body is one this
    /// codec cannot produce.
    pub fn encode(&self) -> UperResult<Vec<u8>> {
        let mut w = UperWriter::new();
        w.write_sequence_preamble(
            None, // non-extensible: no extension bit
            &[
                self.transaction_id.is_some(),
                false, // sequenceNumber: not produced
                false, // acknowledgement: not produced
                self.body.is_some(),
            ],
        );
        if let Some(tid) = &self.transaction_id {
            tid.write(&mut w)?;
        }
        w.write_bit(self.end_transaction);
        if let Some(body) = &self.body {
            body.write(&mut w)?;
        }
        Ok(w.into_bytes())
    }

    /// Decode from UPER octets.
    ///
    /// # Errors
    /// [`UperError`] when the bytes are short, a constraint is violated, or the
    /// message names a construct this codec does not implement.
    pub fn decode(bytes: &[u8]) -> UperResult<Self> {
        let mut r = UperReader::new(bytes);
        let (_, opts) = r.read_sequence_preamble(false, 4)?;
        let transaction_id = if opts[0] {
            Some(LppTransactionId::read(&mut r)?)
        } else {
            None
        };
        let end_transaction = r.read_bit()?;
        if opts[1] {
            // SequenceNumber ::= INTEGER (0..255)
            let _ = r.read_constrained(0, 255)?;
        }
        if opts[2] {
            // Acknowledgement is a SEQUENCE this codec does not model; skipping it
            // by guesswork would misalign the body, so it is refused.
            return Err(UperError::Unsupported("LPP Acknowledgement"));
        }
        let body = if opts[3] {
            Some(LppBody::read(&mut r)?)
        } else {
            None
        };
        Ok(Self {
            transaction_id,
            end_transaction,
            body,
        })
    }
}
