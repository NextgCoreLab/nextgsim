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

/// The NR E-CID measurement bit map (Rel-16): `requestedMeasurements-r16` /
/// `nr-ECID-MeasSupported-r16`, `BIT STRING { ss-RSRP (0), ss-RSRQ (1), csi-RSRP
/// (2), csi-RSRQ (3) } (SIZE(1..8))` (#330).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NrEcidMeasurementBits {
    /// SS-RSRP requested or supported.
    pub ss_rsrp: bool,
    /// SS-RSRQ requested or supported.
    pub ss_rsrq: bool,
    /// CSI-RSRP requested or supported.
    pub csi_rsrp: bool,
    /// CSI-RSRQ requested or supported.
    pub csi_rsrq: bool,
}

impl NrEcidMeasurementBits {
    /// The bits up to the last one set, at least two: the two SS bits are what
    /// nextgcore's lmfd sends, so a UE that names only those encodes as it does.
    fn as_bits(self) -> Vec<bool> {
        let bits = [self.ss_rsrp, self.ss_rsrq, self.csi_rsrp, self.csi_rsrq];
        let len = if self.csi_rsrp || self.csi_rsrq { 4 } else { 2 };
        bits[..len].to_vec()
    }

    fn from_bits(bits: &[bool]) -> Self {
        Self {
            ss_rsrp: bits.first().copied().unwrap_or(false),
            ss_rsrq: bits.get(1).copied().unwrap_or(false),
            csi_rsrp: bits.get(2).copied().unwrap_or(false),
            csi_rsrq: bits.get(3).copied().unwrap_or(false),
        }
    }
}

/// `NR-MeasuredResultsElement-r16`, the members this UE fills (#330).
///
/// `nr-CellGlobalID-r16` and the CSI-RS and per-index results are always absent:
/// this UE knows its serving cell's PCI and SS levels and nothing more, so absent
/// is the honest encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NrMeasuredResultsElement {
    /// `NR-PhysCellID-r16 ::= INTEGER (0..1007)`.
    pub phys_cell_id: u16,
    /// `nr-ARFCN-r16`: the `ssb-ARFCN-r16` alternative, `ARFCN-ValueNR-r15 ::=
    /// INTEGER (0..3279165)`.
    pub ssb_arfcn: u32,
    /// `systemFrameNumber-r16 ::= BIT STRING (SIZE (10))`.
    pub system_frame_number: Option<u16>,
    /// `resultsSSB-Cell-r16.nr-RSRP-r16 ::= INTEGER (0..127)` (TS 38.133 §10.1.6.1).
    pub ss_rsrp: Option<u8>,
    /// `resultsSSB-Cell-r16.nr-RSRQ-r16 ::= INTEGER (0..127)` (TS 38.133 §10.1.11.1).
    pub ss_rsrq: Option<u8>,
}

impl NrMeasuredResultsElement {
    const SFN_BITS: usize = 10;
    const ARFCN_MAX: i64 = 3_279_165;

    fn write(&self, w: &mut UperWriter) -> UperResult<()> {
        let ssb_results = self.ss_rsrp.is_some() || self.ss_rsrq.is_some();
        // Extensible; optionals [nr-CellGlobalID, systemFrameNumber, resultsSSB-Cell,
        // resultsCSI-RS-Cell, resultsSSB-Indexes, resultsCSI-RS-Indexes].
        w.write_sequence_preamble(
            Some(false),
            &[
                false,
                self.system_frame_number.is_some(),
                ssb_results,
                false,
                false,
                false,
            ],
        );
        w.write_constrained(i64::from(self.phys_cell_id), 0, 1007)?;
        // nr-ARFCN-r16 CHOICE { ssb-ARFCN-r16, csi-RS-pointA-r16 }, not extensible.
        w.write_choice_index(0, 2)?;
        w.write_constrained(i64::from(self.ssb_arfcn), 0, Self::ARFCN_MAX)?;
        if let Some(sfn) = self.system_frame_number {
            let bits: Vec<bool> = (0..Self::SFN_BITS)
                .rev()
                .map(|index| (sfn >> index) & 1 == 1)
                .collect();
            w.write_bit_string(&bits, Self::SFN_BITS, Self::SFN_BITS)?;
        }
        if ssb_results {
            // MeasQuantityResults-r16 ::= SEQUENCE { nr-RSRP-r16 OPTIONAL,
            // nr-RSRQ-r16 OPTIONAL } -- not extensible.
            w.write_sequence_preamble(None, &[self.ss_rsrp.is_some(), self.ss_rsrq.is_some()]);
            if let Some(v) = self.ss_rsrp {
                w.write_constrained(i64::from(v), 0, 127)?;
            }
            if let Some(v) = self.ss_rsrq {
                w.write_constrained(i64::from(v), 0, 127)?;
            }
        }
        Ok(())
    }

    fn read(r: &mut UperReader<'_>) -> UperResult<Self> {
        let (additions, opts) = r.read_sequence_preamble(true, 6)?;
        if opts[0] || opts[3] || opts[4] || opts[5] {
            return Err(UperError::Unsupported(
                "NR-MeasuredResultsElement-r16 cell global id, CSI-RS or index results",
            ));
        }
        let phys_cell_id = r.read_constrained(0, 1007)? as u16;
        if r.read_choice_index(2)? != 0 {
            return Err(UperError::Unsupported("nr-ARFCN-r16 csi-RS-pointA-r16"));
        }
        let ssb_arfcn = r.read_constrained(0, Self::ARFCN_MAX)? as u32;
        let system_frame_number = if opts[1] {
            let bits = r.read_bit_string(Self::SFN_BITS, Self::SFN_BITS)?;
            Some(bits.iter().fold(0u16, |acc, &b| (acc << 1) | u16::from(b)))
        } else {
            None
        };
        let (ss_rsrp, ss_rsrq) = if opts[2] {
            let (_, q) = r.read_sequence_preamble(false, 2)?;
            let rsrp = if q[0] {
                Some(r.read_constrained(0, 127)? as u8)
            } else {
                None
            };
            let rsrq = if q[1] {
                Some(r.read_constrained(0, 127)? as u8)
            } else {
                None
            };
            (rsrp, rsrq)
        } else {
            (None, None)
        };
        if additions {
            r.skip_extension_additions()?;
        }
        Ok(Self {
            phys_cell_id,
            ssb_arfcn,
            system_frame_number,
            ss_rsrp,
            ss_rsrq,
        })
    }
}

/// `NR-ECID-SignalMeasurementInformation-r16`: the primary cell (mandatory) and an
/// optional `NR-MeasuredResultsList-r16 ::= SEQUENCE (SIZE(1..32))` (#330).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NrEcidSignalMeasurementInformation {
    /// The serving cell.
    pub primary_cell: NrMeasuredResultsElement,
    /// Neighbours; empty encodes as absent.
    pub measured_results: Vec<NrMeasuredResultsElement>,
}

impl NrEcidSignalMeasurementInformation {
    fn write(&self, w: &mut UperWriter) -> UperResult<()> {
        if self.measured_results.len() > 32 {
            return Err(UperError::InvalidLength(self.measured_results.len()));
        }
        w.write_sequence_preamble(Some(false), &[!self.measured_results.is_empty()]);
        self.primary_cell.write(w)?;
        if !self.measured_results.is_empty() {
            w.write_constrained(self.measured_results.len() as i64, 1, 32)?;
            for element in &self.measured_results {
                element.write(w)?;
            }
        }
        Ok(())
    }

    fn read(r: &mut UperReader<'_>) -> UperResult<Self> {
        let (additions, opts) = r.read_sequence_preamble(true, 1)?;
        let primary_cell = NrMeasuredResultsElement::read(r)?;
        let mut measured_results = Vec::new();
        if opts[0] {
            let count = r.read_constrained(1, 32)? as usize;
            for _ in 0..count {
                measured_results.push(NrMeasuredResultsElement::read(r)?);
            }
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

/// The Rel-16 extension-addition group of an `-r9` body, as an open type (#330):
/// `[[ nr-ECID-..., nr-Multi-RTT-..., nr-DL-AoD-..., nr-DL-TDOA-...
/// (, nr-UL-... for the capability bodies) ]]`. The `-r9` bodies carry three
/// groups (r13, r16, r19); only the second is produced here.
const R9_EXTENSION_GROUPS: usize = 3;
const R16_GROUP: usize = 1;

/// Write `group` (the r16 group's complete encoding) as the extension additions of
/// an `-r9` body whose preamble set the extension bit.
fn write_r16_group(w: &mut UperWriter, group: Vec<u8>) -> UperResult<()> {
    let mut additions: Vec<Option<Vec<u8>>> = vec![None; R9_EXTENSION_GROUPS];
    additions[R16_GROUP] = Some(group);
    w.write_extension_additions(&additions)
}

/// Read an `-r9` body's extension additions and return the r16 group's octets.
/// The r13 group and anything newer are not modelled: present, they are refused,
/// because each names a positioning method this UE does not implement.
fn read_r16_group(r: &mut UperReader<'_>) -> UperResult<Option<Vec<u8>>> {
    let mut additions = r.read_extension_additions()?;
    if additions.first().is_some_and(Option::is_some) {
        return Err(UperError::Unsupported(
            "LPP Rel-13 sensor, TBS, WLAN or Bluetooth positioning",
        ));
    }
    if additions.iter().skip(R16_GROUP + 1).any(Option::is_some) {
        return Err(UperError::Unsupported("LPP Rel-19 positioning"));
    }
    Ok(additions.get_mut(R16_GROUP).and_then(Option::take))
}

/// Within an r16 group: its presence bits (`members` of them) and whether only the
/// first (NR E-CID) is set; any other NR method is refused.
fn read_r16_group_nr_ecid(r: &mut UperReader<'_>, members: usize) -> UperResult<bool> {
    let (_, present) = r.read_sequence_preamble(false, members)?;
    if present.iter().skip(1).any(|p| *p) {
        return Err(UperError::Unsupported(
            "LPP NR positioning other than NR E-CID",
        ));
    }
    Ok(present[0])
}

/// The root of a `RequestCapabilities-r9-IEs`, read through its last root member.
/// Returns `(additions, ecid_requested)`.
///
/// TS 37.355 §5.1.3 answers each requested method the UE supports, so the
/// requests for the methods it does not (A-GNSS, OTDOA, an EPDU) are read past
/// rather than refused (#377 review): `A-GNSS-RequestCapabilities` is three
/// BOOLEANs, `OTDOA-RequestCapabilities` and `ECID-RequestCapabilities` are empty,
/// all extensible, and `epdu-RequestCapabilities` is an `EPDU-Sequence`.
fn read_capability_request_root(r: &mut UperReader<'_>) -> UperResult<(bool, bool)> {
    let (additions, opts) = r.read_sequence_preamble(true, 5)?;
    let skip_extensible = |r: &mut UperReader<'_>, booleans: usize| -> UperResult<()> {
        let (inner_additions, _) = r.read_sequence_preamble(true, 0)?;
        for _ in 0..booleans {
            r.read_bit()?;
        }
        if inner_additions {
            r.skip_extension_additions()?;
        }
        Ok(())
    };
    if opts[0] {
        skip_extensible(r, 0)?; // CommonIEsRequestCapabilities
    }
    if opts[1] {
        skip_extensible(r, 3)?; // A-GNSS-RequestCapabilities
    }
    if opts[2] {
        skip_extensible(r, 0)?; // OTDOA-RequestCapabilities
    }
    if opts[3] {
        skip_extensible(r, 0)?; // ECID-RequestCapabilities
    }
    if opts[4] {
        skip_epdu_sequence(r)?;
    }
    Ok((additions, opts[3]))
}

/// An X.691 §11.9 unconstrained length determinant (single octet or two
/// octets; fragmentation is refused).
fn read_unconstrained_length(r: &mut UperReader<'_>) -> UperResult<usize> {
    if !r.read_bit()? {
        return Ok(r.read_bits(7)? as usize);
    }
    if !r.read_bit()? {
        return Ok(r.read_bits(14)? as usize);
    }
    Err(UperError::Unsupported("a fragmented EPDU-Body"))
}

/// Reads past an `EPDU-Sequence` (TS 37.355 §6.4.1): 1..16 `EPDU`s, each an
/// extensible `EPDU-Identifier` (an id, an optional name) and an OCTET STRING body.
fn skip_epdu_sequence(r: &mut UperReader<'_>) -> UperResult<()> {
    let count = r.read_constrained(1, 16)?;
    for _ in 0..count {
        let (ext, opts) = r.read_sequence_preamble(true, 1)?;
        r.read_constrained(1, 256)?;
        if opts[0] {
            let len = r.read_constrained(1, 32)?;
            for _ in 0..len {
                r.read_bits(7)?;
            }
        }
        if ext {
            r.skip_extension_additions()?;
        }
        let len = read_unconstrained_length(r)?;
        for _ in 0..len {
            r.read_bits(8)?;
        }
    }
    Ok(())
}

/// A `RequestCapabilities`' extension additions: whether its r16 group asks for
/// the NR E-CID capabilities. TS 37.355 §5.1.3 has the target answer for each
/// requested method it supports, so a request that also names methods this UE
/// lacks (Multi-RTT, DL-TDOA, the r13 sensor/TBS/WLAN/Bluetooth group, ...) is
/// answered for E-CID and NR E-CID alone rather than refused (#377).
fn read_capability_request_nr_ecid(r: &mut UperReader<'_>) -> UperResult<bool> {
    let mut additions = r.read_extension_additions()?;
    let Some(group) = additions.get_mut(R16_GROUP).and_then(Option::take) else {
        return Ok(false);
    };
    let mut g = UperReader::new(&group);
    let (_, present) = g.read_sequence_preamble(false, R16_CAPABILITY_MEMBERS)?;
    // NR-ECID-RequestCapabilities-r16 is the group's first member, so nothing
    // after it needs reading.
    Ok(present[0])
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

/// The LPP message bodies this UE handles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LppBody {
    /// `RequestCapabilities` (§5.1.3).
    RequestCapabilities {
        /// Whether the server asked about E-UTRA E-CID.
        ecid_requested: bool,
        /// Whether it asked about NR E-CID (Rel-16 group, #330).
        nr_ecid_requested: bool,
    },
    /// `ProvideCapabilities` (§5.1.3).
    ProvideCapabilities {
        /// The E-UTRA E-CID measurements this UE supports, when it reports them:
        /// a reply carries only the capabilities the request asked for (TS 37.355
        /// §5.1.1, §5.1.3; #377).
        ecid_supported: Option<EcidMeasurementBits>,
        /// The NR E-CID measurements it supports, when it reports them (#330).
        nr_ecid_supported: Option<NrEcidMeasurementBits>,
    },
    /// `RequestLocationInformation` (§5.3.3). At least one method is present.
    RequestLocationInformation {
        /// The E-UTRA E-CID measurements the server asked for, if it asked for E-CID.
        requested: Option<EcidMeasurementBits>,
        /// The NR E-CID measurements it asked for, if it asked for NR E-CID (#330).
        nr_requested: Option<NrEcidMeasurementBits>,
    },
    /// `ProvideLocationInformation` (§5.3.3). At least one method is present.
    ProvideLocationInformation {
        /// The E-UTRA E-CID measurement report.
        measurements: Option<EcidSignalMeasurementInformation>,
        /// The NR E-CID measurement report (#330).
        nr_measurements: Option<NrEcidSignalMeasurementInformation>,
    },
}

/// A complete encoding of an r16 group whose only present member (`members`
/// presence bits in all) is the first, NR E-CID, written by `body`.
fn r16_group_with_nr_ecid(
    members: usize,
    body: impl FnOnce(&mut UperWriter) -> UperResult<()>,
) -> UperResult<Vec<u8>> {
    let mut g = UperWriter::new();
    let mut present = vec![false; members];
    present[0] = true;
    g.write_sequence_preamble(None, &present);
    body(&mut g)?;
    Ok(g.into_bytes())
}

/// The r16 group of a capability body has five members (NR E-CID, Multi-RTT,
/// DL-AoD, DL-TDOA, UL); that of a location-information body four.
const R16_CAPABILITY_MEMBERS: usize = 5;
const R16_LOCATION_MEMBERS: usize = 4;

impl LppBody {
    fn write(&self, w: &mut UperWriter) -> UperResult<()> {
        // LPP-MessageBody CHOICE: c1 (index 0 of 2), then the c1 index.
        w.write_choice_index(0, 2)?;
        match self {
            Self::RequestCapabilities {
                ecid_requested,
                nr_ecid_requested,
            } => {
                w.write_choice_index(c1::REQUEST_CAPABILITIES, c1::ALTERNATIVES)?;
                write_critical_extensions(w)?;
                w.write_sequence_preamble(
                    Some(*nr_ecid_requested),
                    &[false, false, false, *ecid_requested, false],
                );
                if *ecid_requested {
                    // ECID-RequestCapabilities is an empty extensible SEQUENCE.
                    w.write_sequence_preamble(Some(false), &[]);
                }
                if *nr_ecid_requested {
                    // NR-ECID-RequestCapabilities-r16 is an empty extensible SEQUENCE.
                    let group = r16_group_with_nr_ecid(R16_CAPABILITY_MEMBERS, |g| {
                        g.write_sequence_preamble(Some(false), &[]);
                        Ok(())
                    })?;
                    write_r16_group(w, group)?;
                }
                Ok(())
            }
            Self::ProvideCapabilities {
                ecid_supported,
                nr_ecid_supported,
            } => {
                w.write_choice_index(c1::PROVIDE_CAPABILITIES, c1::ALTERNATIVES)?;
                write_critical_extensions(w)?;
                w.write_sequence_preamble(
                    Some(nr_ecid_supported.is_some()),
                    &[false, false, false, ecid_supported.is_some(), false],
                );
                if let Some(ecid) = ecid_supported {
                    ecid.write(w)?;
                }
                if let Some(nr) = nr_ecid_supported {
                    // NR-ECID-ProvideCapabilities-r16: extensible, optionals
                    // [periodicalReporting, triggeredReporting], then the bit map.
                    let group = r16_group_with_nr_ecid(R16_CAPABILITY_MEMBERS, |g| {
                        g.write_sequence_preamble(Some(false), &[false, false]);
                        g.write_bit_string(&nr.as_bits(), 1, 8)
                    })?;
                    write_r16_group(w, group)?;
                }
                Ok(())
            }
            Self::RequestLocationInformation {
                requested,
                nr_requested,
            } => {
                w.write_choice_index(c1::REQUEST_LOCATION_INFORMATION, c1::ALTERNATIVES)?;
                write_critical_extensions(w)?;
                w.write_sequence_preamble(
                    Some(nr_requested.is_some()),
                    &[false, false, false, requested.is_some(), false],
                );
                if let Some(bits) = requested {
                    bits.write(w)?;
                }
                if let Some(nr) = nr_requested {
                    // NR-ECID-RequestLocationInformation-r16: extensible, then the map.
                    let group = r16_group_with_nr_ecid(R16_LOCATION_MEMBERS, |g| {
                        g.write_sequence_preamble(Some(false), &[]);
                        g.write_bit_string(&nr.as_bits(), 1, 8)
                    })?;
                    write_r16_group(w, group)?;
                }
                Ok(())
            }
            Self::ProvideLocationInformation {
                measurements,
                nr_measurements,
            } => {
                if measurements.is_none() && nr_measurements.is_none() {
                    return Err(UperError::Unsupported(
                        "ProvideLocationInformation with no measurements",
                    ));
                }
                w.write_choice_index(c1::PROVIDE_LOCATION_INFORMATION, c1::ALTERNATIVES)?;
                write_critical_extensions(w)?;
                // Root presence bits [commonIEs, a-gnss, otdoa, ecid, epdu]; the
                // extension bit announces the Rel-16 group carrying NR E-CID.
                // Sidelink ranging is SLPP's (TS 38.355, `nas::slpp`), so no EPDU
                // is ever written (nextgcore #557).
                w.write_sequence_preamble(
                    Some(nr_measurements.is_some()),
                    &[false, false, false, measurements.is_some(), false],
                );
                if let Some(m) = measurements {
                    // ECID-ProvideLocationInformation: [signalMeasurementInformation,
                    // ecid-Error], and only the measurements are produced here.
                    w.write_sequence_preamble(Some(false), &[true, false]);
                    m.write(w)?;
                }
                if let Some(nr) = nr_measurements {
                    // NR-ECID-ProvideLocationInformation-r16: extensible, optionals
                    // [nr-ECID-SignalMeasurementInformation, nr-ECID-Error].
                    let group = r16_group_with_nr_ecid(R16_LOCATION_MEMBERS, |g| {
                        g.write_sequence_preamble(Some(false), &[true, false]);
                        nr.write(g)
                    })?;
                    write_r16_group(w, group)?;
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
                let (additions, ecid) = read_capability_request_root(r)?;
                let nr_ecid_requested = additions && read_capability_request_nr_ecid(r)?;
                Ok(Self::RequestCapabilities {
                    ecid_requested: ecid,
                    nr_ecid_requested,
                })
            }
            c1::PROVIDE_CAPABILITIES => {
                read_critical_extensions(r)?;
                let (additions, ecid) = read_r9_body_preamble(r, CommonIes::EmptySequence)?;
                let ecid_supported = if ecid {
                    Some(EcidMeasurementBits::read(r)?)
                } else {
                    None
                };
                let mut nr_ecid_supported = None;
                if additions {
                    if let Some(group) = read_r16_group(r)? {
                        let mut g = UperReader::new(&group);
                        if read_r16_group_nr_ecid(&mut g, R16_CAPABILITY_MEMBERS)? {
                            let (inner_additions, opts) = g.read_sequence_preamble(true, 2)?;
                            for present in opts {
                                if present {
                                    // ENUMERATED { supported }, not extensible: 0 bits.
                                    let _ = g.read_constrained(0, 0)?;
                                }
                            }
                            let bits = g.read_bit_string(1, 8)?;
                            if inner_additions {
                                g.skip_extension_additions()?;
                            }
                            nr_ecid_supported = Some(NrEcidMeasurementBits::from_bits(&bits));
                        }
                    }
                }
                Ok(Self::ProvideCapabilities {
                    ecid_supported,
                    nr_ecid_supported,
                })
            }
            c1::REQUEST_LOCATION_INFORMATION => {
                read_critical_extensions(r)?;
                let (additions, ecid) = read_r9_body_preamble(r, CommonIes::Unmodelled)?;
                let requested = if ecid {
                    Some(EcidMeasurementBits::read(r)?)
                } else {
                    None
                };
                let mut nr_requested = None;
                if additions {
                    if let Some(group) = read_r16_group(r)? {
                        let mut g = UperReader::new(&group);
                        if read_r16_group_nr_ecid(&mut g, R16_LOCATION_MEMBERS)? {
                            let (inner_additions, _) = g.read_sequence_preamble(true, 0)?;
                            let bits = g.read_bit_string(1, 8)?;
                            if inner_additions {
                                g.skip_extension_additions()?;
                            }
                            nr_requested = Some(NrEcidMeasurementBits::from_bits(&bits));
                        }
                    }
                }
                if requested.is_none() && nr_requested.is_none() {
                    // A request naming no method this UE answers. Reported rather
                    // than answered with a default, which would report measurements
                    // the server did not ask for.
                    return Err(UperError::Unsupported(
                        "RequestLocationInformation with no E-CID or NR E-CID body",
                    ));
                }
                Ok(Self::RequestLocationInformation {
                    requested,
                    nr_requested,
                })
            }
            c1::PROVIDE_LOCATION_INFORMATION => {
                read_critical_extensions(r)?;
                let (additions, ecid) = read_r9_body_preamble(r, CommonIes::Unmodelled)?;
                let measurements = if ecid {
                    let (inner_additions, opts) = r.read_sequence_preamble(true, 2)?;
                    let m = if opts[0] {
                        EcidSignalMeasurementInformation::read(r)?
                    } else {
                        return Err(UperError::Unsupported(
                            "ProvideLocationInformation without signal measurements",
                        ));
                    };
                    if inner_additions {
                        r.skip_extension_additions()?;
                    }
                    Some(m)
                } else {
                    None
                };
                let mut nr_measurements = None;
                if additions {
                    if let Some(group) = read_r16_group(r)? {
                        let mut g = UperReader::new(&group);
                        if read_r16_group_nr_ecid(&mut g, R16_LOCATION_MEMBERS)? {
                            let (inner_additions, opts) = g.read_sequence_preamble(true, 2)?;
                            if opts[1] || !opts[0] {
                                return Err(UperError::Unsupported(
                                    "NR E-CID ProvideLocationInformation without measurements",
                                ));
                            }
                            let m = NrEcidSignalMeasurementInformation::read(&mut g)?;
                            if inner_additions {
                                g.skip_extension_additions()?;
                            }
                            nr_measurements = Some(m);
                        }
                    }
                }
                if measurements.is_none() && nr_measurements.is_none() {
                    return Err(UperError::Unsupported(
                        "ProvideLocationInformation with no E-CID or NR E-CID body",
                    ));
                }
                Ok(Self::ProvideLocationInformation {
                    measurements,
                    nr_measurements,
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
