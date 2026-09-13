//! Paging Procedure (PCCH)
//!
//! Implements the RRC Paging message of 3GPP TS 38.331 §5.3.2 / §6.2.2. The
//! network transmits `PCCH-Message` on the PCCH logical channel carrying a
//! `PagingRecordList`; a UE in RRC_IDLE or RRC_INACTIVE compares each record's
//! UE identity with its own and, on a match, initiates the NAS service request
//! procedure (TS 24.501 §5.6.1.1) or RRC resume.
//!
//! Two identities may be paged (`PagingUE-Identity`):
//!
//! - `ng-5G-S-TMSI` (48 bits) — a UE in RRC_IDLE, paged by the 5G-S-TMSI the
//!   AMF sent in the NGAP Paging (TS 38.413 §9.3.3.20).
//! - `fullI-RNTI` (40 bits) — a UE in RRC_INACTIVE, paged by the NG-RAN.
//!
//! The 5G-S-TMSI octets are the canonical TS 23.003 §2.10.1 form: AMF Set ID
//! (10 bits) and AMF Pointer (6 bits) packed into two octets, followed by the
//! 32-bit 5G-TMSI.

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use bitvec::prelude::*;
use thiserror::Error;

/// Maximum number of paging records in one PCCH Paging message
/// (TS 38.331 `maxNrofPageRec`).
pub const MAX_PAGE_RECORDS: usize = 32;

/// Length in octets of a 5G-S-TMSI (48 bits).
pub const FIVE_G_S_TMSI_LEN: usize = 6;

/// Length in octets of a full I-RNTI (40 bits).
pub const I_RNTI_LEN: usize = 5;

/// Errors that can occur during Paging procedures
#[derive(Debug, Error)]
pub enum PagingError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] RrcCodecError),

    /// The decoded PCCH message is not a Paging message
    #[error("PCCH message is not a Paging message")]
    NotAPagingMessage,

    /// A Paging message must carry at least one record when built
    /// (`PagingRecordList` is `SIZE (1..maxNrofPageRec)`).
    #[error("no paging records given")]
    NoRecords,

    /// More records than `maxNrofPageRec` were given
    #[error("too many paging records: {count} (max {MAX_PAGE_RECORDS})")]
    TooManyRecords {
        /// Number of records the caller supplied
        count: usize,
    },

    /// A decoded identity had a length the ASN.1 constraint forbids
    #[error("invalid {field} length: {len} bits")]
    InvalidIdentityLength {
        /// Name of the identity field
        field: &'static str,
        /// Length actually decoded, in bits
        len: usize,
    },
}

/// UE identity carried in a `PagingRecord` (TS 38.331 `PagingUE-Identity`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedUeIdentity {
    /// 5G-S-TMSI (48 bits): AMF Set ID (10 bits) + AMF Pointer (6 bits) packed
    /// into two octets, then the 32-bit 5G-TMSI (TS 23.003 §2.10.1).
    FiveGSTmsi([u8; FIVE_G_S_TMSI_LEN]),
    /// Full I-RNTI (40 bits), used to page a UE in RRC_INACTIVE.
    FullIRnti([u8; I_RNTI_LEN]),
}

/// One paging record (TS 38.331 `PagingRecord`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagingRecordParams {
    /// The paged UE identity
    pub ue_identity: PagedUeIdentity,
    /// `accessType` = non3GPP: the paging is for a PDU session over non-3GPP
    /// access (TS 38.331 `PagingRecord`). Absent when false.
    pub non_3gpp_access: bool,
}

impl PagingRecordParams {
    /// A record paging a UE in RRC_IDLE by its 5G-S-TMSI over 3GPP access.
    pub fn five_g_s_tmsi(s_tmsi: [u8; FIVE_G_S_TMSI_LEN]) -> Self {
        Self {
            ue_identity: PagedUeIdentity::FiveGSTmsi(s_tmsi),
            non_3gpp_access: false,
        }
    }
}

/// Encodes a `PCCH-Message` carrying a Paging message with `records`
/// (TS 38.331 §6.2.2).
///
/// `records` must hold between 1 and [`MAX_PAGE_RECORDS`] entries: the ASN.1
/// `PagingRecordList` is `SIZE (1..maxNrofPageRec)`, so an empty list has no
/// encoding. A caller with nothing to page must not send a Paging message at
/// all rather than send an empty one.
pub fn encode_paging(records: &[PagingRecordParams]) -> Result<Vec<u8>, PagingError> {
    if records.is_empty() {
        return Err(PagingError::NoRecords);
    }
    if records.len() > MAX_PAGE_RECORDS {
        return Err(PagingError::TooManyRecords {
            count: records.len(),
        });
    }

    let asn_records = records
        .iter()
        .map(|record| PagingRecord {
            ue_identity: match record.ue_identity {
                PagedUeIdentity::FiveGSTmsi(s_tmsi) => {
                    PagingUE_Identity::Ng_5G_S_TMSI(NG_5G_S_TMSI(BitVec::from_slice(&s_tmsi)))
                }
                PagedUeIdentity::FullIRnti(i_rnti) => {
                    PagingUE_Identity::FullI_RNTI(I_RNTI_Value(BitVec::from_slice(&i_rnti)))
                }
            },
            access_type: record
                .non_3gpp_access
                .then_some(PagingRecordAccessType(PagingRecordAccessType::NON3_GPP)),
        })
        .collect();

    let message = PCCH_Message {
        message: PCCH_MessageType::C1(PCCH_MessageType_c1::Paging(Paging {
            paging_record_list: Some(PagingRecordList(asn_records)),
            late_non_critical_extension: None,
            non_critical_extension: None,
        })),
    };

    Ok(encode_rrc(&message)?)
}

/// Decodes a `PCCH-Message` and returns its paging records.
///
/// An empty vector is returned for a Paging message whose optional
/// `pagingRecordList` is absent — that is a legal message which pages nobody,
/// not a decode failure.
pub fn decode_paging(bytes: &[u8]) -> Result<Vec<PagingRecordParams>, PagingError> {
    let message: PCCH_Message = decode_rrc(bytes)?;

    let paging = match message.message {
        PCCH_MessageType::C1(PCCH_MessageType_c1::Paging(paging)) => paging,
        _ => return Err(PagingError::NotAPagingMessage),
    };

    let Some(PagingRecordList(records)) = paging.paging_record_list else {
        return Ok(Vec::new());
    };

    records
        .into_iter()
        .map(|record| {
            let ue_identity = match record.ue_identity {
                PagingUE_Identity::Ng_5G_S_TMSI(NG_5G_S_TMSI(bits)) => {
                    PagedUeIdentity::FiveGSTmsi(identity_octets(&bits, "ng-5G-S-TMSI")?)
                }
                PagingUE_Identity::FullI_RNTI(I_RNTI_Value(bits)) => {
                    PagedUeIdentity::FullIRnti(identity_octets(&bits, "fullI-RNTI")?)
                }
            };
            Ok(PagingRecordParams {
                ue_identity,
                non_3gpp_access: record.access_type.is_some(),
            })
        })
        .collect()
}

/// Converts a decoded fixed-length BIT STRING into its octets.
///
/// The bit count is checked against the target width rather than trusting the
/// decoder: a BIT STRING whose size constraint was violated would otherwise
/// silently yield a zero-padded identity that could match the wrong UE.
fn identity_octets<const N: usize>(
    bits: &BitVec<u8, Msb0>,
    field: &'static str,
) -> Result<[u8; N], PagingError> {
    if bits.len() != N * 8 {
        return Err(PagingError::InvalidIdentityLength {
            field,
            len: bits.len(),
        });
    }
    let mut octets = [0u8; N];
    for (index, bit) in bits.iter().by_vals().enumerate() {
        if bit {
            octets[index / 8] |= 0x80 >> (index % 8);
        }
    }
    Ok(octets)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_TMSI: [u8; 6] = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];

    #[test]
    fn a_single_5g_s_tmsi_record_round_trips() {
        let pdu = encode_paging(&[PagingRecordParams::five_g_s_tmsi(S_TMSI)]).unwrap();
        let records = decode_paging(&pdu).unwrap();

        assert_eq!(
            records,
            vec![PagingRecordParams {
                ue_identity: PagedUeIdentity::FiveGSTmsi(S_TMSI),
                non_3gpp_access: false,
            }]
        );
    }

    /// A byte-level assertion, not just a round trip: an encoder and decoder
    /// that agree on a wrong bit layout round-trip perfectly, so only the wire
    /// bytes can show that the PCCH choice indices, the extension markers and
    /// the 48-bit identity land where TS 38.331 puts them.
    ///
    /// UPER layout for one 5G-S-TMSI record (62 bits + 2 pad bits):
    ///
    /// ```text
    /// bit 0     PCCH-MessageType CHOICE index = c1
    /// bit 1     c1 CHOICE index = paging
    /// bit 2     pagingRecordList present
    /// bit 3-4   lateNonCriticalExtension, nonCriticalExtension absent
    /// bit 5-9   PagingRecordList length determinant (1 record - 1)
    /// bit 10    PagingRecord extension marker
    /// bit 11    accessType absent
    /// bit 12    PagingUE-Identity extension marker
    /// bit 13    PagingUE-Identity CHOICE index = ng-5G-S-TMSI
    /// bit 14-61 5G-S-TMSI (48 bits)
    /// ```
    ///
    /// Note there is NO extension bit for `Paging` itself: TS 38.331 declares
    /// `Paging` without an extension marker (`PagingRecord` and
    /// `PagingUE-Identity` have one, hence bits 10 and 12). The generated Rust
    /// type nonetheless carries `extensible = true`, so the record boundary can
    /// only be confirmed from the wire bytes.
    #[test]
    fn the_encoded_pcch_paging_matches_the_ts_38_331_bit_layout() {
        let pdu = encode_paging(&[PagingRecordParams::five_g_s_tmsi(S_TMSI)]).unwrap();

        // 0x20 = 0010_0000: c1, paging, pagingRecordList present. The identity
        // starts at bit 14, so every S-TMSI octet is split across two encoded
        // octets (shifted left by two bits).
        assert_eq!(pdu, vec![0x20, 0x00, 0x48, 0xD1, 0x59, 0xE2, 0x6A, 0xF0]);
    }

    #[test]
    fn a_full_i_rnti_record_round_trips() {
        let i_rnti = [0x01, 0x23, 0x45, 0x67, 0x89];
        let pdu = encode_paging(&[PagingRecordParams {
            ue_identity: PagedUeIdentity::FullIRnti(i_rnti),
            non_3gpp_access: false,
        }])
        .unwrap();

        assert_eq!(
            decode_paging(&pdu).unwrap(),
            vec![PagingRecordParams {
                ue_identity: PagedUeIdentity::FullIRnti(i_rnti),
                non_3gpp_access: false,
            }]
        );
    }

    #[test]
    fn the_non_3gpp_access_type_survives_a_round_trip() {
        let pdu = encode_paging(&[PagingRecordParams {
            ue_identity: PagedUeIdentity::FiveGSTmsi(S_TMSI),
            non_3gpp_access: true,
        }])
        .unwrap();

        assert!(decode_paging(&pdu).unwrap()[0].non_3gpp_access);
    }

    #[test]
    fn every_record_of_a_multi_ue_paging_message_is_decoded_in_order() {
        let records: Vec<PagingRecordParams> = (0..MAX_PAGE_RECORDS)
            .map(|index| PagingRecordParams::five_g_s_tmsi([0, 0, 0, 0, 0, index as u8]))
            .collect();

        let pdu = encode_paging(&records).unwrap();
        assert_eq!(decode_paging(&pdu).unwrap(), records);
    }

    #[test]
    fn an_empty_record_list_is_refused_rather_than_encoded() {
        assert!(matches!(encode_paging(&[]), Err(PagingError::NoRecords)));
    }

    #[test]
    fn more_records_than_max_nrof_page_rec_are_refused() {
        let records = vec![PagingRecordParams::five_g_s_tmsi(S_TMSI); MAX_PAGE_RECORDS + 1];
        assert!(matches!(
            encode_paging(&records),
            Err(PagingError::TooManyRecords { count }) if count == MAX_PAGE_RECORDS + 1
        ));
    }

    #[test]
    fn a_paging_message_with_no_record_list_decodes_to_no_records() {
        let message = PCCH_Message {
            message: PCCH_MessageType::C1(PCCH_MessageType_c1::Paging(Paging {
                paging_record_list: None,
                late_non_critical_extension: None,
                non_critical_extension: None,
            })),
        };
        let pdu = encode_rrc(&message).unwrap();

        assert!(decode_paging(&pdu).unwrap().is_empty());
    }

    #[test]
    fn a_non_paging_pcch_message_is_rejected() {
        let message = PCCH_Message {
            message: PCCH_MessageType::C1(PCCH_MessageType_c1::Spare1(PCCH_MessageType_c1_spare1)),
        };
        let pdu = encode_rrc(&message).unwrap();

        assert!(matches!(
            decode_paging(&pdu),
            Err(PagingError::NotAPagingMessage)
        ));
    }

    #[test]
    fn garbage_bytes_are_a_decode_error_not_a_panic() {
        assert!(decode_paging(&[0xFF, 0xFF]).is_err());
    }
}
