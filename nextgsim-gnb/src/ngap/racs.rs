//! RACS on the NG-RAN side (TS 23.501 §5.4.4.1a, TS 38.413 §8.14.3).
//!
//! A UE Radio Capability ID stands for a full UE radio capability set. The gNB keeps
//! the sets it has learned, keyed by ID, so a UE whose AMF sends only the ID (Initial
//! Context Setup, Handover Request, Path Switch Request Acknowledge) costs no radio
//! capability transfer at all. On a miss the gNB runs UE Radio Capability ID Mapping
//! toward the AMF.
//!
//! # Key
//!
//! `(serving PLMN, ID)`. TS 23.501 §5.4.4.1a requires the RAN to "cache PLMN assigned
//! UE Radio Capability IDs per PLMN ID" for MOCN; keying manufacturer-assigned IDs by
//! PLMN as well costs at most a duplicate entry and needs no inspection of the ID.
//!
//! # Bound and eviction
//!
//! At most [`RACS_CACHE_CAPACITY`] entries, evicting the least recently used. The ID
//! is treated as opaque bytes: TS 38.413 §9.3.1.142 carries it as an OCTET STRING
//! "defined in TS 23.003" without fixing how TS 23.003 §29.2's hexadecimal digits
//! are packed, so the gNB does not read the Type Field or Version ID out of it. The
//! TS 23.501 §5.4.4.1a preference to purge PLMN-assigned IDs of an old Version ID
//! first is therefore not applied; plain LRU is the documented policy.

use std::collections::{HashMap, VecDeque};

/// Maximum number of cached ID → capability entries. An implementation choice: TS
/// 23.501 leaves the cache size to the RAN.
pub const RACS_CACHE_CAPACITY: usize = 256;

/// How long the gNB waits for a UE RADIO CAPABILITY ID MAPPING RESPONSE before it
/// gives up on the mapping. TS 38.413 §8.14.3 defines no timer (and no unsuccessful
/// outcome); TS 23.502 §4.9.1.3.2 step 9 only says the capability must be "promptly
/// received". An implementation choice.
pub const RACS_MAPPING_TIMEOUT_SECS: u64 = 2;

/// Seconds a UE Radio Capability Check (TS 38.413 §8.14.2) waits for the UE's radio
/// capability before answering with what the gNB holds. §8.14.2 defines no timer and
/// no unsuccessful outcome, so the AMF is always answered; this bounds the wait to a
/// mapping timeout plus a UE capability enquiry. An implementation choice.
pub const RADIO_CAPABILITY_CHECK_TIMEOUT_SECS: u64 = 3 * RACS_MAPPING_TIMEOUT_SECS;

/// Whether `capability` (an encoded TS 38.331 `UERadioAccessCapabilityInformation`)
/// is compatible with a network configured for IMS voice over NR (TS 38.413
/// §8.14.2): the gNB must support it (`network_supports`) and some NR container
/// must declare `voiceOverNR` (TS 38.331 §6.3.3). An undecodable capability is
/// not compatible.
pub fn ims_voice_compatible(network_supports: bool, capability: &[u8]) -> bool {
    use nextgsim_rrc::procedures::ue_capability::{
        decode_ue_radio_access_capability_information, parse_nr_capability_voice_over_nr, RatType,
    };
    network_supports
        && decode_ue_radio_access_capability_information(capability).is_ok_and(|containers| {
            containers.iter().any(|c| {
                matches!(c.rat_type, RatType::Nr)
                    && parse_nr_capability_voice_over_nr(&c.container).unwrap_or(false)
            })
        })
}

/// Cache key: serving PLMN (encoded, 3 octets) and the opaque UE Radio Capability ID.
pub type RacsKey = ([u8; 3], Vec<u8>);

/// A bounded, least-recently-used cache of UE radio capability sets by ID.
#[derive(Debug)]
pub struct RacsCache {
    capacity: usize,
    entries: HashMap<RacsKey, Vec<u8>>,
    /// Recency order, least recent at the front.
    order: VecDeque<RacsKey>,
}

impl Default for RacsCache {
    fn default() -> Self {
        Self::with_capacity(RACS_CACHE_CAPACITY)
    }
}

impl RacsCache {
    /// A cache holding at most `capacity` entries (at least one).
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// The capability set for `key`, marking it most recently used.
    pub fn get(&mut self, key: &RacsKey) -> Option<Vec<u8>> {
        let cap = self.entries.get(key)?.clone();
        self.touch(key);
        Some(cap)
    }

    /// Store `capability` for `key`, evicting the least recently used entry if full.
    /// Returns the evicted key, if any.
    pub fn insert(&mut self, key: RacsKey, capability: Vec<u8>) -> Option<RacsKey> {
        if self.entries.insert(key.clone(), capability).is_some() {
            self.touch(&key);
            return None;
        }
        self.order.push_back(key);
        if self.entries.len() > self.capacity {
            let evicted = self.order.pop_front()?;
            self.entries.remove(&evicted);
            return Some(evicted);
        }
        None
    }

    /// Number of entries held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn touch(&mut self, key: &RacsKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            if let Some(k) = self.order.remove(pos) {
                self.order.push_back(k);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: u8) -> RacsKey {
        ([0x09, 0xF1, 0x07], vec![id])
    }

    #[test]
    fn a_full_cache_evicts_the_least_recently_used_entry() {
        let mut cache = RacsCache::with_capacity(2);
        assert_eq!(cache.insert(key(1), vec![1]), None);
        assert_eq!(cache.insert(key(2), vec![2]), None);
        // Touch 1, so 2 is now the least recent.
        assert_eq!(cache.get(&key(1)), Some(vec![1]));
        assert_eq!(cache.insert(key(3), vec![3]), Some(key(2)));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&key(2)), None);
        assert_eq!(cache.get(&key(1)), Some(vec![1]));
        assert_eq!(cache.get(&key(3)), Some(vec![3]));
    }

    #[test]
    fn the_same_id_under_another_plmn_is_another_entry() {
        let mut cache = RacsCache::with_capacity(4);
        cache.insert(([0x09, 0xF1, 0x07], vec![7]), vec![1]);
        assert_eq!(cache.get(&([0x00, 0xF1, 0x10], vec![7])), None);
    }

    #[test]
    fn re_inserting_a_key_replaces_without_evicting() {
        let mut cache = RacsCache::with_capacity(1);
        cache.insert(key(1), vec![1]);
        assert_eq!(cache.insert(key(1), vec![9]), None);
        assert_eq!(cache.get(&key(1)), Some(vec![9]));
    }

    #[test]
    fn the_cache_never_exceeds_its_capacity() {
        let mut cache = RacsCache::with_capacity(8);
        for i in 0..=255u8 {
            cache.insert(key(i), vec![i]);
            assert!(cache.len() <= 8);
        }
        assert_eq!(cache.len(), 8);
    }
}
