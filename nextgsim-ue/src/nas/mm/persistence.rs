//! Non-volatile storage of the 5GMM parameters TS 24.501 Annex C.1 requires
//! the ME to keep across power-off.
//!
//! Annex C.1 names the 5G-GUTI, the last visited registered TAI, the 5GS
//! update status and the 5G NAS security context. Holding them in process
//! memory only means every restart re-registers with a SUCI and runs a fresh
//! primary authentication, so GUTI-first identification (§5.5.1.2.2), native
//! security-context re-use (§4.4.2.1.3) and NAS COUNT continuity are all
//! unobservable — exactly the behaviours a reconnect or load test cares about.
//!
//! # This is off by default, and deliberately
//!
//! The snapshot contains K_AMF and the NAS keys derived from it. Writing those
//! to disk is an operator decision, not a default: without
//! [`UeConfig::state_file`] set, nothing is ever written or read and the UE
//! behaves byte-for-byte as before. When it *is* set, the file is created
//! `0600` on Unix.
//!
//! # Why the snapshot is defined here rather than derived on the NAS types
//!
//! `NasSecurityContext` and `UeKeys` deliberately do not implement
//! `Serialize`: a `Serialize` on a type holding K_AMF is one careless
//! `serde_json::to_string` in a log statement away from printing the key. The
//! snapshot is an explicit, separate shape populated through the public
//! accessors, so every field that leaves the security context does so at a
//! site that names it.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use nextgsim_nas::messages::mm::{Ie5gsMobileIdentity, MobileIdentityType};
use nextgsim_nas::security::{
    CipheringAlgorithm, IntegrityAlgorithm, NasCount, NasSecurityContext, SecurityContextType,
};

use super::UpdateStatus;

/// Errors from reading or writing the stored 5GMM state.
#[derive(Debug)]
pub enum StateFileError {
    /// The file could not be read or written.
    Io(std::io::Error),
    /// The file exists but is not a valid snapshot.
    Malformed(String),
}

impl std::fmt::Display for StateFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "state file I/O error: {e}"),
            Self::Malformed(e) => write!(f, "state file is malformed: {e}"),
        }
    }
}

impl std::error::Error for StateFileError {}

/// The 5G NAS security context as stored, per TS 24.501 §4.4.2.1.3.
///
/// The `recent_sequence_numbers` replay window is deliberately NOT stored: it
/// is an anti-replay cache, and restoring an empty one is the conservative
/// choice — a restored context accepts a downlink COUNT it would previously
/// have rejected as a replay, but the stored `downlink_count` still floors
/// what is acceptable, so a replay below the floor is refused regardless.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityContextSnapshot {
    /// NAS key set identifier (ngKSI, 3 bits; 7 = no key available)
    pub ng_ksi: u8,
    /// `true` when the context is native, `false` when mapped from EPS
    pub native: bool,
    /// K_AMF (256-bit)
    pub kamf: Option<[u8; 32]>,
    /// K_AUSF (256-bit)
    pub kausf: Option<[u8; 32]>,
    /// K_SEAF (256-bit)
    pub kseaf: Option<[u8; 32]>,
    /// K_NASint (128-bit)
    pub knas_int: Option<[u8; 16]>,
    /// K_NASenc (128-bit)
    pub knas_enc: Option<[u8; 16]>,
    /// Uplink NAS COUNT as (overflow, sqn)
    pub uplink_count: (u16, u8),
    /// Downlink NAS COUNT as (overflow, sqn)
    pub downlink_count: (u16, u8),
    /// Selected integrity algorithm (NIA number, 0-3)
    pub integrity_algorithm: u8,
    /// Selected ciphering algorithm (NEA number, 0-3)
    pub ciphering_algorithm: u8,
}

impl SecurityContextSnapshot {
    /// Capture an ACTIVE security context. Returns `None` for a context that
    /// is not active or has no K_AMF: there is nothing worth restoring, and a
    /// half-populated context would be worse than none — the UE would take the
    /// GUTI-first branch believing it has a native context and then fail every
    /// integrity check.
    pub fn capture(sec: &NasSecurityContext) -> Option<Self> {
        if !sec.is_active() {
            return None;
        }
        let keys = sec.keys();
        let kamf = keys.kamf().copied()?;
        Some(Self {
            ng_ksi: sec.ng_ksi(),
            native: sec.tsc() == SecurityContextType::Native,
            kamf: Some(kamf),
            kausf: keys.kausf().copied(),
            kseaf: keys.kseaf().copied(),
            knas_int: keys.knas_int().copied(),
            knas_enc: keys.knas_enc().copied(),
            uplink_count: (sec.uplink_count().overflow, sec.uplink_count().sqn),
            downlink_count: (sec.downlink_count().overflow, sec.downlink_count().sqn),
            integrity_algorithm: sec.integrity_algorithm() as u8,
            ciphering_algorithm: sec.ciphering_algorithm() as u8,
        })
    }

    /// Rebuild an active security context from this snapshot.
    ///
    /// Refuses rather than half-restores when the integrity or ciphering
    /// algorithm number is not one this build implements: silently substituting
    /// NIA0/NEA0 would downgrade the restored context to null security, which
    /// is the opposite of what restoring it is for.
    pub fn restore(&self) -> Result<NasSecurityContext, StateFileError> {
        let integrity = integrity_from_u8(self.integrity_algorithm).ok_or_else(|| {
            StateFileError::Malformed(format!(
                "unknown integrity algorithm NIA{}",
                self.integrity_algorithm
            ))
        })?;
        let ciphering = ciphering_from_u8(self.ciphering_algorithm).ok_or_else(|| {
            StateFileError::Malformed(format!(
                "unknown ciphering algorithm NEA{}",
                self.ciphering_algorithm
            ))
        })?;
        let Some(kamf) = self.kamf else {
            return Err(StateFileError::Malformed(
                "security context without K_AMF".to_string(),
            ));
        };

        let mut sec = NasSecurityContext::new_3gpp();
        sec.set_ng_ksi(self.ng_ksi);
        sec.set_tsc(if self.native {
            SecurityContextType::Native
        } else {
            SecurityContextType::Mapped
        });
        sec.set_algorithms(ciphering, integrity);
        {
            let keys = sec.keys_mut();
            keys.set_kamf(&kamf);
            if let Some(ref k) = self.kausf {
                keys.set_kausf(k);
            }
            if let Some(ref k) = self.kseaf {
                keys.set_kseaf(k);
            }
            if let Some(ref k) = self.knas_int {
                keys.set_knas_int(k);
            }
            if let Some(ref k) = self.knas_enc {
                keys.set_knas_enc(k);
            }
        }
        sec.set_uplink_count(NasCount::new(self.uplink_count.0, self.uplink_count.1));
        sec.set_downlink_count(NasCount::new(self.downlink_count.0, self.downlink_count.1));
        sec.activate();
        Ok(sec)
    }
}

fn integrity_from_u8(n: u8) -> Option<IntegrityAlgorithm> {
    match n {
        0 => Some(IntegrityAlgorithm::Nia0),
        1 => Some(IntegrityAlgorithm::Nia1),
        2 => Some(IntegrityAlgorithm::Nia2),
        3 => Some(IntegrityAlgorithm::Nia3),
        _ => None,
    }
}

fn ciphering_from_u8(n: u8) -> Option<CipheringAlgorithm> {
    match n {
        0 => Some(CipheringAlgorithm::Nea0),
        1 => Some(CipheringAlgorithm::Nea1),
        2 => Some(CipheringAlgorithm::Nea2),
        3 => Some(CipheringAlgorithm::Nea3),
        _ => None,
    }
}

/// The 5GMM parameters TS 24.501 Annex C.1 requires to survive a restart.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UeStateSnapshot {
    /// The assigned 5G-GUTI, as the raw 5GS mobile identity IE value.
    ///
    /// Stored as the IE octets rather than as parsed fields so a GUTI this
    /// build cannot fully parse still round-trips: the identity is opaque to
    /// the UE and is echoed back to the network verbatim.
    pub guti: Option<Vec<u8>>,
    /// The TAI list assigned in the last Registration Accept (IE value).
    pub tai_list: Option<Vec<u8>>,
    /// The last visited registered TAI (PLMN + TAC, 6 octets).
    pub last_visited_tai: Option<[u8; 6]>,
    /// 5GS update status: `U1`, `U2` or `U3` (TS 24.501 §5.1.3.2.2).
    pub update_status: StoredUpdateStatus,
    /// The native 5G NAS security context, when one was active.
    pub security: Option<SecurityContextSnapshot>,
}

/// 5GS update status as stored. A separate enum from [`UpdateStatus`] so the
/// on-disk spelling is stable if the in-memory enum is ever reordered — an
/// integer discriminant written by one build and read as a different state by
/// the next would silently change registration behaviour.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoredUpdateStatus {
    /// U1 UPDATED
    #[default]
    #[serde(rename = "U1")]
    Updated,
    /// U2 NOT UPDATED
    #[serde(rename = "U2")]
    NotUpdated,
    /// U3 ROAMING NOT ALLOWED
    #[serde(rename = "U3")]
    RoamingNotAllowed,
}

impl From<UpdateStatus> for StoredUpdateStatus {
    fn from(status: UpdateStatus) -> Self {
        match status {
            UpdateStatus::Updated => Self::Updated,
            UpdateStatus::NotUpdated => Self::NotUpdated,
            UpdateStatus::RoamingNotAllowed => Self::RoamingNotAllowed,
        }
    }
}

impl From<StoredUpdateStatus> for UpdateStatus {
    fn from(status: StoredUpdateStatus) -> Self {
        match status {
            StoredUpdateStatus::Updated => Self::Updated,
            StoredUpdateStatus::NotUpdated => Self::NotUpdated,
            StoredUpdateStatus::RoamingNotAllowed => Self::RoamingNotAllowed,
        }
    }
}

impl UeStateSnapshot {
    /// The stored 5G-GUTI as a 5GS mobile identity IE, if any.
    pub fn guti_ie(&self) -> Option<Ie5gsMobileIdentity> {
        let data = self.guti.as_ref()?;
        // A GUTI mobile identity is type(1) + PLMN(3) + AMF region(1) +
        // AMF set/ptr(2) + 5G-TMSI(4). Anything shorter cannot yield a
        // 5G-S-TMSI for a Service Request, so it is not a usable GUTI.
        if data.len() < 11 {
            warn!(
                "stored 5G-GUTI is {} octets, too short to be one: ignoring",
                data.len()
            );
            return None;
        }
        Some(Ie5gsMobileIdentity::new(
            MobileIdentityType::Guti,
            data.clone(),
        ))
    }

    /// Whether this snapshot carries anything worth writing.
    ///
    /// A snapshot with no GUTI and no security context would produce a file
    /// whose only effect on the next start is a SUCI registration — which is
    /// what happens with no file at all — so it is not written.
    pub fn is_empty(&self) -> bool {
        self.guti.is_none() && self.security.is_none()
    }

    /// Read a snapshot from `path`.
    ///
    /// A missing file is `Ok(None)`: a first run is not an error. A file that
    /// exists but does not parse is an `Err`, never a silently fresh state —
    /// see [`Self::store`] for why that distinction is load-bearing.
    pub fn load(path: &Path) -> Result<Option<Self>, StateFileError> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(StateFileError::Io(e)),
        };
        let snapshot: Self =
            serde_json::from_slice(&bytes).map_err(|e| StateFileError::Malformed(e.to_string()))?;
        Ok(Some(snapshot))
    }

    /// Write this snapshot to `path`, creating parent directories, and set
    /// `0600` on Unix.
    ///
    /// The permissions are applied to the file as CREATED, not afterwards:
    /// setting them after the write leaves a window in which K_AMF is
    /// world-readable.
    pub fn store(&self, path: &Path) -> Result<(), StateFileError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(StateFileError::Io)?;
            }
        }
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| StateFileError::Malformed(e.to_string()))?;

        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(path).map_err(StateFileError::Io)?;
        use std::io::Write;
        file.write_all(&json).map_err(StateFileError::Io)?;
        file.flush().map_err(StateFileError::Io)?;
        Ok(())
    }

    /// Remove the stored state at `path`. A missing file is success.
    ///
    /// Called when the registration context is deleted, so a subsequent
    /// restart cannot present a 5G-GUTI or reuse a security context the UE has
    /// been told to forget (TS 24.501 §5.5.1.2.5 / §5.6.1.5).
    pub fn remove(path: &Path) -> Result<(), StateFileError> {
        match std::fs::remove_file(path) {
            Ok(()) => {
                info!("stored 5GMM state removed: {}", path.display());
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StateFileError::Io(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A per-test subdirectory, so the tests also exercise `store`'s
    /// parent-directory creation. PID- and nanosecond-tagged because these
    /// tests run concurrently in one process and a shared path would make them
    /// race each other rather than test the code.
    fn temp_path(name: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir()
            .join(format!(
                "nextgsim-ue-state-{}-{name}-{nanos}",
                std::process::id()
            ))
            .join("state.json")
    }

    fn sample_security() -> SecurityContextSnapshot {
        SecurityContextSnapshot {
            ng_ksi: 3,
            native: true,
            kamf: Some([0xA5; 32]),
            kausf: Some([0x11; 32]),
            kseaf: Some([0x22; 32]),
            knas_int: Some([0x33; 16]),
            knas_enc: Some([0x44; 16]),
            uplink_count: (2, 7),
            downlink_count: (1, 9),
            integrity_algorithm: 2,
            ciphering_algorithm: 1,
        }
    }

    fn sample_snapshot() -> UeStateSnapshot {
        let mut guti = vec![0xF2, 0x99, 0xF9, 0x07, 0x01, 0x00, 0x40];
        guti.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        UeStateSnapshot {
            guti: Some(guti),
            tai_list: Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]),
            last_visited_tai: Some([0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]),
            update_status: StoredUpdateStatus::Updated,
            security: Some(sample_security()),
        }
    }

    #[test]
    fn a_snapshot_round_trips_through_a_file() {
        let path = temp_path("roundtrip");
        let snapshot = sample_snapshot();
        snapshot.store(&path).expect("store");

        let loaded = UeStateSnapshot::load(&path)
            .expect("load")
            .expect("the file exists");
        assert_eq!(loaded, snapshot, "every Annex C.1 field must survive");
        UeStateSnapshot::remove(&path).expect("remove");
    }

    /// The security context must come back ACTIVE with its COUNTs intact —
    /// that is the whole point of storing it. A context restored inactive
    /// would take the SUCI branch anyway, and one restored with zeroed COUNTs
    /// would replay NAS sequence numbers the network has already seen.
    #[test]
    fn a_restored_security_context_is_active_with_its_counts() {
        let sec = sample_security().restore().expect("restore");
        assert!(sec.is_active(), "a restored context must be usable");
        assert_eq!(sec.ng_ksi(), 3, "the stored ngKSI is what identifies it");
        assert_eq!(sec.tsc(), SecurityContextType::Native);
        assert_eq!(sec.uplink_count(), &NasCount::new(2, 7));
        assert_eq!(sec.downlink_count(), &NasCount::new(1, 9));
        assert_eq!(sec.keys().kamf(), Some(&[0xA5; 32]));
        assert_eq!(sec.keys().knas_int(), Some(&[0x33; 16]));
        assert_eq!(sec.integrity_algorithm(), IntegrityAlgorithm::Nia2);
        assert_eq!(sec.ciphering_algorithm(), CipheringAlgorithm::Nea1);
    }

    /// An unknown algorithm number is refused, not silently downgraded.
    /// Substituting NIA0/NEA0 would turn a restored context into null
    /// security — worse than not restoring at all.
    #[test]
    fn an_unknown_algorithm_is_refused_rather_than_downgraded() {
        let mut snap = sample_security();
        snap.integrity_algorithm = 9;
        assert!(matches!(
            snap.restore(),
            Err(StateFileError::Malformed(ref m)) if m.contains("NIA9")
        ));

        let mut snap = sample_security();
        snap.ciphering_algorithm = 7;
        assert!(matches!(
            snap.restore(),
            Err(StateFileError::Malformed(ref m)) if m.contains("NEA7")
        ));
    }

    /// A context with no K_AMF cannot be restored: every NAS key derives from
    /// it, so a context without it authenticates nothing.
    #[test]
    fn a_context_without_kamf_is_refused() {
        let mut snap = sample_security();
        snap.kamf = None;
        assert!(matches!(snap.restore(), Err(StateFileError::Malformed(_))));
    }

    /// An inactive context is not captured. Capturing one would write a file
    /// that makes the next start claim a native context it cannot use.
    #[test]
    fn only_an_active_context_is_captured() {
        let inactive = NasSecurityContext::new_3gpp();
        assert!(!inactive.is_active());
        assert!(SecurityContextSnapshot::capture(&inactive).is_none());

        let mut active = NasSecurityContext::new_3gpp();
        active.keys_mut().set_kamf(&[0x01; 32]);
        active.activate();
        assert!(SecurityContextSnapshot::capture(&active).is_some());
    }

    /// A missing file is a first run, not a failure; a file that exists and
    /// does not parse is a failure, not a fresh start.
    #[test]
    fn a_missing_file_is_none_and_a_corrupt_one_is_an_error() {
        let path = temp_path("missing");
        assert!(UeStateSnapshot::load(&path)
            .expect("a missing file is not an error")
            .is_none());

        let corrupt = temp_path("corrupt");
        std::fs::create_dir_all(corrupt.parent().unwrap()).unwrap();
        std::fs::write(&corrupt, b"{not json").unwrap();
        assert!(
            matches!(
                UeStateSnapshot::load(&corrupt),
                Err(StateFileError::Malformed(_))
            ),
            "a corrupt file must not read as a fresh state"
        );
        UeStateSnapshot::remove(&corrupt).expect("remove");
    }

    /// The file holds K_AMF, so it is owner-only from the moment it exists.
    #[cfg(unix)]
    #[test]
    fn the_state_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let path = temp_path("perms");
        sample_snapshot().store(&path).expect("store");
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "K_AMF must not be group/world readable"
        );
        UeStateSnapshot::remove(&path).expect("remove");
    }

    /// A GUTI too short to yield a 5G-S-TMSI is ignored rather than handed to
    /// the registration builder, which would slice past the end of it.
    #[test]
    fn a_truncated_stored_guti_is_ignored() {
        let snap = UeStateSnapshot {
            guti: Some(vec![0xF2, 0x99, 0xF9]),
            ..Default::default()
        };
        assert!(snap.guti_ie().is_none());
    }

    /// The on-disk update status is a NAME, so reordering the in-memory enum
    /// cannot silently turn U3 into U1.
    #[test]
    fn the_update_status_is_stored_by_name() {
        let snap = UeStateSnapshot {
            update_status: StoredUpdateStatus::RoamingNotAllowed,
            ..Default::default()
        };
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"U3\""), "got {json}");
        assert_eq!(
            UpdateStatus::from(StoredUpdateStatus::RoamingNotAllowed),
            UpdateStatus::RoamingNotAllowed
        );
    }

    /// A snapshot with nothing to restore is not written: the next start would
    /// take the SUCI branch either way, so the file would be noise that also
    /// has to be kept in step.
    #[test]
    fn an_empty_snapshot_is_recognised_as_empty() {
        assert!(UeStateSnapshot::default().is_empty());
        assert!(!sample_snapshot().is_empty());
        assert!(!UeStateSnapshot {
            guti: Some(vec![0xF2]),
            ..Default::default()
        }
        .is_empty());
    }
}
