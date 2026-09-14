//! Turning the SMF's user-plane security policy into what the gNB actually does
//! (TS 33.501 §6.6.1, TS 38.413 §9.3.1.27; issue #32).
//!
//! The `SecurityIndication` IE states an *obligation*; the negotiated NEA/NIA and
//! this build's `up-security` feature state a *capability*. This module is the one
//! place the two meet, so the answer to "is this DRB protected, and if not may the
//! session still be established?" is decided once and read everywhere.

use nextgsim_ngap::procedures::transfer::{UpProtectionPolicy, UpSecurityPolicy};

/// The `NIA0`/`NEA0` identity: an algorithm that provides nothing
/// (TS 33.501 §5.11.1).
pub const NULL_ALGORITHM: u8 = 0;

/// Which protection the gNB will apply to one DRB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrbSecurityDecision {
    /// Apply `K_UPint` integrity protection, appending a MAC-I to every PDU.
    pub integrity: bool,
    /// Apply `K_UPenc` ciphering to the data part.
    pub ciphering: bool,
}

impl DrbSecurityDecision {
    /// Whether any protection at all is applied, and therefore whether a PDCP
    /// security binding needs installing.
    pub fn any(self) -> bool {
        self.integrity || self.ciphering
    }
}

/// Why a PDU session cannot be established.
///
/// Carries which protection was impossible, because TS 38.413 §9.3.1.2 gives the
/// two cases distinct causes and the SMF acts on them differently — it may retry
/// with a relaxed policy for one and not the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum UpSecurityRefusal {
    /// Integrity was `required` and this gNB cannot provide it.
    #[error("user-plane integrity protection is required for this session but not possible")]
    IntegrityNotPossible,
    /// Confidentiality was `required` and this gNB cannot provide it.
    #[error("user-plane confidentiality protection is required for this session but not possible")]
    ConfidentialityNotPossible,
}

/// Decide what to apply to a DRB, or refuse the session.
///
/// `available` is whether this build can protect a DRB at all — the `up-security`
/// feature. It is an argument rather than a `cfg!` inside so that both arms are
/// compiled and tested in every build; a `cfg!` here would leave the refusal path
/// unreachable in the default build and therefore untested.
///
/// A `required` policy the gNB cannot satisfy is a **refusal**, not a downgrade
/// (TS 33.501 §6.6.1): establishing the session unprotected would tell the SMF the
/// traffic is protected when it is not, which is the failure this whole issue is
/// about.
pub fn resolve(
    policy: &UpSecurityPolicy,
    ciphering_alg_id: u8,
    integrity_alg_id: u8,
    available: bool,
) -> Result<DrbSecurityDecision, UpSecurityRefusal> {
    let can_protect_integrity = available && integrity_alg_id != NULL_ALGORITHM;
    let can_cipher = available && ciphering_alg_id != NULL_ALGORITHM;

    if policy.integrity.is_mandatory() && !can_protect_integrity {
        return Err(UpSecurityRefusal::IntegrityNotPossible);
    }
    if policy.confidentiality.is_mandatory() && !can_cipher {
        return Err(UpSecurityRefusal::ConfidentialityNotPossible);
    }

    Ok(DrbSecurityDecision {
        integrity: policy.integrity.wants_protection() && can_protect_integrity,
        // NOTE the asymmetry with integrity, and it is a codec ceiling rather than
        // a choice: `PDCP-Config.cipheringDisabled` is in the Rel-15 schema's
        // extension addition group and the generated codec dropped the field, so a
        // `not-needed` confidentiality policy CANNOT be signalled to the UE. The
        // gNB therefore ciphers whenever it can, which over-protects rather than
        // under-protects — the peer still agrees, because it derives the same
        // decision from the same absent IE. `honours_confidentiality_policy` reports
        // when this deviates, and the caller logs it.
        ciphering: can_cipher,
    })
}

/// Whether the decision actually matches what the SMF asked for on the
/// confidentiality side.
///
/// Split out rather than folded into [`resolve`] so the deviation is visible to a
/// caller that has to log it, instead of being a comment nobody reads at run time.
pub fn honours_confidentiality_policy(
    policy: &UpSecurityPolicy,
    decision: DrbSecurityDecision,
) -> bool {
    policy.confidentiality != UpProtectionPolicy::NotNeeded || !decision.ciphering
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_ngap::procedures::transfer::UpProtectionPolicy::*;

    fn policy(
        integrity: UpProtectionPolicy,
        confidentiality: UpProtectionPolicy,
    ) -> UpSecurityPolicy {
        UpSecurityPolicy {
            integrity,
            confidentiality,
            max_integrity_protected_data_rate: None,
        }
    }

    /// #32, criterion 4: `required` with no acceptable algorithm is a refusal, and
    /// the cause distinguishes which protection was impossible.
    #[test]
    fn required_protection_without_an_algorithm_refuses_the_session() {
        // NIA0 is not an acceptable integrity algorithm for a `required` policy.
        assert_eq!(
            resolve(&policy(Required, Preferred), 2, NULL_ALGORITHM, true),
            Err(UpSecurityRefusal::IntegrityNotPossible)
        );
        // NEA0 likewise for confidentiality.
        assert_eq!(
            resolve(&policy(Preferred, Required), NULL_ALGORITHM, 2, true),
            Err(UpSecurityRefusal::ConfidentialityNotPossible)
        );
        // And a build that cannot protect at all refuses both.
        assert_eq!(
            resolve(&policy(Required, Preferred), 2, 2, false),
            Err(UpSecurityRefusal::IntegrityNotPossible)
        );
        assert_eq!(
            resolve(&policy(Preferred, Required), 2, 2, false),
            Err(UpSecurityRefusal::ConfidentialityNotPossible)
        );
    }

    /// A `required` policy the gNB CAN satisfy is not refused. The positive control
    /// for the test above: without it, a `resolve` that refused everything would
    /// pass.
    #[test]
    fn required_protection_with_an_algorithm_is_applied_not_refused() {
        assert_eq!(
            resolve(&policy(Required, Required), 2, 2, true),
            Ok(DrbSecurityDecision {
                integrity: true,
                ciphering: true
            })
        );
    }

    /// `preferred` never refuses, and applies protection exactly when it can.
    #[test]
    fn preferred_protection_is_applied_when_possible_and_never_refuses() {
        assert_eq!(
            resolve(&policy(Preferred, Preferred), 2, 2, true),
            Ok(DrbSecurityDecision {
                integrity: true,
                ciphering: true
            })
        );
        assert_eq!(
            resolve(
                &policy(Preferred, Preferred),
                NULL_ALGORITHM,
                NULL_ALGORITHM,
                true
            ),
            Ok(DrbSecurityDecision::default()),
            "null algorithms provide nothing, so `preferred` gets nothing -- but the \
             session still comes up"
        );
        assert_eq!(
            resolve(&policy(Preferred, Preferred), 2, 2, false),
            Ok(DrbSecurityDecision::default()),
            "a build without the feature establishes the session unprotected"
        );
    }

    /// `not-needed` integrity is honoured. Ciphering is not, and the deviation is
    /// reported rather than hidden — see the note in `resolve`.
    #[test]
    fn not_needed_integrity_is_honoured_and_not_needed_confidentiality_is_not() {
        let p = policy(NotNeeded, NotNeeded);
        let decision = resolve(&p, 2, 2, true).expect("never refuses");
        assert!(
            !decision.integrity,
            "a `not-needed` integrity policy must not append a MAC-I"
        );
        assert!(
            decision.ciphering,
            "and ciphering cannot be turned off, because PDCP-Config.cipheringDisabled \
             is absent from the generated codec"
        );
        assert!(
            !honours_confidentiality_policy(&p, decision),
            "so the deviation must be reportable, or it is a silent one"
        );

        // With nothing to cipher with, `not-needed` is honoured after all, and the
        // reporter must say so rather than always claiming a deviation.
        let unciphered = resolve(&p, NULL_ALGORITHM, 2, true).expect("never refuses");
        assert!(!unciphered.ciphering);
        assert!(honours_confidentiality_policy(&p, unciphered));
        // And a policy that wanted ciphering is never a deviation.
        assert!(honours_confidentiality_policy(
            &policy(NotNeeded, Required),
            resolve(&policy(NotNeeded, Required), 2, 2, true).unwrap()
        ));
    }

    /// The two protections are decided independently: an integrity-only and a
    /// ciphering-only DRB are both reachable. Pinned because a `resolve` that
    /// returned one flag for both would satisfy every test above that uses matching
    /// policies.
    #[test]
    fn integrity_and_ciphering_are_decided_independently() {
        assert_eq!(
            resolve(&policy(Required, NotNeeded), NULL_ALGORITHM, 2, true),
            Ok(DrbSecurityDecision {
                integrity: true,
                ciphering: false
            }),
            "integrity on, ciphering impossible"
        );
        assert_eq!(
            resolve(&policy(NotNeeded, Required), 2, NULL_ALGORITHM, true),
            Ok(DrbSecurityDecision {
                integrity: false,
                ciphering: true
            }),
            "ciphering on, integrity not wanted"
        );
    }

    /// `any()` decides whether a PDCP binding is installed at all, so it must not
    /// claim protection for a decision that applies none.
    #[test]
    fn any_reports_whether_there_is_anything_to_install() {
        assert!(!DrbSecurityDecision::default().any());
        assert!(DrbSecurityDecision {
            integrity: true,
            ciphering: false
        }
        .any());
        assert!(DrbSecurityDecision {
            integrity: false,
            ciphering: true
        }
        .any());
    }
}
