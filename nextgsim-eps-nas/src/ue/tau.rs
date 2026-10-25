//! Normal and periodic tracking area updating (TS 24.301 §5.5.3.2, #356).
//!
//! The update is owed for two reasons this UE can have: T3412 expiry
//! (§5.3.5, "periodic updating") and a serving cell whose TAI is not in the
//! TAI list (§5.5.3.2.2 a), "TA updating"), plus a paging in
//! EMM-REGISTERED.ATTEMPTING-TO-UPDATE (r). It goes as the initial NAS message
//! of the next connection, integrity protected and unciphered (§4.4.5), with
//! the EPS bearer context status IE and, when uplink data or a paging waits,
//! the "active" flag.

use std::hash::{BuildHasher, Hasher};
use std::time::{Duration, Instant};

use super::{EmmState, EpsNasUe, NasOutput, MAX_TAU_ATTEMPTS, T3411, T3430, T3440};
use crate::ie::{EpsBearerContextStatus, NasKsi, Tai};
use crate::msg::{self, emm_cause, EpsUpdateType, TrackingAreaUpdateRequest};
use crate::security::{header_type, EpsSecurityContext};

/// TS 23.122 §4.4.3.3.1: the default T of the higher priority PLMN search;
/// TAU REJECT #42 holds E-UTRA off for twice that (§5.5.3.2.5).
const T_DEFAULT: Duration = Duration::from_secs(60 * 60);

/// EMM cause #42 "severe network failure".
const SEVERE_NETWORK_FAILURE: u8 = 42;

impl EpsNasUe {
    /// The lower layers camped on a cell of tracking area `tac` in the
    /// selected PLMN (TS 36.304 §5.2).
    ///
    /// - EMM-REGISTERED (and ATTEMPTING-TO-UPDATE): a TAI outside the TAI list
    ///   owes a "TA updating" update (§5.5.3.2.2 a), not in a forbidden TA;
    ///   entering a new TA in ATTEMPTING-TO-UPDATE resets the attempt
    ///   counter (§5.5.3.1) and starts the update without waiting for T3411
    ///   or T3402 (§5.2.3.2.2).
    /// - After TAU REJECT #13 / #15 (§5.2.3.2.3), a TA that is not forbidden
    ///   is updated into; after #12, a UE that is EMM-DEREGISTERED attaches
    ///   there.
    pub fn on_serving_cell(&mut self, tac: u16) {
        let tai = Tai {
            plmn: self.cfg.plmn,
            tac,
        };
        if self.current_tai == Some(tai) {
            return;
        }
        self.current_tai = Some(tai);
        let forbidden = self.forbidden_tais.contains(&tai);
        match self.state {
            EmmState::Registered | EmmState::AttemptingToUpdate
                if !self.tai_list.contains(&tai) =>
            {
                if forbidden {
                    log::info!("EMM: TA {tac} is forbidden; no tracking area update there");
                    return;
                }
                log::info!("EMM: TA {tac} is not in the TAI list; tracking area update due");
                if self.state == EmmState::AttemptingToUpdate {
                    self.tau_attempts = 0;
                    self.tau_retry_at = None;
                }
                self.tau_ta_due = true;
            }
            EmmState::RegisteredLimited(cause) if !forbidden => {
                log::info!("EMM: TA {tac} after TAU REJECT #{cause}; tracking area update due");
                self.state = EmmState::AttemptingToUpdate;
                self.tau_attempts = 0;
                self.tau_retry_at = None;
                self.tau_ta_due = true;
            }
            EmmState::NoService(emm_cause::TA_NOT_ALLOWED)
                if !forbidden && !self.forbidden_tais.is_empty() =>
            {
                log::info!("EMM: TA {tac} after TAU REJECT #12; attach due");
                self.state = EmmState::Deregistered;
            }
            _ => {}
        }
    }

    /// The EPS bearer context status (§9.9.2.1) of the active contexts.
    pub fn eps_bearer_context_status(&self) -> EpsBearerContextStatus {
        EpsBearerContextStatus::from_ebis(
            self.bearer
                .iter()
                .map(|b| b.ebi)
                .chain(self.dedicated.iter().map(|b| b.ebi)),
        )
    }

    /// T3412 as the network last set it; `None` is deactivated.
    pub fn t3412(&self) -> Option<Duration> {
        self.t3412_value
    }

    /// The tracking area updating attempt counter (§5.5.3.1).
    pub fn tau_attempts(&self) -> u8 {
        self.tau_attempts
    }

    /// Registered in the current TA, by ATTACH ACCEPT or TRACKING AREA UPDATE
    /// ACCEPT: EU1 UPDATED, the attempt counter reset (§5.5.3.1), no update
    /// owed -- unless the TAI list the network gave leaves out the TA the UE
    /// camps on (§5.5.3.2.2 a).
    pub(super) fn registered_in_current_ta(&mut self) {
        self.updated = true;
        self.tau_attempts = 0;
        self.tau_retry_at = None;
        self.tau_periodic_due = false;
        self.tau_ta_due = false;
        self.last_visited_tai = self.current_tai.or_else(|| self.tai_list.first().copied());
        if let Some(tai) = self.current_tai {
            if !self.tai_list.contains(&tai) {
                log::warn!(
                    "EMM: the TAI list leaves out the serving TA {}; tracking area update due",
                    tai.tac
                );
                self.tau_ta_due = true;
            }
        }
    }

    /// Leaving EMM-REGISTERED: T3412 stops (§5.3.5) and no update is owed.
    pub(super) fn stop_tau(&mut self) {
        self.t3412 = None;
        self.t3430 = None;
        self.tau_retry_at = None;
        self.tau_attempts = 0;
        self.tau_periodic_due = false;
        self.tau_ta_due = false;
        self.tau_in_progress = None;
        self.detach_after_tau = false;
    }

    /// §5.3.5: T3412 "is reset and started with its initial value, when the
    /// UE changes from EMM-CONNECTED ... to EMM-IDLE mode", if not
    /// deactivated. It is not restarted while running: a connection that was
    /// never set up did not stop it.
    pub(super) fn start_t3412(&mut self, now: Instant) {
        let registered = matches!(
            self.state,
            EmmState::Registered | EmmState::AttemptingToUpdate | EmmState::RegisteredLimited(_)
        );
        if registered && self.t3412.is_none() {
            self.t3412 = self.t3412_value.map(|v| now + v);
        }
    }

    /// T3412, T3411 and T3402 of the tracking area update.
    pub(super) fn on_tau_timers(&mut self, now: Instant) {
        if self.t3412.is_some_and(|t| now >= t) {
            self.t3412 = None;
            if matches!(
                self.state,
                EmmState::Registered
                    | EmmState::AttemptingToUpdate
                    | EmmState::RegisteredLimited(_)
                    | EmmState::ServiceRequestInitiated
            ) {
                // §5.3.5: the periodic update starts, delayed until
                // EMM-REGISTERED.NORMAL-SERVICE if the UE is not there.
                log::info!("EMM: T3412 expired; periodic tracking area update due (§5.3.5)");
                self.tau_periodic_due = true;
            }
        }
        if self.tau_retry_at.is_some_and(|t| now >= t) {
            self.tau_retry_at = None;
            if self.tau_attempts >= MAX_TAU_ATTEMPTS {
                // §5.5.3.1: the counter is reset when T3402 expires.
                self.tau_attempts = 0;
            }
            if !self.tau_periodic_due && !self.tau_ta_due {
                self.tau_ta_due = true;
            }
            log::info!("EMM: T3411/T3402 expired; tracking area update again (§5.5.3.2.6)");
        }
    }

    /// TRACKING AREA UPDATE REQUEST (§5.5.3.2.2), when one is owed, the UE is
    /// EMM-REGISTERED (NORMAL-SERVICE or ATTEMPTING-TO-UPDATE), and neither
    /// T3346 (l) nor T3411/T3402 holds it back -- a paging does (r, l).
    pub(super) fn tau_request(&mut self, now: Instant) -> Option<Vec<u8>> {
        if !matches!(
            self.state,
            EmmState::Registered | EmmState::AttemptingToUpdate
        ) || !(self.tau_periodic_due || self.tau_ta_due)
        {
            return None;
        }
        let paging = self.paging_response_pending;
        if !paging && self.t3346.is_some_and(|t| now < t) {
            // §5.5.3.2.6 l): no update while T3346 runs; outside the TAI list,
            // EU2 NOT UPDATED and ATTEMPTING-TO-UPDATE.
            if self.tau_ta_due && self.state == EmmState::Registered {
                self.updated = false;
                self.state = EmmState::AttemptingToUpdate;
            }
            return None;
        }
        if !paging && self.tau_retry_at.is_some_and(|t| now < t) {
            return None;
        }
        let Some(old_guti) = self.guti else {
            log::warn!("EMM: tracking area update due but no GUTI; not sent");
            return None;
        };
        // §5.5.3.2.2: "periodic updating" when T3412 expiry is the only
        // reason, "TA updating" otherwise.
        let update_type = if self.tau_ta_due {
            EpsUpdateType::TaUpdating
        } else {
            EpsUpdateType::Periodic
        };
        // §5.5.3.2.2: the "active" flag for uplink user data pending to go
        // via the user plane, and for a paging (r); ATTEMPTING-TO-UPDATE sends
        // no user data (§5.2.3.2.2), so it remembers none.
        let active = paging || (self.state == EmmState::Registered && self.uplink_pending);
        let req = TrackingAreaUpdateRequest {
            update_type,
            active,
            ksi: self
                .ctx
                .as_ref()
                .map_or(NasKsi::NO_KEY, EpsSecurityContext::ksi),
            old_guti,
            // §8.2.29.7: "unless the UE performs a periodic tracking area
            // updating procedure".
            ue_network_capability: (update_type == EpsUpdateType::TaUpdating)
                .then_some(self.capability),
            last_visited_tai: self.last_visited_tai,
            // §5.5.3.2.2: "the UE may also include an EPS bearer context
            // status IE ... indicating which EPS bearer contexts are active".
            eps_bearer_context_status: Some(self.eps_bearer_context_status()),
        }
        .encode();
        // §5.5.3.2.2 / §4.4.5: integrity protected with the current context,
        // never ciphered; plain without one.
        let pdu = match &mut self.ctx {
            Some(c) => c.protect(&req, header_type::INTEGRITY),
            None => req,
        };
        self.uplink_pending = false;
        self.paging_response_pending = false;
        // "If timer T3402 is currently running, the UE shall stop timer
        // T3402. If timer T3411 is currently running, the UE shall stop timer
        // T3411."
        self.tau_retry_at = None;
        self.t3430 = Some(now + T3430);
        self.state = EmmState::TrackingAreaUpdatingInitiated;
        self.tau_in_progress = Some((update_type, active));
        log::info!(
            "EMM: TRACKING AREA UPDATE REQUEST ({update_type:?}{}); T3430 started",
            if active { ", active" } else { "" }
        );
        Some(pdu)
    }

    /// The abnormal cases a), b), c), d) of §5.5.3.2.6, items 1) to 5).
    /// `counter_to_5` is d)'s "the UE should set the tracking area updating
    /// attempt counter to 5" for #95, #96, #97, #99 and #111.
    pub(super) fn tau_failed(&mut self, now: Instant, counter_to_5: bool) {
        self.t3430 = None;
        self.tau_in_progress = None;
        if counter_to_5 {
            self.tau_attempts = MAX_TAU_ATTEMPTS;
        } else if self.tau_attempts < MAX_TAU_ATTEMPTS {
            self.tau_attempts += 1;
        }
        let in_tai_list = self.current_tai.is_none_or(|t| self.tai_list.contains(&t));
        if self.tau_attempts < MAX_TAU_ATTEMPTS {
            self.tau_retry_at = Some(now + T3411);
            if in_tai_list && self.updated {
                // 3): EU1 UPDATED, EMM-REGISTERED.NORMAL-SERVICE, T3411.
                self.state = EmmState::Registered;
            } else {
                // 4): EU2 NOT UPDATED, ATTEMPTING-TO-UPDATE, T3411.
                self.updated = false;
                self.state = EmmState::AttemptingToUpdate;
            }
            log::info!(
                "EMM: tracking area update attempt {} failed; T3411 ({:?})",
                self.tau_attempts,
                self.state
            );
        } else {
            // 5): T3402, EU2 NOT UPDATED, ATTEMPTING-TO-UPDATE; a zero T3402
            // is its own expiry.
            self.updated = false;
            self.state = EmmState::AttemptingToUpdate;
            if self.t3402.is_zero() {
                self.tau_attempts = 0;
                self.tau_retry_at = None;
            } else {
                self.tau_retry_at = Some(now + self.t3402);
            }
            log::info!(
                "EMM: tracking area update failed {MAX_TAU_ATTEMPTS} times; T3402 ({} s)",
                self.t3402.as_secs()
            );
        }
    }

    /// TRACKING AREA UPDATE ACCEPT (§5.5.3.2.4).
    pub(super) fn on_tau_accept(
        &mut self,
        a: msg::TrackingAreaUpdateAccept,
        now: Instant,
    ) -> NasOutput {
        if self.state != EmmState::TrackingAreaUpdatingInitiated {
            log::warn!(
                "EMM: TRACKING AREA UPDATE ACCEPT in {:?}; ignored",
                self.state
            );
            return NasOutput::default();
        }
        let (update_type, active) = self
            .tau_in_progress
            .take()
            .unwrap_or((EpsUpdateType::TaUpdating, false));
        // "the UE shall stop timer T3430, reset the service request attempt
        // counter, tracking area updating attempt counter, enter state
        // EMM-REGISTERED and set the EPS update status to EU1 UPDATED".
        self.t3430 = None;
        self.state = EmmState::Registered;
        if let Some(g) = a.guti {
            self.guti = Some(g);
        }
        if let Some(l) = a.tai_list {
            self.tai_list = l;
        }
        // T3412 extended value over T3412 value; with neither, the stored one.
        if let Some(v) = a.t3412_extended_s.or(a.t3412_s) {
            self.t3412_value = v.filter(|s| *s > 0).map(Duration::from_secs);
        }
        if let Some(s) = a.t3402_s {
            self.t3402 = Duration::from_secs(s);
        }
        self.registered_in_current_ta();
        // The accept passed the integrity check of the current context on
        // this connection: the MME has secured the exchange (§4.4.2.3,
        // §4.4.5), so TRACKING AREA UPDATE COMPLETE goes ciphered.
        if self.ctx.is_some() {
            self.secure_exchange = true;
        }
        log::info!(
            "EMM: TRACKING AREA UPDATE ACCEPT ({update_type:?}, result {}){}",
            a.result,
            a.guti
                .map_or(String::new(), |g| format!(", M-TMSI {:#010x}", g.m_tmsi))
        );
        if let Some(status) = a.eps_bearer_context_status {
            if self.reconcile_bearers(status) {
                // The default bearer is gone and with it the only PDN
                // connection: EMM-REGISTERED without PDN connection is not
                // supported here, so the UE attaches again, as for #40.
                self.t3440 = Some(now + T3440);
                return NasOutput::default();
            }
        }
        let mut out = NasOutput::default();
        if a.guti.is_some() {
            // "If the TRACKING AREA UPDATE ACCEPT message contained a GUTI, the
            // UE shall return a TRACKING AREA UPDATE COMPLETE message".
            out.uplink
                .extend(self.send_emm(msg::encode_tau_complete()).uplink);
        }
        if std::mem::take(&mut self.detach_after_tau) {
            // §5.5.3.2.6 m): the detach asked for during the update.
            out.uplink
                .extend(self.send_detach_request(false, now).uplink);
            return out;
        }
        if !active {
            // §5.3.1.2.1 b): no "active" flag, started in EMM-IDLE, no user
            // plane: T3440 lets the network release the connection.
            self.t3440 = Some(now + T3440);
        }
        out
    }

    /// §5.5.3.2.4: "the UE shall deactivate all those EPS bearers contexts
    /// locally ... which are active in the UE but are indicated by the MME as
    /// being inactive"; contexts the MME lists but the UE does not hold are
    /// ignored. Returns whether the default bearer went, which takes the
    /// UE's only PDN connection with it: the UE then detaches locally (and
    /// attaches again).
    fn reconcile_bearers(&mut self, status: EpsBearerContextStatus) -> bool {
        self.dedicated.retain(|b| {
            let keep = status.is_active(b.ebi);
            if !keep {
                log::info!(
                    "ESM: EPS bearer {} inactive at the MME; deactivated locally (§5.5.3.2.4)",
                    b.ebi
                );
            }
            keep
        });
        let Some(default) = self.bearer.as_ref().map(|b| b.ebi) else {
            return false;
        };
        if status.is_active(default) {
            return false;
        }
        log::warn!(
            "ESM: default EPS bearer {default} inactive at the MME; the PDN connection is gone \
             and EMM-REGISTERED without PDN connection is not supported: local detach, attach"
        );
        self.local_detach();
        true
    }

    /// TRACKING AREA UPDATE REJECT (§5.5.3.2.5). `protected`: it passed the
    /// integrity check (#22 then takes the T3346 value the network gave).
    pub(super) fn on_tau_reject(
        &mut self,
        r: msg::TrackingAreaUpdateReject,
        protected: bool,
        now: Instant,
    ) -> NasOutput {
        if self.state != EmmState::TrackingAreaUpdatingInitiated {
            log::warn!(
                "EMM: TRACKING AREA UPDATE REJECT #{} in {:?}; ignored",
                r.cause,
                self.state
            );
            return NasOutput::default();
        }
        log::warn!("EMM: TRACKING AREA UPDATE REJECT #{}", r.cause);
        // "the UE shall stop timer T3430"
        self.t3430 = None;
        self.tau_in_progress = None;
        // §5.3.1.2.1 a), d), f) and i) name the causes after which T3440 runs.
        match r.cause {
            emm_cause::ILLEGAL_UE
            | emm_cause::ILLEGAL_ME
            | emm_cause::EPS_SERVICES_NOT_ALLOWED
            | emm_cause::EPS_AND_NON_EPS_NOT_ALLOWED => {
                // EU3; GUTI, last visited registered TAI, TAI list and eKSI
                // deleted; the USIM invalid for EPS services; EMM-DEREGISTERED.
                self.clear_bearers();
                self.last_visited_tai = None;
                self.invalidate(r.cause);
                self.t3440 = Some(now + T3440);
            }
            emm_cause::UE_IDENTITY_CANNOT_BE_DERIVED => {
                // EU2; GUTI, last visited registered TAI, TAI list and eKSI
                // deleted; EMM-DEREGISTERED.NORMAL-SERVICE and attach.
                self.guti = None;
                self.last_visited_tai = None;
                self.tai_list.clear();
                self.ctx = None;
                self.local_detach();
                self.t3440 = Some(now + T3440);
            }
            emm_cause::IMPLICITLY_DETACHED | emm_cause::NO_EPS_BEARER_CONTEXT_ACTIVATED => {
                // #10: EMM-DEREGISTERED.NORMAL-SERVICE and a new attach (only
                // a mapped or partial native context would be deleted; this
                // UE holds neither). #40: the EPS bearer contexts deactivated
                // locally, the same.
                self.local_detach();
                self.t3440 = Some(now + T3440);
            }
            emm_cause::PLMN_NOT_ALLOWED
            | emm_cause::SERVICE_OPTION_NOT_AUTHORIZED_IN_PLMN
            | emm_cause::EPS_NOT_ALLOWED_IN_PLMN
            | emm_cause::TA_NOT_ALLOWED => {
                // EU3; GUTI, last visited registered TAI, TAI list and eKSI
                // deleted; the counter reset; EMM-DEREGISTERED.PLMN-SEARCH
                // (#11, #14, #35: this single-PLMN UE has no other) or
                // .LIMITED-SERVICE with the TA forbidden (#12).
                if r.cause == emm_cause::TA_NOT_ALLOWED {
                    self.forbid_current_ta();
                }
                self.clear_bearers();
                self.stop_tau();
                self.guti = None;
                self.last_visited_tai = None;
                self.tai_list.clear();
                self.ctx = None;
                self.updated = false;
                self.state = EmmState::NoService(r.cause);
                self.t3440 = Some(now + T3440);
            }
            emm_cause::ROAMING_NOT_ALLOWED_IN_TA | emm_cause::NO_SUITABLE_CELLS_IN_TA => {
                // EU3; the counter reset; the TA forbidden and out of the TAI
                // list; EMM-REGISTERED.PLMN-SEARCH (#13) or .LIMITED-SERVICE
                // (#15). The GUTI and the bearer contexts stay.
                self.forbid_current_ta();
                if let Some(tai) = self.current_tai {
                    self.tai_list.retain(|t| *t != tai);
                }
                self.tau_attempts = 0;
                self.tau_retry_at = None;
                self.updated = false;
                self.state = EmmState::RegisteredLimited(r.cause);
                self.t3440 = Some(now + T3440);
            }
            emm_cause::CONGESTION if r.t3346_s.flatten().is_some_and(|s| s > 0) => {
                // Abort, the counter reset, EU2, ATTEMPTING-TO-UPDATE; T3346
                // with the network's value when the reject was integrity
                // protected, "a random value from the default range" (15 to
                // 30 minutes, TS 24.008 Table 11.3a NOTE 7) otherwise.
                let wait = if protected {
                    Duration::from_secs(r.t3346_s.flatten().unwrap_or(0))
                } else {
                    random_t3346()
                };
                self.tau_attempts = 0;
                self.tau_retry_at = None;
                self.updated = false;
                self.state = EmmState::AttemptingToUpdate;
                self.t3346 = Some(now + wait);
                self.t3440 = Some(now + T3440);
                log::info!("EMM: T3346 {} s", wait.as_secs());
            }
            SEVERE_NETWORK_FAILURE => {
                // EU2; GUTI, last visited registered TAI, TAI list, eKSI
                // deleted; the counter at 5; E-UTRA off for 2 T, then a PLMN
                // selection -- for this single-RAT UE, an attach after it.
                self.clear_bearers();
                self.stop_tau();
                self.guti = None;
                self.last_visited_tai = None;
                self.tai_list.clear();
                self.ctx = None;
                self.updated = false;
                self.tau_attempts = MAX_TAU_ATTEMPTS;
                self.state = EmmState::AttemptingToAttach;
                self.retry_at = Some(now + 2 * T_DEFAULT);
                self.t3440 = Some(now + T3440);
            }
            // #22 without a usable T3346, #25 off a CSG cell (this UE has no
            // CSG), #31, #78, #80, #83 and every other value: abnormal case
            // d); #95, #96, #97, #99 and #111 set the counter to 5.
            c => self.tau_failed(now, matches!(c, 95 | 96 | 97 | 99 | 111)),
        }
        NasOutput::default()
    }

    /// Add the serving TAI to the forbidden tracking areas (§5.5.3.2.5).
    fn forbid_current_ta(&mut self) {
        if let Some(tai) = self.current_tai {
            if !self.forbidden_tais.contains(&tai) {
                self.forbidden_tais.push(tai);
            }
        }
    }
}

/// A T3346 value "chosen randomly from a default value range of 15 – 30
/// minutes" (TS 24.008 Table 11.3a NOTE 7), from the standard library's
/// random hash keys.
fn random_t3346() -> Duration {
    let r = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();
    Duration::from_secs(15 * 60 + r % (15 * 60 + 1))
}

#[cfg(test)]
mod tests;
