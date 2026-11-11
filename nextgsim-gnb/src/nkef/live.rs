//! The gNB side of demo d447a-nkef-live (nextgsim#488): the tees that feed NKEF
//! live facts, and the helpers they share.
//!
//! A tee sends only while the demo is active in mode `on`. In a build without
//! the `nkef-sba` feature the demo is not registered, so [`tees_on`] is always
//! false and no tee ever sends: the default gNB is unchanged. A tee never waits
//! on NKEF: a full or closed inbox drops the fact.

use nextgsim_common::Plmn;
use nextgsim_nkef::live::{Domain, Fact, FactKind};

use crate::demo::D447A_KEY;
use crate::tasks::{GnbTaskBase, NkefMessage};

/// The demo's active mode, if it is active.
pub fn mode() -> Option<String> {
    nextgsim_demo::active_mode(D447A_KEY)
}

/// Whether the tees feed NKEF: the demo is active in mode `on`.
pub fn tees_on() -> bool {
    mode().as_deref() == Some("on")
}

/// Now, in ms since the Unix epoch.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Hand `fact` to NKEF without waiting, if the tees are on.
pub fn offer(base: &GnbTaskBase, fact: Fact) {
    if !tees_on() {
        return;
    }
    if let Some(sixg) = &base.sixg {
        let _ = sixg.nkef_tx.try_send(NkefMessage::LiveFact(fact));
    }
}

/// The GUAMI as both N2 and the NRF can name it: `<mcc>-<mnc>-<amfId>`, the
/// AMF ID as TS 29.571 `AmfId` (6 hex digits: region, set, pointer).
pub fn guami_key(mcc: &str, mnc: &str, amf_id_hex: &str) -> String {
    format!("{mcc}-{mnc}-{}", amf_id_hex.to_ascii_lowercase())
}

/// [`guami_key`] of an NGAP GUAMI (TS 38.413 §9.3.3.3).
pub fn ngap_guami_key(plmn: [u8; 3], region: u8, set: u16, pointer: u8) -> String {
    let p = Plmn::decode(plmn);
    let mnc = if p.long_mnc {
        format!("{:03}", p.mnc)
    } else {
        format!("{:02}", p.mnc)
    };
    let amf_id =
        (u32::from(region) << 16) | (u32::from(set & 0x3ff) << 6) | u32::from(pointer & 0x3f);
    guami_key(&format!("{:03}", p.mcc), &mnc, &format!("{amf_id:06x}"))
}

/// RLS tee: the cell's measured load over the last window.
pub fn offer_cell_load(base: &GnbTaskBase, connected_ues: u32, throughput_mbps: Option<f32>) {
    let nci = base.config.nci;
    offer(
        base,
        Fact::new(
            Domain::Ran,
            FactKind::CellLoad,
            "gnb-rls",
            format!("cell-{nci}"),
            now_ms(),
        )
        .with("connected_ues", connected_ues)
        .with(
            "throughput_mbps",
            format!("{:.3}", throughput_mbps.unwrap_or(0.0)),
        ),
    );
}

/// RLS tee: a UE's serving cell and received power.
pub fn offer_ue_radio(base: &GnbTaskBase, ue_id: i32, rsrp_dbm: i32) {
    offer(
        base,
        Fact::new(
            Domain::Ran,
            FactKind::UeRadio,
            "gnb-rls",
            format!("ue-{ue_id}"),
            now_ms(),
        )
        .with("cell", base.config.nci)
        .with("rsrp_dbm", rsrp_dbm),
    );
}

/// N2 tee: this gNB and the AMF it set up NG with.
pub fn offer_ng_setup(
    base: &GnbTaskBase,
    amf_id: i32,
    response: &nextgsim_ngap::procedures::NgSetupResponseData,
) {
    let now = now_ms();
    offer(
        base,
        Fact::new(
            Domain::N2,
            FactKind::Gnb,
            "gnb-ngap",
            format!("gnb-{}", base.config.gnb_id()),
            now,
        )
        .with("nci", base.config.nci)
        .with("tac", base.config.tac),
    );
    let mut amf = Fact::new(
        Domain::N2,
        FactKind::Amf,
        "gnb-ngap",
        format!("amf-{amf_id}"),
        now,
    )
    .with("amf_name", &response.amf_name);
    if let Some(item) = response.served_guami_list.first() {
        let g = &item.guami;
        amf = amf.with(
            "guami",
            ngap_guami_key(
                g.plmn_identity,
                g.amf_region_id,
                g.amf_set_id,
                g.amf_pointer,
            ),
        );
    }
    offer(base, amf);
}

/// N2 tee: a UE's association with the AMF.
pub fn offer_ue_n2(base: &GnbTaskBase, ue_id: i32, amf_ctx_id: i32, amf_ue_ngap_id: i64) {
    offer(
        base,
        Fact::new(
            Domain::N2,
            FactKind::UeN2,
            "gnb-ngap",
            format!("ue-{ue_id}"),
            now_ms(),
        )
        .with("amf", format!("amf-{amf_ctx_id}"))
        .with("amf_ue_ngap_id", amf_ue_ngap_id),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ngap_guami_keys_as_the_nrf_names_it() {
        // PLMN 001/01, region 2, set 0x155, pointer 0x2a: AmfId 0x02556a.
        let plmn = Plmn::new(1, 1, false).encode();
        assert_eq!(ngap_guami_key(plmn, 2, 0x155, 0x2a), "001-01-02556a");
        assert_eq!(guami_key("001", "01", "02556A"), "001-01-02556a");
    }

    #[test]
    #[cfg(not(feature = "nkef-sba"))]
    fn without_the_demo_no_tee_is_on() {
        assert!(!tees_on());
    }
}
