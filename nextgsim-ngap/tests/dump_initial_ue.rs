//! Dump the exact APER bytes the gNB sends for InitialUEMessage, so the core's
//! decoder can be checked against real wire data.
use nextgsim_ngap::codec::encode_ngap_pdu;
use nextgsim_ngap::procedures::initial_ue_message::{
    build_initial_ue_message, AllowedSnssai, InitialUeMessageParams, NrCgi,
    RrcEstablishmentCauseValue, Tai, UeContextRequestValue, UserLocationInfoNr,
};

#[test]
fn dump_bytes() {
    // Mirror the deployed gNB: PLMN 999-70, TAC 1, cell-id bits 0,
    // ran_ue_ngap_id 1, 23-byte Registration Request.
    let params = InitialUeMessageParams {
        ran_ue_ngap_id: 1,
        nas_pdu: vec![0x7e; 23],
        user_location_info: UserLocationInfoNr {
            nr_cgi: NrCgi {
                plmn_identity: [0x99, 0xF9, 0x07],
                nr_cell_identity: 0x10,
            },
            tai: Tai {
                plmn_identity: [0x99, 0xF9, 0x07],
                tac: [0x00, 0x00, 0x01],
            },
            time_stamp: None,
        },
        rrc_establishment_cause: RrcEstablishmentCauseValue::MoSignalling,
        five_g_s_tmsi: None,
        amf_set_id: None,
        ue_context_request: Some(UeContextRequestValue::Requested),
        // The deployed gNB sets this when the UE requests a slice.
        allowed_nssai: Some(vec![AllowedSnssai { sst: 1, sd: None }]),
    };
    let pdu = build_initial_ue_message(&params).expect("build");
    let bytes = encode_ngap_pdu(&pdu).expect("encode");
    println!("LEN={}", bytes.len());
    println!(
        "HEX={}",
        bytes
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join("")
    );
}
