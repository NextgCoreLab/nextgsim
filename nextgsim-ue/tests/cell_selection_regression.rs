//! Regression tests for initial cell selection.
//!
//! These drive the real `CellSelector` the way `RrcTask` does
//! (`handle_signal_changed` -> `provide_simulated_system_info` -> the 2500 ms
//! `perform_cycle` timer) and pin the bug that stopped the UE from ever
//! registering: cell ID 0 was used as a "no cell" sentinel, so a successful
//! first selection on cell 0 compared equal to "nothing selected",
//! `perform_cell_selection` returned `None`, `ActiveCellChanged` was never sent
//! to NAS, and initial registration never started.
//!
//! Cell ID 0 is legal: the deployed gNB uses `nci: 0x000000010` with
//! `gnb_id_length: 32`, which leaves the cell-ID bits at 0.
use nextgsim_ue::rrc::cell_selection::{CellSelector, MibInfo, Plmn as CellPlmn, Sib1Info};

/// Mirror of `RrcTask::provide_simulated_system_info`.
fn provide_system_info(sel: &mut CellSelector, cell_id: i32, mcc: u16, mnc: u16) {
    sel.update_mib(
        cell_id,
        MibInfo {
            has_mib: true,
            is_barred: false,
            is_intra_freq_reselect_allowed: true,
        },
    );
    sel.update_sib1(
        cell_id,
        Sib1Info {
            has_sib1: true,
            is_reserved: false,
            nci: i64::from(cell_id),
            tac: 1,
            plmn: CellPlmn::new(mcc, mnc, false),
            q_rx_lev_min: -70,
            q_rx_lev_min_offset: None,
            q_qual_min: None,
            nid: None,
        },
    );
}

/// Get past the 1000 ms startup guard in `perform_cell_selection`.
fn wait_out_startup_guard() {
    std::thread::sleep(std::time::Duration::from_millis(1100));
}

fn selector_for(mcc: u16, mnc: u16) -> CellSelector {
    let mut sel = CellSelector::new();
    sel.set_selected_plmn(Some(CellPlmn::new(mcc, mnc, false)));
    sel
}

/// The exact regression: cell ID 0 must be selectable on the first pass.
#[test]
fn cell_id_zero_is_selectable_on_first_selection() {
    let mut sel = selector_for(999, 70);
    sel.handle_signal_change(0, -1);
    provide_system_info(&mut sel, 0, 999, 70);
    wait_out_startup_guard();

    let selected = sel
        .perform_cell_selection()
        .expect("cell 0 must be selectable; returning None is the bug this guards");
    assert_eq!(selected.cell_id, 0);
    assert_eq!(sel.current_cell().map(|c| c.cell_id), Some(0));
}

/// A non-zero cell ID must keep working; the fix must not be specific to 0.
#[test]
fn nonzero_cell_id_is_selectable_on_first_selection() {
    let mut sel = selector_for(999, 70);
    sel.handle_signal_change(7, -1);
    provide_system_info(&mut sel, 7, 999, 70);
    wait_out_startup_guard();

    let selected = sel
        .perform_cell_selection()
        .expect("cell 7 must be selectable");
    assert_eq!(selected.cell_id, 7);
}

/// Selection returns `Some` only on the camp-changed edge. NAS drives initial
/// registration off that edge, so a repeat must not re-fire it.
#[test]
fn reselecting_the_same_cell_reports_no_change() {
    let mut sel = selector_for(999, 70);
    sel.handle_signal_change(0, -1);
    provide_system_info(&mut sel, 0, 999, 70);
    wait_out_startup_guard();

    assert!(
        sel.perform_cell_selection().is_some(),
        "first camp is a change"
    );
    assert!(
        sel.perform_cell_selection().is_none(),
        "still camped on the same cell: must report no change, or NAS would \
         restart registration every cycle"
    );
}

/// Before the fix, `current_cell()` could not distinguish "camped on cell 0"
/// from "not camped". It must now be None until the first successful selection.
#[test]
fn current_cell_is_none_until_first_selection() {
    let mut sel = selector_for(999, 70);
    assert!(
        sel.current_cell().is_none(),
        "not camped before any selection"
    );

    sel.handle_signal_change(0, -1);
    provide_system_info(&mut sel, 0, 999, 70);
    wait_out_startup_guard();
    sel.perform_cell_selection();

    assert!(
        sel.current_cell().is_some(),
        "camped after a successful selection"
    );
}

/// A failed selection round must not clear an existing camp: losing the serving
/// cell is signalled by `ActiveCellLost`, not by a selection round finding
/// nothing.
#[test]
fn failed_selection_round_preserves_existing_camp() {
    let mut sel = selector_for(999, 70);
    sel.handle_signal_change(0, -1);
    provide_system_info(&mut sel, 0, 999, 70);
    wait_out_startup_guard();
    assert!(sel.perform_cell_selection().is_some());

    // A cell from a different PLMN is not suitable, and with no acceptable cell
    // either this round finds nothing new. MCC 1 / MNC 1 written without leading
    // zeros: `001`/`01` look like octal literals to the reader (and to clippy),
    // and these fields are plain integers, not BCD digit strings.
    sel.handle_signal_change(9, -1);
    provide_system_info(&mut sel, 9, 1, 1);

    assert!(sel.perform_cell_selection().is_none(), "no new camp");
    assert_eq!(
        sel.current_cell().map(|c| c.cell_id),
        Some(0),
        "the original camp must survive a round that selects nothing"
    );
}
