//! Demo D440A, model-predicted paging (nextgsim#462 PR 3b), the gNB side.
//!
//! **Non-normative.** TS 38.300 §16.20.2: "AI/ML algorithms and models are out
//! of 3GPP scope". What is normative is the IE the prediction is carried in.
//!
//! At UE CONTEXT RELEASE the NGAP task asks the NWDAF task for a
//! recommendation for the released UE ([`recommend`]). The NWDAF task
//! extrapolates the UE's heartbeat positions [`HORIZON_MS`] ahead with the
//! demo mode's predictor ([`Predictor`]) and resolves the predicted position
//! to the nearest cell of this gNB's paging grid ([`Grid`]). The answer becomes
//! *Information on Recommended Cells and RAN Nodes for Paging* (TS 38.413
//! §9.3.1.100) in the UE CONTEXT RELEASE COMPLETE:
//! - *Recommended Cells for Paging* (§9.3.1.71): this cell first, as visited,
//!   with the time the UE stayed in it; then the predicted cell, non-visited,
//!   after the visited cell it is associated with;
//! - *Recommended RAN Nodes for Paging* (§9.3.1.101): this gNB first, as
//!   visited; then the predicted gNB, non-visited, after it.
//!
//! A prediction that stays in this cell lists this cell and this gNB only.
//! Mode `page-all` asks nothing and sends no IE, so the AMF pages the
//! registration area exactly as with no demo.
//!
//! An AMF that supports the IE (nextgcore amfd `paging.strategy:
//! recommended-first`, TS 23.502 §4.2.3.3) pages the listed gNBs on its first
//! attempt and the whole registration area on every retransmission.
//!
//! Every constant here, the grid and the route seeds
//! (`config/demo-ai-paging/d440a-scenario.yaml`) were committed before the
//! first replay ran; nothing is tuned after.

use nextgsim_common::config::GnbConfig;
use nextgsim_ngap::procedures::ue_context_release::{
    RecommendedCell, RecommendedPagingInfo, RecommendedRanNode,
};
use nextgsim_nwdaf::{NwdafManager, Vector3};

/// How far ahead the UE's position is predicted, from its newest heartbeat:
/// the middle of the replay's 10..30 s release-to-page range.
pub const HORIZON_MS: u32 = 20_000;

/// How long the NGAP task waits for the NWDAF task's answer before it sends
/// the Release Complete without the IE.
pub const RECOMMEND_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(200);

/// The predictor a D440A mode runs. `page-all` has none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predictor {
    /// The UE stays where it was served: this gNB alone is recommended.
    LastServed,
    /// Constant-turn extrapolation over a Kalman-filtered track (the live
    /// physics baseline, `nextgsim_nwdaf::constant_turn`).
    ConstantTurn,
    /// Least-squares constant velocity (the control, #467).
    LeastSquares,
    /// The d440b-verified ONNX trajectory model (#463), through the NWDAF
    /// manager's serving path. A prediction no verified model served is
    /// labelled `fallback`.
    Onnx,
}

impl Predictor {
    /// The predictor of `mode`; `None` for `page-all` or an unknown mode.
    pub fn for_mode(mode: &str) -> Option<Self> {
        match mode {
            "last-served" => Some(Self::LastServed),
            "ct" => Some(Self::ConstantTurn),
            "lsq" => Some(Self::LeastSquares),
            "onnx" => Some(Self::Onnx),
            _ => None,
        }
    }

    /// The mode name.
    pub fn mode(self) -> &'static str {
        match self {
            Self::LastServed => "last-served",
            Self::ConstantTurn => "ct",
            Self::LeastSquares => "lsq",
            Self::Onnx => "onnx",
        }
    }
}

/// One cell of the paging grid, and the gNB it belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridCell {
    /// The NR Cell Identity (36 bits).
    pub nci: u64,
    /// The gNB ID, the leftmost `gnb_id_len` bits of `nci` (TS 38.413 §9.3.1.6).
    pub gnb_id: u32,
    /// The gNB ID length, 22..=32.
    pub gnb_id_len: u8,
    /// Where the cell's antenna stands, metres.
    pub position: Vector3,
}

/// This gNB's cell and its paging-grid neighbours, all in this gNB's PLMN.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    /// The PLMN Identity on the wire.
    pub plmn: [u8; 3],
    /// This gNB's own cell.
    pub own: GridCell,
    /// The configured `paging_grid`, in config order.
    pub neighbours: Vec<GridCell>,
}

impl Grid {
    /// The grid `config` describes.
    pub fn from_config(config: &GnbConfig) -> Self {
        let pos = |p: nextgsim_common::config::SimPosition| {
            Vector3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z))
        };
        let own_len = config.gnb_id_length.clamp(22, 32);
        Self {
            plmn: config.plmn.encode(),
            own: GridCell {
                nci: config.nci & 0xF_FFFF_FFFF,
                gnb_id: config.gnb_id(),
                gnb_id_len: own_len,
                position: pos(config.sim_position),
            },
            neighbours: config
                .paging_grid
                .iter()
                .map(|e| GridCell {
                    nci: e.nci,
                    gnb_id: e.gnb_id(own_len),
                    gnb_id_len: e.gnb_id_length.unwrap_or(own_len).clamp(22, 32),
                    position: pos(e.position),
                })
                .collect(),
        }
    }

    /// The cell nearest `p`: under the simulator's distance channel model the
    /// strongest, so the one a UE there would camp on. A tie goes to this
    /// gNB's own cell, then to the earlier grid entry.
    pub fn nearest(&self, p: &Vector3) -> GridCell {
        let d = |c: &GridCell| {
            (c.position.x - p.x)
                .hypot(c.position.y - p.y)
                .hypot(c.position.z - p.z)
        };
        let mut best = self.own;
        let mut best_d = d(&self.own);
        for c in &self.neighbours {
            let dc = d(c);
            if dc < best_d {
                best = *c;
                best_d = dc;
            }
        }
        best
    }
}

/// What the NWDAF task answers for one released UE.
#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    /// The UE.
    pub ue_id: i32,
    /// The predictor that produced it, as served: `last-served`, `ct`, `lsq`,
    /// `onnx`, or `fallback` (mode `onnx` with no verified model serving).
    pub predictor: &'static str,
    /// The heartbeat samples it had.
    pub samples: usize,
    /// The predicted position [`HORIZON_MS`] ahead, when there was one.
    pub predicted_position: Option<Vector3>,
    /// The predictor's confidence, when it reports one.
    pub confidence: Option<f32>,
    /// This gNB's cell, where the UE was served.
    pub serving: GridCell,
    /// The predicted neighbour, or `None` when the UE is predicted to stay.
    pub predicted: Option<GridCell>,
    /// The PLMN Identity on the wire.
    pub plmn: [u8; 3],
}

impl Recommendation {
    /// The IE content (TS 38.413 §9.3.1.100), in the §9.3.1.71 / §9.3.1.101
    /// order: this cell and gNB first as visited, then the predicted ones.
    pub fn paging_info(&self, time_stayed_s: Option<u16>) -> RecommendedPagingInfo {
        let mut cells = vec![RecommendedCell {
            plmn: self.plmn,
            nci: self.serving.nci,
            time_stayed_s: time_stayed_s.map(|t| t.min(4095)),
        }];
        let mut ran_nodes = vec![RecommendedRanNode {
            plmn: self.plmn,
            gnb_id: self.serving.gnb_id,
            gnb_id_len: self.serving.gnb_id_len,
        }];
        if let Some(p) = self.predicted {
            cells.push(RecommendedCell {
                plmn: self.plmn,
                nci: p.nci,
                time_stayed_s: None,
            });
            if (p.gnb_id, p.gnb_id_len) != (self.serving.gnb_id, self.serving.gnb_id_len) {
                ran_nodes.push(RecommendedRanNode {
                    plmn: self.plmn,
                    gnb_id: p.gnb_id,
                    gnb_id_len: p.gnb_id_len,
                });
            }
        }
        RecommendedPagingInfo { cells, ran_nodes }
    }

    /// The gNB IDs recommended, in order, the first visited.
    pub fn gnb_ids(&self) -> Vec<u32> {
        self.paging_info(None)
            .ran_nodes
            .iter()
            .map(|n| n.gnb_id)
            .collect()
    }

    /// `paging recommendation ...`, the NWDAF task's demo line.
    pub fn log_line(&self) -> String {
        let at = self
            .predicted_position
            .map(|p| format!(" at ({:.0}, {:.0}, {:.0})", p.x, p.y, p.z))
            .unwrap_or_default();
        let conf = self
            .confidence
            .map(|c| format!(" conf={c:.2}"))
            .unwrap_or_default();
        let predicted = match (self.predicted, self.predicted_position) {
            (Some(c), _) => format!("gNB {:#x} cell {:#x}{at}", c.gnb_id, c.nci),
            (None, Some(_)) => format!(
                "this gNB {:#x} cell {:#x}{at}",
                self.serving.gnb_id, self.serving.nci
            ),
            (None, None) => "none".to_string(),
        };
        format!(
            "paging recommendation ue={} predictor={} predicted={predicted}{conf} samples={} \
             horizon={}ms",
            self.ue_id, self.predictor, self.samples, HORIZON_MS
        )
    }
}

/// `nodes=[0x1(visited),0x2] cells=[0x10(visited, 42 s),0x20]`: the IE as the
/// Release Complete demo line shows it.
pub fn describe(info: &RecommendedPagingInfo) -> String {
    let nodes: Vec<String> = info
        .ran_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            if i == 0 {
                format!("{:#x}(visited)", n.gnb_id)
            } else {
                format!("{:#x}", n.gnb_id)
            }
        })
        .collect();
    let cells: Vec<String> = info
        .cells
        .iter()
        .enumerate()
        .map(|(i, c)| match (i, c.time_stayed_s) {
            (0, Some(t)) => format!("{:#x}(visited, {t} s)", c.nci),
            (0, None) => format!("{:#x}(visited)", c.nci),
            _ => format!("{:#x}", c.nci),
        })
        .collect();
    format!("nodes=[{}] cells=[{}]", nodes.join(","), cells.join(","))
}

/// The recommendation for `ue_id` from what `nwdaf` holds of it. See the
/// module docs. Never fails: with too little history the UE is predicted to
/// stay (this gNB only), and the line says `samples=`.
pub fn recommend(
    nwdaf: &NwdafManager,
    grid: &Grid,
    ue_id: i32,
    predictor: Predictor,
) -> Recommendation {
    let history = nwdaf.get_ue_history(ue_id);
    let samples = history.map_or(0, nextgsim_nwdaf::UeMeasurementHistory::len);
    let mut rec = Recommendation {
        ue_id,
        predictor: predictor.mode(),
        samples,
        predicted_position: None,
        confidence: None,
        serving: grid.own,
        predicted: None,
        plmn: grid.plmn,
    };
    let prediction: Option<(Vector3, Option<f32>)> = match (predictor, history) {
        (Predictor::LastServed, _) | (_, None) => None,
        (Predictor::ConstantTurn, Some(h)) => {
            let positions = h.position_sequence();
            let ts: Vec<u64> = h.all().iter().map(|m| m.timestamp_ms).collect();
            let now = ts.last().copied().unwrap_or(0);
            nextgsim_nwdaf::constant_turn::constant_turn_trajectory(
                &positions, &ts, HORIZON_MS, now,
            )
            .and_then(|p| Some((p.waypoints.last()?.0, Some(p.confidence))))
        }
        (Predictor::LeastSquares, Some(h)) if h.len() >= 2 => {
            let positions = h.position_sequence();
            let ts: Vec<u64> = h.all().iter().map(|m| m.timestamp_ms).collect();
            let now = ts.last().copied().unwrap_or(0);
            let p = nextgsim_nwdaf::predictor::least_squares_trajectory(
                &positions, &ts, HORIZON_MS, now,
            );
            p.waypoints.last().map(|w| (w.0, Some(p.confidence)))
        }
        (Predictor::LeastSquares, Some(_)) => None,
        (Predictor::Onnx, Some(_)) => {
            let before = nwdaf.serving_stats().onnx_inferences();
            let p = nwdaf.predict_trajectory(ue_id, HORIZON_MS);
            if nwdaf.serving_stats().onnx_inferences() == before {
                rec.predictor = "fallback";
            }
            p.and_then(|p| Some((p.waypoints.last()?.0, Some(p.confidence))))
        }
    };
    if let Some((position, confidence)) = prediction {
        if [position.x, position.y, position.z]
            .iter()
            .all(|v| v.is_finite())
        {
            rec.predicted_position = Some(position);
            rec.confidence = confidence;
            let cell = grid.nearest(&position);
            if cell.nci != grid.own.nci {
                rec.predicted = Some(cell);
            }
        }
    }
    rec
}

#[cfg(test)]
pub(crate) mod replay;

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_nwdaf::UeMeasurement;

    fn cell(nci: u64, x: f64, y: f64) -> GridCell {
        GridCell {
            nci,
            gnb_id: (nci >> 4) as u32,
            gnb_id_len: 32,
            position: Vector3::new(x, y, 0.0),
        }
    }

    fn grid() -> Grid {
        Grid {
            plmn: [0x99, 0xF9, 0x07],
            own: cell(0x10, 0.0, 0.0),
            neighbours: vec![cell(0x20, 120.0, 0.0), cell(0x40, 0.0, 120.0)],
        }
    }

    fn feed(nwdaf: &mut NwdafManager, ue_id: i32, track: impl Iterator<Item = (f64, f64)>) {
        for (i, (x, y)) in track.enumerate() {
            nwdaf.record_measurement(UeMeasurement {
                ue_id,
                rsrp: -60.0,
                rsrq: None,
                sinr: None,
                position: Vector3::new(x.round(), y.round(), 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: 1_000_000 + 1000 * i as u64,
            });
        }
    }

    /// #462: each mode maps to its predictor and back; `page-all` and unknown
    /// modes have none.
    #[test]
    fn each_predictor_mode_maps_to_its_predictor() {
        for mode in ["last-served", "ct", "lsq", "onnx"] {
            assert_eq!(Predictor::for_mode(mode).map(Predictor::mode), Some(mode));
        }
        assert_eq!(Predictor::for_mode("page-all"), None);
        assert_eq!(Predictor::for_mode("bogus"), None);
    }

    /// #462: the nearest cell wins; a tie stays with this gNB.
    #[test]
    fn the_grid_resolves_a_position_to_its_nearest_cell() {
        let g = grid();
        assert_eq!(g.nearest(&Vector3::new(100.0, 10.0, 0.0)).nci, 0x20);
        assert_eq!(g.nearest(&Vector3::new(5.0, 90.0, 0.0)).nci, 0x40);
        assert_eq!(g.nearest(&Vector3::new(60.0, 0.0, 0.0)).nci, 0x10, "tie");
        assert_eq!(g.nearest(&Vector3::new(-500.0, 0.0, 0.0)).nci, 0x10);
    }

    /// #462: a UE walking east at 5 m/s from x = 20 is predicted 20 s on at
    /// x ≈ 165, nearest gNB 2: the IE lists this cell (visited, with its
    /// time) then cell 0x20, and gNB 1 (visited) then gNB 2 (§9.3.1.71,
    /// §9.3.1.101). `last-served`, and `ct` for a UE standing still, list this
    /// gNB alone.
    #[test]
    fn a_moving_ue_is_recommended_its_next_gnb_after_this_one() {
        let g = grid();
        let mut nwdaf = NwdafManager::new(100);
        feed(&mut nwdaf, 7, (0..10).map(|i| (20.0 + 5.0 * i as f64, 0.0)));
        for predictor in [Predictor::ConstantTurn, Predictor::LeastSquares] {
            let rec = recommend(&nwdaf, &g, 7, predictor);
            assert_eq!(rec.predicted.map(|c| c.nci), Some(0x20), "{rec:?}");
            assert_eq!(rec.samples, 10);
            let info = rec.paging_info(Some(42));
            assert_eq!(
                describe(&info),
                "nodes=[0x1(visited),0x2] cells=[0x10(visited, 42 s),0x20]"
            );
            assert!(
                rec.log_line().starts_with(&format!(
                    "paging recommendation ue=7 predictor={} predicted=gNB 0x2 cell 0x20 at (",
                    predictor.mode()
                )),
                "{}",
                rec.log_line()
            );
        }
        let rec = recommend(&nwdaf, &g, 7, Predictor::LastServed);
        assert_eq!(rec.predicted, None);
        assert_eq!(rec.gnb_ids(), vec![1]);
        assert!(rec
            .log_line()
            .contains("predictor=last-served predicted=none"));

        feed(&mut nwdaf, 8, (0..10).map(|_| (10.0, 10.0)));
        let rec = recommend(&nwdaf, &g, 8, Predictor::ConstantTurn);
        assert_eq!(rec.predicted, None);
        assert_eq!(
            describe(&rec.paging_info(None)),
            "nodes=[0x1(visited)] cells=[0x10(visited)]"
        );
        assert!(
            rec.log_line()
                .contains("predicted=this gNB 0x1 cell 0x10 at (10, 10, 0)"),
            "{}",
            rec.log_line()
        );
    }

    /// #462: no history (a UE this gNB never heard) or a single sample gives
    /// no prediction and this gNB alone, never a guessed neighbour. Time stayed
    /// above 4095 s is sent as 4095 (§9.3.1.71).
    #[test]
    fn too_little_history_recommends_this_gnb_alone() {
        let g = grid();
        let mut nwdaf = NwdafManager::new(100);
        for predictor in [
            Predictor::ConstantTurn,
            Predictor::LeastSquares,
            Predictor::Onnx,
        ] {
            let rec = recommend(&nwdaf, &g, 99, predictor);
            assert_eq!((rec.samples, rec.predicted), (0, None), "{predictor:?}");
            assert!(rec.log_line().contains("predicted=none"));
        }
        feed(&mut nwdaf, 9, std::iter::once((100.0, 0.0)));
        let rec = recommend(&nwdaf, &g, 9, Predictor::ConstantTurn);
        assert_eq!((rec.samples, rec.predicted), (1, None));
        assert_eq!(
            rec.paging_info(Some(u16::MAX)).cells[0].time_stayed_s,
            Some(4095)
        );
    }

    /// #462: mode `onnx` with no verified model is served by the fallback and
    /// says so (`predictor=fallback`), rather than claiming the model.
    #[test]
    fn onnx_without_a_verified_model_is_labelled_fallback() {
        let g = grid();
        let mut nwdaf = NwdafManager::new(100);
        feed(&mut nwdaf, 7, (0..10).map(|i| (20.0 + 5.0 * i as f64, 0.0)));
        let rec = recommend(&nwdaf, &g, 7, Predictor::Onnx);
        assert_eq!(rec.predictor, "fallback");
        assert_eq!(rec.predicted.map(|c| c.nci), Some(0x20));
    }

    /// #462: the grid comes from the config: this gNB's NCI, gNB ID and
    /// position, and each `paging_grid` entry's gNB ID at its own length.
    #[test]
    fn the_grid_is_read_from_the_gnb_config() {
        let mut cfg: GnbConfig = serde_yaml::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../config/demo-ai-paging/gnb-west.yaml"
        )))
        .expect("parse");
        cfg.paging_grid
            .push(nextgsim_common::config::PagingGridEntry {
                nci: 0x0_0000_3C00,
                gnb_id_length: Some(26),
                position: nextgsim_common::config::SimPosition { x: 0, y: 99, z: 0 },
            });
        let g = Grid::from_config(&cfg);
        assert_eq!(g.plmn, cfg.plmn.encode());
        assert_eq!((g.own.nci, g.own.gnb_id, g.own.gnb_id_len), (0x10, 1, 32));
        assert_eq!(
            g.neighbours
                .iter()
                .map(|c| (c.nci, c.gnb_id, c.gnb_id_len))
                .collect::<Vec<_>>(),
            vec![(0x20, 2, 32), (0x3C00, 0x3C00 >> 10, 26)]
        );
        assert_eq!(g.neighbours[1].position, Vector3::new(0.0, 99.0, 0.0));
    }
}
