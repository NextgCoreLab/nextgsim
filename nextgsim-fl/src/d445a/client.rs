//! The UE side of demo D445A: one FL member's local training and its session
//! with the AF. `nr-ue` runs it over a TCP socket bound to the PDU session's TUN
//! (`nextgsim-ue/src/fl_participant/d445a.rs`); the AF's tests run it over
//! loopback.
//!
//! A session is: send `Hello { label }`, then answer every `GlobalModel { round }`
//! with one `Update { round, samples, loss, delta }`, until the AF closes. The
//! member's data never leaves it: the update carries the model delta and two
//! scalars, and the frame codec refuses anything that is not exactly the model's
//! size ([`crate::frame`]).
//!
//! In mode `poisoned-ue` member [`ATTACKER_MEMBER`] (UE 005), and only it, sends
//! `-POISON_SCALE ×` its honest delta. The mode is read at every round, so
//! `nr-cli <ue-005> demo d445a-fl-t8 set mode=…` applies from the next round.

use super::{label, FrameReader, ReadError, ATTACKER_MEMBER, MODE_POISONED_UE, POISON_SCALE};
use crate::frame::{self, Frame};
use crate::mlp::{self, TrainConfig};
use crate::sim::{self, RunConfig};
use crate::synthetic::{Dataset, Task};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};

/// What one member trains on.
#[derive(Debug, Clone, PartialEq)]
pub struct ClientParams {
    /// The member number, 1-based (`ue3` is 3).
    pub member: u32,
    /// The run seed (480 for the demo).
    pub seed: u64,
    /// How many members the task has (3 in `t8-exclusion`, 5 in `poisoned-ue`):
    /// the synthetic task draws one input region per member.
    pub clients: u32,
    /// Training rows of this member.
    pub samples: usize,
    /// Local training settings.
    pub train: TrainConfig,
    /// Revert arm `leak-raw` (`revert-hooks` builds only): append the member's
    /// first canary row to every update, bypassing the frame encoder. The AF must
    /// refuse the frame and close the connection.
    pub leak_raw: bool,
}

impl ClientParams {
    /// The demo's member `member` of a `clients`-member task on `seed`.
    pub fn demo(member: u32, seed: u64, clients: u32) -> Self {
        let run = RunConfig::demo(seed);
        Self {
            member,
            seed,
            clients,
            samples: run.samples_per_client,
            train: run.train,
            leak_raw: false,
        }
    }

    fn run_config(&self) -> RunConfig {
        RunConfig {
            seed: self.seed,
            clients: self.clients,
            samples_per_client: self.samples,
            train: self.train,
            ..RunConfig::demo(self.seed)
        }
    }

    /// Refuse a member outside the task, or a knob this build does not have.
    pub fn validate(&self) -> Result<(), String> {
        if self.member == 0 || self.member > self.clients {
            return Err(format!(
                "fl.member {} must be in 1..={} (fl.clients)",
                self.member, self.clients
            ));
        }
        if self.samples == 0 {
            return Err("fl.samples must be at least 1".to_string());
        }
        if self.leak_raw && !cfg!(feature = "revert-hooks") {
            return Err("fl.leak_raw is a revert arm and needs a revert-hooks build".to_string());
        }
        Ok(())
    }
}

/// One member's data and training.
#[derive(Debug, Clone)]
pub struct Trainer {
    params: ClientParams,
    data: Dataset,
}

/// What one round produced, before it is framed.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalUpdate {
    pub round: u32,
    pub samples: u32,
    /// Training-set MSE after the local epochs (honest model).
    pub loss: f32,
    pub delta: Vec<f32>,
    pub poisoned: bool,
}

impl Trainer {
    /// The member's data, drawn from the pinned task.
    pub fn new(params: ClientParams) -> Self {
        let data =
            Task::new(params.seed, params.clients).client_data(params.member - 1, params.samples);
        Self { params, data }
    }

    /// The member's parameters.
    pub fn params(&self) -> &ClientParams {
        &self.params
    }

    /// The label it says Hello with.
    pub fn label(&self) -> String {
        label(self.params.member)
    }

    /// Whether this member poisons in `mode`.
    pub fn poisons_in(&self, mode: Option<&str>) -> bool {
        self.params.member == ATTACKER_MEMBER && mode == Some(MODE_POISONED_UE)
    }

    /// Train from `global` for `round`; poisoned when `poison` is set. The delta
    /// is [`sim::local_delta`]'s, so a network run matches the in-process one.
    pub fn update(&self, round: u32, global: &[f32], poison: bool) -> LocalUpdate {
        let cfg = self.params.run_config();
        let honest = sim::local_delta(global, &self.data, &cfg, self.params.member - 1, round);
        let trained: Vec<f32> = global.iter().zip(&honest).map(|(g, d)| g + d).collect();
        let loss = mlp::mse(&trained, &self.data.xs, &self.data.ys);
        let delta = if poison {
            honest.iter().map(|d| -POISON_SCALE * d).collect()
        } else {
            honest
        };
        LocalUpdate {
            round,
            samples: self.data.len() as u32,
            loss,
            delta,
            poisoned: poison,
        }
    }

    /// The bytes sent for `u`.
    pub fn encode(&self, u: &LocalUpdate) -> Result<Vec<u8>, frame::FrameError> {
        if self.params.leak_raw {
            return Ok(self.leaky_frame(u));
        }
        frame::encode(&Frame::Update {
            round: u.round,
            samples: u.samples,
            loss: u.loss,
            delta: u.delta.clone(),
        })
    }

    /// Revert arm `leak-raw`: an Update whose vector is the delta with this
    /// member's first canary row appended, framed by hand because the encoder
    /// refuses it.
    fn leaky_frame(&self, u: &LocalUpdate) -> Vec<u8> {
        let row = self.data.canaries().next().unwrap_or_default();
        let floats: Vec<f32> = u.delta.iter().chain(row).copied().collect();
        let mut p = u.round.to_be_bytes().to_vec();
        p.extend_from_slice(&u.samples.to_be_bytes());
        p.extend_from_slice(&u.loss.to_le_bytes());
        p.extend_from_slice(&(floats.len() as u32).to_be_bytes());
        for x in floats {
            p.extend_from_slice(&x.to_le_bytes());
        }
        let mut out = frame::MAGIC.to_vec();
        out.push(3);
        out.extend_from_slice(&(p.len() as u32).to_be_bytes());
        out.extend_from_slice(&p);
        out
    }

    /// The canary rows of this member (for tests and the AF's scan).
    pub fn canaries(&self) -> Vec<Vec<f32>> {
        self.data.canaries().map(<[f32]>::to_vec).collect()
    }
}

/// What a client reports through `nr-cli … fl-status`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClientStatus {
    /// `idle`, `waiting-for-device`, `connecting`, `connected`, `backoff`.
    pub state: String,
    /// Sessions opened so far.
    pub sessions: u64,
    /// Updates sent.
    pub rounds_trained: u64,
    /// Of which poisoned.
    pub poisoned_rounds: u64,
    /// The last round answered.
    pub last_round: Option<u32>,
    /// Its local training loss.
    pub last_loss: Option<f32>,
    /// Why the last session ended.
    pub last_end: Option<String>,
}

/// How a session ended.
#[derive(Debug)]
pub enum SessionEnd {
    /// The AF closed the connection between frames.
    Closed,
    /// No GlobalModel within the idle limit.
    Idle,
    /// The transport failed or a frame was refused.
    Failed(String),
}

impl std::fmt::Display for SessionEnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => write!(f, "closed by the AF"),
            Self::Idle => write!(f, "no global model within the idle limit"),
            Self::Failed(why) => write!(f, "{why}"),
        }
    }
}

/// Run one session on `stream`: Hello, then one Update per GlobalModel.
///
/// `mode` is read at every round (the demo's runtime switch). `on_update` sees
/// every update after it is written; the caller logs and counts with it.
pub async fn run_session<S, M, U>(
    stream: S,
    trainer: &Trainer,
    idle: std::time::Duration,
    mode: M,
    mut on_update: U,
) -> SessionEnd
where
    S: AsyncRead + AsyncWrite + Unpin,
    M: Fn() -> Option<String>,
    U: FnMut(&LocalUpdate),
{
    let (mut rd, mut wr) = tokio::io::split(stream);
    let hello = match frame::encode(&Frame::Hello {
        label: trainer.label(),
    }) {
        Ok(b) => b,
        Err(e) => return SessionEnd::Failed(format!("hello: {e}")),
    };
    if let Err(e) = wr.write_all(&hello).await {
        return SessionEnd::Failed(format!("send hello: {e}"));
    }
    let mut reader = FrameReader::default();
    loop {
        let next = match tokio::time::timeout(idle, reader.next(&mut rd)).await {
            Err(_) => return SessionEnd::Idle,
            Ok(n) => n,
        };
        let (round, weights) = match next {
            Ok(None) => return SessionEnd::Closed,
            Ok(Some((Frame::GlobalModel { round, weights }, _))) => (round, weights),
            Ok(Some((other, _))) => {
                return SessionEnd::Failed(format!("unexpected frame from the AF: {other:?}"))
            }
            Err(ReadError::Io(e)) => return SessionEnd::Failed(format!("transport: {e}")),
            Err(e) => return SessionEnd::Failed(e.to_string()),
        };
        let poison = trainer.poisons_in(mode().as_deref());
        let update = trainer.update(round, &weights, poison);
        let bytes = match trainer.encode(&update) {
            Ok(b) => b,
            Err(e) => return SessionEnd::Failed(format!("encode update: {e}")),
        };
        if let Err(e) = wr.write_all(&bytes).await {
            return SessionEnd::Failed(format!("send update: {e}"));
        }
        on_update(&update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::contains_row;

    /// The client's update is the library's: the same delta as `sim::local_delta`
    /// for the same member, bit for bit, so a network run reproduces the
    /// in-process one.
    #[test]
    fn the_update_is_the_simulated_members_delta() {
        let t = Trainer::new(ClientParams::demo(2, 480, 3));
        let cfg = RunConfig::demo(480);
        let data = sim::Data::new(&cfg);
        let global = mlp::init(480);
        let want = sim::local_delta(&global, &data.clients[1], &cfg, 1, 4);
        let got = t.update(4, &global, false);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got.delta), bits(&want));
        assert_eq!(got.samples, 240);
        assert!(got.loss.is_finite());
    }

    /// Only UE 005 poisons, and only in `poisoned-ue`; the poisoned delta is
    /// ×10 sign-flipped.
    #[test]
    fn only_ue5_poisons_and_only_in_poisoned_ue() {
        let five = Trainer::new(ClientParams::demo(5, 480, 5));
        let three = Trainer::new(ClientParams::demo(3, 480, 5));
        assert!(five.poisons_in(Some(MODE_POISONED_UE)));
        assert!(!five.poisons_in(Some(super::super::MODE_T8_EXCLUSION)));
        assert!(!five.poisons_in(None));
        assert!(!three.poisons_in(Some(MODE_POISONED_UE)));
        let g = mlp::init(480);
        let honest = five.update(1, &g, false);
        let bad = five.update(1, &g, true);
        for (h, b) in honest.delta.iter().zip(&bad.delta) {
            assert_eq!(b.to_bits(), (-POISON_SCALE * h).to_bits());
        }
    }

    /// The honest frame carries no training row; the leak-raw revert frame does,
    /// and the decoder refuses it (count ≠ 261).
    #[test]
    fn the_update_frame_is_row_free_and_leak_raw_is_refused() {
        let mut p = ClientParams::demo(1, 480, 3);
        let t = Trainer::new(p.clone());
        let u = t.update(1, &mlp::init(480), false);
        let bytes = t.encode(&u).unwrap();
        for c in t.canaries() {
            assert!(!contains_row(&bytes, &c));
        }
        p.leak_raw = true;
        let leaky = Trainer::new(p);
        let bytes = leaky.encode(&u).unwrap();
        assert!(contains_row(&bytes, &leaky.canaries()[0]));
        assert!(matches!(
            frame::decode(&bytes),
            Err(frame::FrameError::Malformed(_))
        ));
    }

    #[test]
    fn params_outside_the_task_are_refused() {
        assert!(ClientParams::demo(3, 480, 3).validate().is_ok());
        assert!(ClientParams::demo(0, 480, 3).validate().is_err());
        assert!(ClientParams::demo(4, 480, 3).validate().is_err());
        let mut p = ClientParams::demo(1, 480, 3);
        p.leak_raw = true;
        assert_eq!(p.validate().is_ok(), cfg!(feature = "revert-hooks"));
    }
}
