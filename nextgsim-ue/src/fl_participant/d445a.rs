//! Demo D445A (nextgsim#480): the UE's FL client. **Non-normative.**
//!
//! When the demo is active and the UE config has an `fl:` block, the FL
//! Participant task starts one client per process (the first UE instance). It
//! connects to the AF (`nr-fl-server`) over a TCP socket bound to the PDU
//! session's TUN (`fl.device`, e.g. `uesimtun0`), so the traffic rides the
//! session through the UPF to the N6 side, and trains as member `fl.member` of
//! the pinned synthetic task ([`nextgsim_fl::d445a::client`]).
//!
//! The session is retried: until the TUN exists the bind or connect fails, and
//! after a deregistration and re-registration (or a restart) the client joins
//! again, which the AF logs as a readmission. Nothing is sent on the old
//! connection when the user plane is gone, as with a real handset.
//!
//! - `nr-cli <ue> fl-train`: start the client if `fl.autostart` is false (or
//!   report that it runs).
//! - `nr-cli <ue> fl-status`: member, state, sessions, rounds trained, the last
//!   round and loss, the mode.
//! - `nr-cli <ue-005> demo d445a-fl-t8 set mode=poisoned-ue`: UE 005 poisons
//!   from the next round (the mode is read at every round; audited).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use nextgsim_common::config::FlClientConfig;
use nextgsim_demo::demo_log;
use nextgsim_fl::d445a::client::{run_session, ClientParams, ClientStatus, Trainer};
use nextgsim_fl::d445a::KEY;
use nextgsim_fl::mlp::TrainConfig;
use tokio::net::{TcpSocket, TcpStream};

/// The client of this process.
struct Shared {
    config: FlClientConfig,
    trainer: Arc<Trainer>,
    status: ClientStatus,
    running: bool,
}

static CLIENT: Mutex<Option<Arc<Mutex<Shared>>>> = Mutex::new(None);

/// The trainer parameters of an `fl:` block.
pub fn params(config: &FlClientConfig) -> Result<ClientParams, String> {
    let p = ClientParams {
        member: config.member,
        seed: config.seed,
        clients: config.clients,
        samples: config.samples,
        train: TrainConfig::default(),
        leak_raw: config.leak_raw,
    };
    p.validate()?;
    Ok(p)
}

/// Install the client if demo D445A is active and `config` is present, and
/// connect it unless `autostart` is false. A UE without the demo is unchanged.
pub fn start(config: Option<&FlClientConfig>) {
    if nextgsim_demo::active_mode(KEY).is_none() {
        return;
    }
    let Some(config) = config.cloned() else {
        demo_log!(
            warn,
            KEY,
            "demo enabled but the UE config has no fl block: no FL client runs"
        );
        return;
    };
    let p = match params(&config) {
        Ok(p) => p,
        Err(e) => {
            demo_log!(error, KEY, "fl block refused: {e}; no FL client runs");
            return;
        }
    };
    let Ok(mut slot) = CLIENT.lock() else { return };
    if slot.is_some() {
        return; // another UE instance of this process runs the client
    }
    let autostart = config.autostart;
    let shared = Arc::new(Mutex::new(Shared {
        config,
        trainer: Arc::new(Trainer::new(p)),
        status: ClientStatus {
            state: "idle".to_string(),
            ..ClientStatus::default()
        },
        running: false,
    }));
    *slot = Some(shared.clone());
    drop(slot);
    if autostart {
        spawn(&shared);
    }
}

/// Start the session loop unless it runs; returns whether it was started.
fn spawn(shared: &Arc<Mutex<Shared>>) -> bool {
    let Ok(mut s) = shared.lock() else {
        return false;
    };
    if s.running {
        return false;
    }
    s.running = true;
    demo_log!(
        info,
        KEY,
        "FL client {} -> {} via {} (seed {}, task of {} members, {} samples)",
        s.trainer.label(),
        s.config.server,
        s.config.device.as_deref().unwrap_or("the routing table"),
        s.config.seed,
        s.config.clients,
        s.config.samples
    );
    drop(s);
    tokio::spawn(run(shared.clone()));
    true
}

fn set_state(shared: &Mutex<Shared>, state: &str) {
    if let Ok(mut s) = shared.lock() {
        s.status.state = state.to_string();
    }
}

async fn connect(config: &FlClientConfig) -> std::io::Result<TcpStream> {
    let socket = TcpSocket::new_v4()?;
    if let Some(dev) = &config.device {
        socket.bind_device(Some(dev.as_bytes()))?;
    }
    tokio::time::timeout(Duration::from_secs(5), socket.connect(config.server))
        .await
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "connect timed out"))?
}

async fn run(shared: Arc<Mutex<Shared>>) {
    let Some((config, trainer)) = shared
        .lock()
        .ok()
        .map(|s| (s.config.clone(), s.trainer.clone()))
    else {
        return;
    };
    let idle = Duration::from_secs(config.idle_timeout_s.max(1));
    let mut backoff = Duration::from_secs(1);
    let mut warned = false;
    loop {
        set_state(&shared, "connecting");
        let stream = match connect(&config).await {
            Ok(s) => s,
            Err(e) => {
                if !warned {
                    demo_log!(
                        info,
                        KEY,
                        "waiting for {} to reach {}: {e}",
                        config.device.as_deref().unwrap_or("a route"),
                        config.server
                    );
                    warned = true;
                }
                set_state(&shared, "waiting-for-device");
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(10));
                continue;
            }
        };
        warned = false;
        backoff = Duration::from_secs(1);
        if let Ok(mut s) = shared.lock() {
            s.status.sessions += 1;
            s.status.state = "connected".to_string();
        }
        demo_log!(
            info,
            KEY,
            "connected to {} as {}",
            config.server,
            trainer.label()
        );
        let end = run_session(
            stream,
            &trainer,
            idle,
            || nextgsim_demo::active_mode(KEY),
            |u| {
                demo_log!(
                    info,
                    KEY,
                    "round={} update sent: samples={} loss={:.6} poisoned={}",
                    u.round,
                    u.samples,
                    u.loss,
                    u.poisoned
                );
                if let Ok(mut s) = shared.lock() {
                    s.status.rounds_trained += 1;
                    s.status.poisoned_rounds += u64::from(u.poisoned);
                    s.status.last_round = Some(u.round);
                    s.status.last_loss = Some(u.loss);
                }
            },
        )
        .await;
        demo_log!(info, KEY, "session ended: {end}; reconnecting");
        if let Ok(mut s) = shared.lock() {
            s.status.last_end = Some(end.to_string());
            s.status.state = "backoff".to_string();
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// Answer `fl-train` and `fl-status`; `None` for any other command.
pub fn route_cli(command: &str) -> Option<String> {
    let word = command.split_whitespace().next()?.to_ascii_lowercase();
    if word != "fl-train" && word != "fl-status" {
        return None;
    }
    let p = nextgsim_demo::prefix(KEY);
    let shared = CLIENT.lock().ok().and_then(|c| c.clone());
    let Some(shared) = shared else {
        return Some(format!(
            "{p} no FL client in this UE (needs demo {KEY} enabled, federated_learning_enabled \
             and an fl block)"
        ));
    };
    if word == "fl-train" {
        return Some(if spawn(&shared) {
            format!("{p} FL client started")
        } else {
            format!("{p} FL client already running")
        });
    }
    let s = shared.lock().ok()?;
    let st = &s.status;
    Some(format!(
        "{p} member={} server={} device={} mode={} state={} sessions={} rounds_trained={} \
         poisoned_rounds={} last_round={} last_loss={} last_end={}",
        s.trainer.label(),
        s.config.server,
        s.config.device.as_deref().unwrap_or("-"),
        nextgsim_demo::active_mode(KEY).as_deref().unwrap_or("-"),
        st.state,
        st.sessions,
        st.rounds_trained,
        st.poisoned_rounds,
        st.last_round
            .map_or_else(|| "-".to_string(), |r| r.to_string()),
        st.last_loss
            .map_or_else(|| "-".to_string(), |l| format!("{l:.6}")),
        st.last_end.as_deref().unwrap_or("-")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(extra: &str) -> FlClientConfig {
        serde_yaml::from_str(&format!("server: 10.46.0.10:7480\nmember: 3\n{extra}")).unwrap()
    }

    /// The `fl:` block's defaults are the demo's: seed 480, 3 members, 240
    /// samples, autostart, and no leak.
    #[test]
    fn the_fl_block_defaults_to_the_demo_shape() {
        let c = config("device: uesimtun0\n");
        assert_eq!(c.device.as_deref(), Some("uesimtun0"));
        assert_eq!((c.seed, c.clients, c.samples), (480, 3, 240));
        assert!(c.autostart && !c.leak_raw);
        let p = params(&c).unwrap();
        assert_eq!((p.member, p.clients), (3, 3));
        assert!(
            serde_yaml::from_str::<FlClientConfig>("server: 1.2.3.4:1\nmember: 1\nbogus: 1\n")
                .is_err()
        );
    }

    /// A member outside the task, or `leak_raw` without revert-hooks, is
    /// refused.
    #[test]
    fn an_fl_block_that_cannot_run_is_refused() {
        assert!(params(&config("clients: 2\n")).is_err());
        let mut c = config("");
        c.member = 0;
        assert!(params(&c).is_err());
        assert_eq!(
            params(&config("leak_raw: true\n")).is_ok(),
            cfg!(feature = "revert-hooks")
        );
    }

    /// Without the client installed, `fl-train` / `fl-status` say why; other
    /// commands are not this module's.
    #[test]
    fn the_cli_words_are_routed_here_and_nothing_else() {
        assert_eq!(route_cli("status"), None);
        assert_eq!(route_cli("deregister normal"), None);
        for w in ["fl-status", "FL-TRAIN"] {
            let out = route_cli(w).unwrap();
            assert!(out.starts_with("[demo:d445a-fl-t8] "), "{out}");
        }
    }
}
