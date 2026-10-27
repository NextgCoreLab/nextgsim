//! The command lines of the `lte-ue` control socket and the bearer commands'
//! answers (#365, #434).
//!
//! One connection carries one line and gets one answer line:
//! - `detach [switch-off] [TIMEOUT_SECS]`: `detached`, or `error: ...`;
//! - `bearer-modify <EBI> <GBR-kbps> [TIMEOUT_SECS]`: a UE-requested bearer
//!   resource modification (TS 24.301 §6.5.4) giving dedicated bearer `EBI`
//!   MBR = GBR = `GBR-kbps` both ways; `modified` once the network's MODIFY
//!   EPS BEARER CONTEXT REQUEST is accepted, or `error: ...`;
//! - `bearer-release <EBI> [TIMEOUT_SECS]`: one deleting every packet filter
//!   of `EBI` (ESM cause #36); `released` once its DEACTIVATE EPS BEARER
//!   CONTEXT REQUEST is accepted, or `error: ...`.

use std::sync::Mutex;
use std::time::Duration;

use nextgsim_eps_nas::{BearerCommand, BearerCommandRefusal};
use tokio::sync::{mpsc, oneshot};

use crate::nas::EpsNasStatus;
use crate::task::ControlCommand;

/// How long a command waits for the UE unless the line names another.
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// The longest wait a line may ask for: a day.
pub const MAX_TIMEOUT_SECS: u64 = 86_400;

/// A parsed control line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlRequest {
    /// `detach [switch-off] [TIMEOUT_SECS]`.
    Detach {
        /// Switch off: no DETACH ACCEPT follows.
        switch_off: bool,
        /// Seconds to wait.
        timeout: u64,
    },
    /// `bearer-modify <EBI> <GBR-kbps> [TIMEOUT_SECS]` or
    /// `bearer-release <EBI> [TIMEOUT_SECS]`.
    Bearer {
        /// The command.
        cmd: BearerCommand,
        /// Seconds to wait.
        timeout: u64,
    },
}

/// Parse one control line; the error is the answer line to send.
pub fn parse_line(line: &str) -> Result<ControlRequest, String> {
    let words: Vec<&str> = line.split_whitespace().collect();
    let timeout = |rest: &[&str]| -> Result<u64, String> {
        match rest {
            [] => Ok(DEFAULT_TIMEOUT_SECS),
            [t] => t
                .parse()
                .ok()
                .filter(|t| *t <= MAX_TIMEOUT_SECS)
                .ok_or_else(|| {
                    format!(
                        "error: timeout {t:?} is not a number of seconds up to {MAX_TIMEOUT_SECS}"
                    )
                }),
            _ => Err(format!("error: too many arguments in {:?}", line.trim())),
        }
    };
    let ebi = |w: &str| {
        w.parse::<u8>()
            .map_err(|_| format!("error: EBI {w:?} is not an EPS bearer identity"))
    };
    match words.as_slice() {
        ["detach", rest @ ..] => Ok(ControlRequest::Detach {
            switch_off: rest.contains(&"switch-off"),
            timeout: rest
                .iter()
                .find_map(|w| w.parse::<u64>().ok())
                .unwrap_or(DEFAULT_TIMEOUT_SECS)
                .min(MAX_TIMEOUT_SECS),
        }),
        ["bearer-modify", e, rate, rest @ ..] => {
            let gbr_kbps = rate
                .parse::<u32>()
                .map_err(|_| format!("error: GBR {rate:?} is not a rate in kbit/s"))?;
            Ok(ControlRequest::Bearer {
                cmd: BearerCommand::Modify {
                    ebi: ebi(e)?,
                    gbr_kbps,
                },
                timeout: timeout(rest)?,
            })
        }
        ["bearer-modify", ..] => {
            Err("error: usage: bearer-modify <EBI> <GBR-kbps> [TIMEOUT_SECS]".to_string())
        }
        ["bearer-release", e, rest @ ..] => Ok(ControlRequest::Bearer {
            cmd: BearerCommand::Release { ebi: ebi(e)? },
            timeout: timeout(rest)?,
        }),
        ["bearer-release", ..] => {
            Err("error: usage: bearer-release <EBI> [TIMEOUT_SECS]".to_string())
        }
        _ => Err(format!("error: unknown command {:?}", line.trim())),
    }
}

/// Run a bearer command against the UE task and wait for its answer line:
/// `modified` / `released`, or `error: ...` for a refusal, a reject, a T3481
/// abort, or no answer within `timeout` seconds.
///
/// The task is asked again, at most once a second, while the UE has no NAS
/// signalling connection for the request: its first ask starts the service
/// request that sets one up.
pub async fn run_bearer_command(
    status: &Mutex<EpsNasStatus>,
    commands: &mpsc::UnboundedSender<ControlCommand>,
    cmd: BearerCommand,
    timeout: u64,
) -> String {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout);
    let pti = loop {
        let (reply, answer) = oneshot::channel();
        if commands
            .send(ControlCommand::Bearer { cmd, reply })
            .is_err()
        {
            return "error: the UE is not running".to_string();
        }
        match answer.await {
            Ok(Ok(pti)) => break pti,
            Ok(Err(BearerCommandRefusal::NotConnected)) => {}
            Ok(Err(refusal)) => return format!("error: {refusal}"),
            Err(_) => return "error: the UE is not running".to_string(),
        }
        if tokio::time::Instant::now() >= deadline {
            return format!("error: no NAS signalling connection within {timeout} s");
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    };
    loop {
        let outcome = status.lock().ok().and_then(|s| s.bearer_outcome(pti));
        if let Some(outcome) = outcome {
            return match cmd.answer(outcome) {
                Ok(word) => word.to_string(),
                Err(e) => format!("error: {e}"),
            };
        }
        if tokio::time::Instant::now() >= deadline {
            return format!(
                "error: the network did not answer the BEARER RESOURCE MODIFICATION REQUEST \
                 (PTI {pti}) within {timeout} s"
            );
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_eps_nas::{BearerResourceOutcome, EmmState};
    use std::sync::Arc;

    #[test]
    fn bearer_lines_parse() {
        assert_eq!(
            parse_line("bearer-modify 6 1000\n"),
            Ok(ControlRequest::Bearer {
                cmd: BearerCommand::Modify {
                    ebi: 6,
                    gbr_kbps: 1000
                },
                timeout: DEFAULT_TIMEOUT_SECS
            })
        );
        assert_eq!(
            parse_line("bearer-release 7 30"),
            Ok(ControlRequest::Bearer {
                cmd: BearerCommand::Release { ebi: 7 },
                timeout: 30
            })
        );
        assert_eq!(
            parse_line("detach switch-off 5"),
            Ok(ControlRequest::Detach {
                switch_off: true,
                timeout: 5
            })
        );
        // A huge detach timeout is capped, not an Instant overflow
        assert_eq!(
            parse_line("detach 18446744073709551615"),
            Ok(ControlRequest::Detach {
                switch_off: false,
                timeout: MAX_TIMEOUT_SECS
            })
        );
        for bad in [
            "bearer-modify 6",
            "bearer-modify x 100",
            "bearer-modify 6 fast",
            "bearer-modify 6 100 30 40",
            "bearer-release",
            "bearer-release six",
            "bearer-release 6 soon",
            "bearer-release 6 18446744073709551615",
            "attach",
        ] {
            let e = parse_line(bad).unwrap_err();
            assert!(e.starts_with("error: "), "{bad}: {e}");
        }
    }

    fn status() -> Arc<Mutex<EpsNasStatus>> {
        Arc::new(Mutex::new(EpsNasStatus {
            state: EmmState::Registered,
            bearer: None,
            tau_attempts: 0,
            m_tmsi: None,
            bearer_outcomes: Vec::new(),
        }))
    }

    /// A UE task that answers the first ask "no connection", the second with
    /// PTI 7, and then has the network end procedure 7 in `outcome`.
    fn fake_task(
        status: Arc<Mutex<EpsNasStatus>>,
        outcome: BearerResourceOutcome,
    ) -> mpsc::UnboundedSender<ControlCommand> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut asks = 0;
            while let Some(c) = rx.recv().await {
                let ControlCommand::Bearer { reply, .. } = c else {
                    panic!("not a bearer command");
                };
                asks += 1;
                if asks == 1 {
                    let _ = reply.send(Err(BearerCommandRefusal::NotConnected));
                    continue;
                }
                let _ = reply.send(Ok(7));
                tokio::time::sleep(Duration::from_millis(300)).await;
                status.lock().unwrap().bearer_outcomes.push((7, outcome));
            }
        });
        tx
    }

    /// #434: through the control path, a modify answered by an accepted
    /// MODIFY is `modified` and a release ended by a DEACTIVATE is
    /// `released`, after asking again for the connection; a reject is an
    /// error line.
    #[tokio::test(start_paused = true)]
    async fn bearer_commands_answer_once_the_network_has() {
        let modify = BearerCommand::Modify {
            ebi: 6,
            gbr_kbps: 64,
        };
        let release = BearerCommand::Release { ebi: 6 };
        for (cmd, outcome, answer) in [
            (modify, BearerResourceOutcome::Modified, "modified"),
            (release, BearerResourceOutcome::Deactivated, "released"),
            (
                modify,
                BearerResourceOutcome::Rejected(30),
                "error: BEARER RESOURCE MODIFICATION REJECT, ESM cause #30",
            ),
        ] {
            let s = status();
            let tx = fake_task(s.clone(), outcome);
            assert_eq!(run_bearer_command(&s, &tx, cmd, 60).await, answer);
        }
    }

    /// #434: a refusal is answered at once; no answer from the network within
    /// the timeout is an error.
    #[tokio::test(start_paused = true)]
    async fn a_refused_or_unanswered_bearer_command_is_an_error() {
        let s = status();
        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut first = true;
            while let Some(ControlCommand::Bearer { reply, .. }) = rx.recv().await {
                let _ = reply.send(if std::mem::take(&mut first) {
                    Err(BearerCommandRefusal::DefaultBearer(5))
                } else {
                    Ok(9)
                });
            }
        });
        let answer = run_bearer_command(&s, &tx, BearerCommand::Release { ebi: 5 }, 60).await;
        assert!(
            answer.starts_with("error: ") && answer.contains("default bearer"),
            "{answer}"
        );
        let answer = run_bearer_command(&s, &tx, BearerCommand::Release { ebi: 6 }, 3).await;
        assert!(answer.contains("did not answer"), "{answer}");
    }
}
