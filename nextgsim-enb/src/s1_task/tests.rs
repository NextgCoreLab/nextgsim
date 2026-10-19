use super::*;
use crate::s1ap::tests::{cfg, failure, response, splice_unknown_ie};
use nextgsim_s1ap::codec::{
    Criticality, InitiatingMessage, InitiatingMessageValue, TimeToWait, S1AP_PDU,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// What the scripted MME does when the eNB next receives.
#[derive(Debug, Clone)]
enum Step {
    /// Deliver a PDU with this stream and PPID.
    Deliver(u16, u32, Vec<u8>),
    /// Close the association.
    Close,
    /// Say nothing (the receive pends until the eNB's timer fires).
    Silent,
}

#[derive(Default)]
struct Log {
    sent: Vec<(u16, u32, Vec<u8>)>,
}

struct Mock {
    script: VecDeque<Step>,
    log: Arc<Mutex<Log>>,
}

impl S1Transport for Mock {
    async fn send(&mut self, stream: u16, data: &[u8], ppid: u32) -> std::io::Result<()> {
        self.log
            .lock()
            .unwrap()
            .sent
            .push((stream, ppid, data.to_vec()));
        Ok(())
    }
    async fn recv(&mut self) -> std::io::Result<Option<S1Message>> {
        match self.script.pop_front() {
            Some(Step::Deliver(stream, ppid, data)) => Ok(Some(S1Message { stream, ppid, data })),
            Some(Step::Close) | None => Ok(None),
            Some(Step::Silent) => std::future::pending().await,
        }
    }
}

fn mock(steps: Vec<Step>) -> (Mock, Arc<Mutex<Log>>) {
    let log = Arc::new(Mutex::new(Log::default()));
    (
        Mock {
            script: steps.into(),
            log: Arc::clone(&log),
        },
        log,
    )
}

fn enc(pdu: &S1AP_PDU) -> Vec<u8> {
    codec::encode_s1ap_pdu(pdu).unwrap()
}

fn s1(pdu: &S1AP_PDU) -> Step {
    Step::Deliver(0, S1AP_PPID, enc(pdu))
}

fn is_setup_request(bytes: &[u8]) -> bool {
    matches!(
        codec::decode_s1ap_pdu(bytes),
        Ok(S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_S1Setup(_),
            ..
        }))
    )
}

fn is_error_indication(bytes: &[u8]) -> bool {
    matches!(
        codec::decode_s1ap_pdu(bytes),
        Ok(S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_ErrorIndication(_),
            ..
        }))
    )
}

#[tokio::test]
async fn request_goes_on_the_non_ue_stream_with_the_s1ap_ppid() {
    let (mut t, log) = mock(vec![s1(&response(true, true))]);
    let r = run_s1_setup_attempt(&cfg(), &mut t).await;
    assert!(matches!(r, AttemptResult::Operational(_)), "{r:?}");
    let sent = &log.lock().unwrap().sent;
    assert_eq!(sent.len(), 1);
    // Literal values, not the constants: stream 0 and PPID 18 (TS 36.412 §7).
    assert_eq!((sent[0].0, sent[0].1), (0, 18));
    assert!(is_setup_request(&sent[0].2));
}

#[tokio::test]
async fn failure_with_time_to_wait_retries_after_it() {
    let (mut t, _) = mock(vec![s1(&failure(Some(TimeToWait::V20S)))]);
    assert_eq!(
        run_s1_setup_attempt(&cfg(), &mut t).await,
        AttemptResult::RetryAfter(Duration::from_secs(20))
    );
}

#[tokio::test]
async fn failure_without_time_to_wait_uses_the_configured_retry() {
    let mut c = cfg();
    c.s1_setup_retry_secs = 7;
    let (mut t, _) = mock(vec![s1(&failure(None))]);
    assert_eq!(
        run_s1_setup_attempt(&c, &mut t).await,
        AttemptResult::RetryAfter(Duration::from_secs(7))
    );
}

#[tokio::test(start_paused = true)]
async fn a_silent_mme_times_the_attempt_out() {
    let (mut t, _) = mock(vec![Step::Silent]);
    let start = tokio::time::Instant::now();
    let r = run_s1_setup_attempt(&cfg(), &mut t).await;
    assert_eq!(r, AttemptResult::RetryAfter(Duration::from_secs(10)));
    assert_eq!(start.elapsed(), Duration::from_millis(5000));
}

#[tokio::test]
async fn a_closed_association_asks_for_a_reconnect() {
    let (mut t, _) = mock(vec![Step::Close]);
    assert_eq!(
        run_s1_setup_attempt(&cfg(), &mut t).await,
        AttemptResult::Reconnect(Duration::from_secs(10))
    );
}

#[tokio::test]
async fn a_wrong_ppid_is_discarded_and_the_real_response_still_counts() {
    // TS 36.412 §7: S1AP is PPID 18. A PDU on NGAP's 60 is not S1AP.
    let (mut t, _) = mock(vec![
        Step::Deliver(0, 60, enc(&failure(Some(TimeToWait::V60S)))),
        s1(&response(true, true)),
    ]);
    assert!(matches!(
        run_s1_setup_attempt(&cfg(), &mut t).await,
        AttemptResult::Operational(_)
    ));
}

#[tokio::test]
async fn a_notify_ie_in_the_response_sends_an_error_indication_and_completes() {
    let bytes = splice_unknown_ie(&enc(&response(true, true)), 9999, Criticality::NOTIFY, &[0]);
    let (mut t, log) = mock(vec![Step::Deliver(0, S1AP_PPID, bytes)]);
    assert!(matches!(
        run_s1_setup_attempt(&cfg(), &mut t).await,
        AttemptResult::Operational(_)
    ));
    let sent = &log.lock().unwrap().sent;
    assert_eq!(sent.len(), 2, "request, then the Error Indication");
    assert_eq!((sent[1].0, sent[1].1), (0, 18));
    assert!(is_error_indication(&sent[1].2));
}

#[tokio::test]
async fn a_reject_ie_in_the_response_is_local_error_handling_only() {
    let bytes = splice_unknown_ie(&enc(&response(true, true)), 9999, Criticality::REJECT, &[0]);
    let (mut t, log) = mock(vec![Step::Deliver(0, S1AP_PPID, bytes)]);
    assert_eq!(
        run_s1_setup_attempt(&cfg(), &mut t).await,
        AttemptResult::RetryAfter(Duration::from_secs(10))
    );
    assert_eq!(
        log.lock().unwrap().sent.len(),
        1,
        "nothing but the request goes to the MME"
    );
}

#[tokio::test]
async fn garbage_is_answered_and_the_attempt_keeps_waiting() {
    // §10.2: a transfer syntax error is reported; the procedure itself is still
    // outstanding, so the next PDU can complete it.
    let truncated = {
        let b = enc(&response(true, true));
        b[..b.len() - 3].to_vec()
    };
    let (mut t, log) = mock(vec![
        Step::Deliver(0, S1AP_PPID, truncated),
        s1(&response(true, true)),
    ]);
    assert!(matches!(
        run_s1_setup_attempt(&cfg(), &mut t).await,
        AttemptResult::Operational(_)
    ));
    let sent = &log.lock().unwrap().sent;
    assert_eq!(sent.len(), 2);
    assert!(is_error_indication(&sent[1].2));
}

/// A connector that fails `fail_first` times, then hands out scripted
/// associations in order.
struct MockConnector {
    fail_first: usize,
    associations: VecDeque<Vec<Step>>,
    logs: Vec<Arc<Mutex<Log>>>,
    connects: usize,
}

impl S1Connector for MockConnector {
    type Transport = Mock;
    async fn connect(&mut self, _l: SocketAddr, _r: SocketAddr) -> std::io::Result<Mock> {
        self.connects += 1;
        if self.fail_first > 0 {
            self.fail_first -= 1;
            return Err(std::io::Error::other("connection refused"));
        }
        let steps = self.associations.pop_front().expect("script exhausted");
        let (m, log) = mock(steps);
        self.logs.push(log);
        Ok(m)
    }
}

#[tokio::test(start_paused = true)]
async fn establish_waits_out_time_to_wait_then_completes() {
    let mut c = MockConnector {
        fail_first: 1,
        associations: vec![
            // First association: closed before any answer.
            vec![Step::Close],
            // Second: failure with Time To Wait 5 s, then success on retry.
            vec![
                s1(&failure(Some(TimeToWait::V5S))),
                s1(&response(true, true)),
            ],
        ]
        .into(),
        logs: vec![],
        connects: 0,
    };
    let start = tokio::time::Instant::now();
    let (mme, _t) = establish_s1(&cfg(), &mut c, "127.0.0.1:36412".parse().unwrap()).await;
    assert_eq!(mme.mme_name.as_deref(), Some("nextgcore-mme0"));
    assert_eq!(
        c.connects, 3,
        "refused, closed, then the one that completes"
    );
    // 10 s after the refused connect, 10 s after the closed association,
    // 5 s Time To Wait: §8.7.3.3's "at least" is honoured exactly.
    assert_eq!(start.elapsed(), Duration::from_secs(25));
    let second = c.logs[1].lock().unwrap();
    assert_eq!(
        second.sent.len(),
        2,
        "request, then the re-initiated request"
    );
    assert!(second.sent.iter().all(|s| is_setup_request(&s.2)));
}
