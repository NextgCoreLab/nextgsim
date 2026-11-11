//! Network Knowledge Exposure Function (NKEF) Task for gNB

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info};

use nextgsim_nkef::live::{rfc3339_ms, Consumer, FactKind, LiveStore};
use nextgsim_nkef::{Entity, EntityType, KnowledgeGraph};

use super::live::{mode, now_ms, tees_on};
use crate::demo::D447A_KEY;
use crate::tasks::{GnbTaskBase, NkefMessage, Task, TaskMessage};

/// NKEF Task for gNB
pub struct NkefTask {
    task_base: GnbTaskBase,
    knowledge_graph: KnowledgeGraph,
    entity_count: HashMap<String, usize>,
    /// The live facts of this run (nextgsim#488), fed only while demo
    /// d447a-nkef-live is on.
    live: LiveStore,
    #[cfg(feature = "nkef-sba")]
    sba: Option<super::sba::Sba>,
}

impl NkefTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        let mut live = LiveStore::new(now_ms());
        live.set_serving_nci(task_base.config.nci);
        #[cfg(feature = "nkef-sba")]
        let sba = super::sba::Sba::new(&task_base.config.nkef_live);
        Self {
            task_base,
            knowledge_graph: KnowledgeGraph::new(),
            entity_count: HashMap::new(),
            live,
            #[cfg(feature = "nkef-sba")]
            sba,
        }
    }

    /// A live fact from a tee or the SBA client. Ignored unless the tees are on,
    /// so a runtime switch to `baseline` stops the store from growing.
    fn handle_live_fact(&mut self, fact: nextgsim_nkef::live::Fact) {
        if tees_on() {
            self.live.upsert(fact);
        }
    }

    /// One NRF poll: record what it returned, expire what it no longer does.
    fn handle_nrf_poll(&mut self, facts: Vec<nextgsim_nkef::live::Fact>, at_ms: u64) {
        if !tees_on() {
            return;
        }
        let present: std::collections::BTreeSet<String> =
            facts.iter().map(|f| f.subject.clone()).collect();
        for f in facts {
            self.live.upsert(f);
        }
        for subject in self
            .live
            .expire_missing(FactKind::NfInstance, "nrf", &present, at_ms)
        {
            nextgsim_demo::demo_log!(
                warn,
                D447A_KEY,
                "NKEF fact expired subject={subject} producer=nrf expired_at={}",
                rfc3339_ms(at_ms)
            );
        }
    }

    /// `nkef-query`: one question, one bundle.
    ///
    /// With the demo on, the AMF's reports are refreshed first (subscribe with
    /// immediateFlag, read, unsubscribe) and the bundle answers from the live
    /// store. In `baseline`, or with the demo off, it answers from an empty
    /// store: today's NKEF returns nothing live.
    async fn handle_query(&mut self, question: &str, consumer: Consumer) -> Result<String, String> {
        let budget = self.task_base.config.nkef_live.budget_tokens;
        let bundle = if tees_on() {
            #[cfg(feature = "nkef-sba")]
            if let Some(sba) = &self.sba {
                for f in sba.amf_reports().await {
                    self.live.upsert(f);
                }
            }
            self.live.bundle(question, consumer, budget, now_ms())
        } else {
            LiveStore::new(self.live.started_at_ms()).bundle(question, consumer, budget, now_ms())
        };
        if bundle.text.is_empty() {
            return Err(format!(
                "budget_tokens={budget} cannot hold a bundle header"
            ));
        }
        if mode().is_some() {
            for line in bundle.text.lines() {
                nextgsim_demo::demo_log!(info, D447A_KEY, "{line}");
            }
        } else {
            debug!("NKEF: {}", bundle.text.lines().next().unwrap_or_default());
        }
        Ok(bundle.text)
    }

    /// `nkef-status`.
    fn status_line(&self) -> String {
        let domains: Vec<&str> = self
            .live
            .live_domains()
            .iter()
            .map(|d| d.as_str())
            .collect();
        #[cfg(feature = "nkef-sba")]
        let sba = if self.sba.is_some() { "on" } else { "off" };
        #[cfg(not(feature = "nkef-sba"))]
        let sba = "off";
        format!(
            "NKEF status mode={} started_at={} facts={} live_domains={} sba={sba}",
            mode().unwrap_or_else(|| "off".to_string()),
            rfc3339_ms(self.live.started_at_ms()),
            self.live.facts().count(),
            if domains.is_empty() {
                "none".to_string()
            } else {
                domains.join(",")
            },
        )
    }

    fn handle_update_knowledge(
        &mut self,
        entity_type: String,
        entity_id: String,
        properties: Vec<(String, String)>,
    ) {
        debug!(
            "NKEF: Updating knowledge for {} '{}' with {} properties",
            entity_type,
            entity_id,
            properties.len()
        );

        let etype = match entity_type.as_str() {
            "cell" => EntityType::Cell,
            "ue" => EntityType::Ue,
            "gnb" => EntityType::Gnb,
            "slice" => EntityType::Slice,
            "amf" => EntityType::Amf,
            _ => EntityType::Service,
        };

        let mut entity = Entity::new(entity_id.clone(), etype);
        for (k, v) in &properties {
            entity = entity.with_property(k, v);
        }
        self.knowledge_graph.add_entity(entity);

        *self.entity_count.entry(entity_type.clone()).or_insert(0) += 1;

        info!(
            "NKEF: Updated {} '{}' (total {} entities of this type)",
            entity_type,
            entity_id,
            self.entity_count.get(&entity_type).unwrap_or(&0)
        );
    }

    fn handle_semantic_query(&mut self, query: String, max_results: u32) {
        debug!(
            "NKEF: Semantic query '{}' (max_results={})",
            query, max_results
        );

        let results = self.knowledge_graph.search(&query, max_results as usize);

        info!("NKEF: Query '{}' returned {} results", query, results.len());
    }

    fn handle_retrieve_context(&mut self, prompt: String, max_tokens: u32) {
        debug!(
            "NKEF: RAG context retrieval for prompt '{}' (max_tokens={})",
            prompt, max_tokens
        );

        let context = self.knowledge_graph.generate_context(&prompt, max_tokens);

        info!(
            "NKEF: Retrieved context with {} sources for prompt",
            context.sources.len()
        );
    }
}

#[async_trait::async_trait]
impl Task for NkefTask {
    type Message = NkefMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!(
            "NKEF task started at={}",
            rfc3339_ms(self.live.started_at_ms())
        );
        #[cfg(feature = "nkef-sba")]
        if let (Some(sba), Some(sixg)) = (&self.sba, &self.task_base.sixg) {
            sba.spawn_nrf_poller(sixg.nkef_tx.clone());
        }

        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => match msg {
                    NkefMessage::UpdateKnowledge {
                        entity_type,
                        entity_id,
                        properties,
                    } => {
                        self.handle_update_knowledge(entity_type, entity_id, properties);
                    }
                    NkefMessage::SemanticQuery { query, max_results } => {
                        self.handle_semantic_query(query, max_results);
                    }
                    NkefMessage::RetrieveContext { prompt, max_tokens } => {
                        self.handle_retrieve_context(prompt, max_tokens);
                    }
                    NkefMessage::LiveFact(fact) => self.handle_live_fact(fact),
                    NkefMessage::NrfPoll { facts, at_ms } => self.handle_nrf_poll(facts, at_ms),
                    NkefMessage::Query {
                        question,
                        consumer,
                        reply,
                    } => {
                        let answer = self.handle_query(&question, consumer).await;
                        let _ = reply.send(answer);
                    }
                    NkefMessage::Status { reply } => {
                        let _ = reply.send(self.status_line());
                    }
                },
                Some(TaskMessage::Shutdown) => {
                    info!("NKEF task received shutdown signal");
                    break;
                }
                None => {
                    info!("NKEF task channel closed");
                    break;
                }
            }
        }

        let total_entities: usize = self.entity_count.values().sum();
        info!(
            "NKEF task stopped, {} total entities in knowledge graph",
            total_entities
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{GnbTaskBase, DEFAULT_CHANNEL_CAPACITY};
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(1, 1, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: "127.0.0.1".parse().unwrap(),
            ngap_ip: "127.0.0.1".parse().unwrap(),
            gtp_ip: "127.0.0.1".parse().unwrap(),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_nkef_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = NkefTask::new(task_base);
        assert_eq!(task.entity_count.len(), 0);
    }

    #[tokio::test]
    async fn test_nkef_task_update_knowledge() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NkefTask::new(task_base);
        let properties = vec![
            ("name".to_string(), "cell-1".to_string()),
            ("load".to_string(), "0.5".to_string()),
        ];
        task.handle_update_knowledge("cell".to_string(), "1".to_string(), properties);
        assert_eq!(*task.entity_count.get("cell").unwrap(), 1);
    }

    #[tokio::test]
    async fn test_nkef_task_semantic_query() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NkefTask::new(task_base);
        task.handle_semantic_query("high load cells".to_string(), 10);
    }

    /// nextgsim#488, end to end inside the gNB: the real tee functions feed the
    /// running NKEF task over its channel, `nkef-query` routed through the App
    /// task's `route_nkef_cli` gets one bundle back on a oneshot (an App task
    /// whose base lacks the 6G handles answers "NKEF task not running"), the
    /// external consumer gets congestion only, and a runtime switch to `baseline` answers with nothing live (a bogus mode is refused
    /// and leaves `on` in place).
    ///
    /// REVERT-VERIFIED: with `offer_cell_load` / `offer_ng_setup` returning
    /// early the header loses `ran` / `n2`; with the external projection
    /// replaced by the internal bundle the leak assertions fail; with
    /// `route_nkef_cli` seeing no 6G handles (the `sixg: None` the
    /// 447-spawn-order arm gives the App task), the first ask fails. That arm
    /// itself is in `GnbWiring::spawn`, which this test does not run: it stays a
    /// live-run arm, and the `sixg: None` answer here is what that arm produces.
    #[cfg(feature = "nkef-sba")]
    #[tokio::test]
    async fn the_tees_feed_one_bundle_per_question() {
        use crate::demo::D447A_KEY;
        use crate::nkef::live;
        use nextgsim_ngap::procedures::{Guami, NgSetupResponseData, ServedGuamiItem};
        use nextgsim_nkef::live::Consumer;

        let _lock = crate::demo::D447A_TEST_LOCK.lock().await;
        let rt = demo_on();

        let config = GnbConfig {
            nkef_enabled: true,
            ..test_config()
        };
        let mut wiring = crate::wiring::GnbWiring::new(config, DEFAULT_CHANNEL_CAPACITY);
        let sixg = wiring.sixg.take().expect("6G receivers");
        let base = wiring.task_base();
        let mut task = NkefTask::new(base.clone());
        let handle = tokio::spawn(async move { task.run(sixg.nkef_rx).await });

        live::offer_cell_load(&base, 1, Some(0.001));
        live::offer_ue_radio(&base, 1, -80);
        live::offer_ng_setup(
            &base,
            0,
            &NgSetupResponseData {
                amf_name: "nextgcore-amf".to_string(),
                served_guami_list: vec![ServedGuamiItem {
                    guami: Guami {
                        plmn_identity: Plmn::new(1, 1, false).encode(),
                        amf_region_id: 2,
                        amf_set_id: 0x155,
                        amf_pointer: 0x2a,
                    },
                    backup_amf_name: None,
                }],
                relative_amf_capacity: 255,
                plmn_support_list: vec![],
            },
        );
        live::offer_ue_n2(&base, 1, 0, 7);

        let nkef = base.sixg.as_ref().expect("handles").nkef_tx.clone();
        // Every question goes the way `nr-cli` sends it: through the App task's
        // own `route_nkef_cli`, on the App task's clone of the base (review m5).
        let app = crate::app::AppTask::new_without_cli(base.clone());
        let ask = |consumer| {
            let command = match consumer {
                Consumer::Internal => {
                    "nkef-query is my route congested and which core functions serve me?"
                }
                Consumer::External => {
                    "nkef-query --as external is my route congested and which core functions serve me?"
                }
            };
            let app = &app;
            async move {
                app.route_nkef_cli(command)
                    .await
                    .expect("an nkef command")
                    .expect("a bundle")
            }
        };
        // An App task whose base holds no 6G handles (the 447-spawn-order arm)
        // gets no bundle at all.
        let mut bare = base.clone();
        bare.sixg = None;
        let refused = crate::app::AppTask::new_without_cli(bare)
            .route_nkef_cli("nkef-query is my route congested?")
            .await;
        assert!(
            matches!(&refused, Some(Err(e)) if e.starts_with("NKEF task not running")),
            "{refused:?}"
        );
        let internal = ask(Consumer::Internal).await;
        let header = internal.lines().next().unwrap();
        assert!(
            header.starts_with("NKEF bundle consumer=internal domains=ran,n2 ")
                && header.contains(" path=ue>cell>gnb>amf "),
            "{internal}"
        );
        assert!(internal.contains("guami=001-01-02556a"), "{internal}");
        let external = ask(Consumer::External).await;
        assert!(
            external.starts_with("NKEF bundle consumer=external areas=1")
                && external.contains("area-1 congestion=Low"),
            "{external}"
        );
        for leak in [
            "imsi-", "ncgi", "nrCellId", "tac", "nci", "ngap_id", "172.23.", "guami",
        ] {
            assert!(!external.contains(leak), "{leak} leaked: {external}");
        }

        assert!(rt.set_mode(D447A_KEY, "bogus", "test").is_err());
        assert_eq!(rt.mode(D447A_KEY).as_deref(), Some("on"));
        rt.set_mode(D447A_KEY, "baseline", "test")
            .expect("baseline");
        let baseline = ask(Consumer::Internal).await;
        rt.set_mode(D447A_KEY, "on", "test").expect("back on");
        assert!(
            baseline.starts_with("NKEF bundle consumer=internal domains=none "),
            "{baseline}"
        );

        nkef.shutdown().await.ok();
        handle.await.unwrap();
    }

    /// Install d447a-nkef-live in mode `on` (once per process) and return it.
    #[cfg(feature = "nkef-sba")]
    fn demo_on() -> &'static nextgsim_demo::DemoRuntime {
        let _ = crate::demo::init(
            "demo:\n  enabled: true\n  d447a-nkef-live:\n    enabled: true\n    mode: \"on\"\n",
        );
        let rt = nextgsim_demo::installed().expect("installed");
        assert_eq!(rt.mode(crate::demo::D447A_KEY).as_deref(), Some("on"));
        rt
    }

    #[cfg(feature = "nkef-sba")]
    fn nf(id: &str, ty: &str) -> nextgsim_nkef::live::Fact {
        use nextgsim_nkef::live::{Domain, Fact, FactKind};
        Fact::new(
            Domain::Sba,
            FactKind::NfInstance,
            "nrf",
            format!("nf-{id}"),
            now_ms(),
        )
        .with("nfInstanceId", id)
        .with("nfType", ty)
        .with("nfStatus", "REGISTERED")
    }

    /// nextgsim#488 review M1: the running task expires an NF the NRF poll no
    /// longer returns (with an RFC 3339 `expired_at`), keeps it, and makes it live
    /// again when a later poll returns it.
    ///
    /// REVERT-VERIFIED: emptying the `expire_missing` loop in `handle_nrf_poll`
    /// fails the second assertion.
    #[cfg(feature = "nkef-sba")]
    #[tokio::test]
    async fn an_nrf_poll_that_drops_an_nf_expires_its_fact() {
        use nextgsim_nkef::live::Consumer;
        let _lock = crate::demo::D447A_TEST_LOCK.lock().await;
        demo_on();
        let mut task = NkefTask::new(GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY).0);
        let bsf_line = |text: &str| {
            text.lines()
                .find(|l| l.contains("nfType=BSF"))
                .unwrap_or_else(|| panic!("no BSF fact: {text}"))
                .to_string()
        };
        task.handle_nrf_poll(vec![nf("amf-1", "AMF"), nf("bsf-1", "BSF")], now_ms());
        let before = task.handle_query("bsf", Consumer::Internal).await.unwrap();
        assert!(!bsf_line(&before).contains("expired_at"), "{before}");
        task.handle_nrf_poll(vec![nf("amf-1", "AMF")], now_ms());
        let after = task.handle_query("bsf", Consumer::Internal).await.unwrap();
        let line = bsf_line(&after);
        assert!(
            line.contains("expired_at=20") && line.ends_with('Z'),
            "the dropped BSF expires: {after}"
        );
        assert!(!after
            .lines()
            .find(|l| l.contains("nfType=AMF"))
            .unwrap()
            .contains("expired_at"));
        task.handle_nrf_poll(vec![nf("amf-1", "AMF"), nf("bsf-1", "BSF")], now_ms());
        let back = task.handle_query("bsf", Consumer::Internal).await.unwrap();
        assert!(
            !bsf_line(&back).contains("expired_at"),
            "back is live again: {back}"
        );
    }

    /// A fake h2c amfd: answers a subscribe with 201, a Location and `answer`, a
    /// DELETE with 204, and records (method, path, body) of every request.
    #[cfg(feature = "nkef-sba")]
    async fn fake_amfd(
        answer: serde_json::Value,
    ) -> (
        String,
        std::sync::Arc<std::sync::Mutex<Vec<(String, String, String)>>>,
    ) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                let seen = seen2.clone();
                let answer = answer.clone();
                tokio::spawn(async move {
                    let Ok(mut conn) = h2::server::handshake(sock).await else {
                        return;
                    };
                    while let Some(Ok((req, mut respond))) = conn.accept().await {
                        // Each request on its own task: the connection is only
                        // driven while `accept` is polled.
                        let seen = seen.clone();
                        let answer = answer.clone();
                        tokio::spawn(async move {
                            let (parts, mut body) = req.into_parts();
                            let mut buf = Vec::new();
                            while let Some(Ok(chunk)) = body.data().await {
                                let _ = body.flow_control().release_capacity(chunk.len());
                                buf.extend_from_slice(&chunk);
                            }
                            let method = parts.method.to_string();
                            seen.lock().unwrap().push((
                                method.clone(),
                                parts.uri.path().to_string(),
                                String::from_utf8_lossy(&buf).into_owned(),
                            ));
                            let (status, out) = if method == "POST" {
                                (201, serde_json::to_vec(&answer).unwrap())
                            } else {
                                (204, Vec::new())
                            };
                            let resp = http::Response::builder()
                                .status(status)
                                .header("content-type", "application/json")
                                .header("location", "/namf-evts/v1/subscriptions/488")
                                .body(())
                                .unwrap();
                            let empty = out.is_empty();
                            let Ok(mut stream) = respond.send_response(resp, empty) else {
                                return;
                            };
                            if !empty {
                                let _ = stream.send_data(bytes::Bytes::from(out), true);
                            }
                        });
                    }
                });
            }
        });
        (format!("http://{addr}"), seen)
    }

    #[cfg(feature = "nkef-sba")]
    fn report_list(with_timestamp: bool) -> serde_json::Value {
        let mut nr = serde_json::json!({
            "tai": { "plmnId": { "mcc": "999", "mnc": "70" }, "tac": "000001" },
            "ncgi": { "plmnId": { "mcc": "999", "mnc": "70" }, "nrCellId": "000000010" }
        });
        if with_timestamp {
            nr["ueLocationTimestamp"] =
                serde_json::json!(nextgsim_nkef::live::rfc3339_ms(now_ms() + 1_000));
        }
        serde_json::json!({ "subscriptionId": "488", "reportList": [
            { "type": "LOCATION_REPORT", "location": { "nrLocation": nr } },
            { "type": "CONNECTIVITY_STATE_REPORT",
              "cmInfoList": [{ "cmState": "CONNECTED", "accessType": "3GPP_ACCESS" }] }
        ]})
    }

    /// nextgsim#488 review M2: a query refreshes the AMF's reports through the
    /// real SBA client against an h2c amfd: one POST asking immediate reports
    /// for both events at the closed notify URI, then a DELETE of the Location,
    /// and the bundle reads them (`sba`, `consistent=true`, `cm=CONNECTED`,
    /// `stale=false`). Without a `ueLocationTimestamp` consistency is
    /// `unknown`; without an `amf_sbi_uri` there is no `sba` at all.
    ///
    /// REVERT-VERIFIED: replacing `sba.amf_reports()` in `handle_query` with an
    /// empty list fails the header assertion.
    #[cfg(feature = "nkef-sba")]
    #[tokio::test]
    async fn a_query_reads_the_amfs_immediate_reports_and_unsubscribes() {
        use nextgsim_nkef::live::Consumer;
        let _lock = crate::demo::D447A_TEST_LOCK.lock().await;
        demo_on();
        for with_ts in [true, false] {
            let (uri, seen) = fake_amfd(report_list(with_ts)).await;
            let mut config = test_config();
            config.nkef_live.amf_sbi_uri = Some(uri);
            config.nkef_live.supi = Some("imsi-999700000000001".into());
            let mut task = NkefTask::new(GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY).0);
            let text = task
                .handle_query("route", Consumer::Internal)
                .await
                .unwrap();
            let header = text.lines().next().unwrap();
            if with_ts {
                assert!(
                    header
                        .starts_with("NKEF bundle consumer=internal domains=sba consistent=true ")
                        && header.contains(" cm=CONNECTED stale=false "),
                    "{text}"
                );
            } else {
                assert!(header.contains(" consistent=unknown "), "{text}");
            }
            let seen = seen.lock().unwrap().clone();
            assert_eq!(seen.len(), 2, "{seen:?}");
            assert_eq!(
                (seen[0].0.as_str(), seen[0].1.as_str()),
                ("POST", "/namf-evts/v1/subscriptions")
            );
            let body: serde_json::Value = serde_json::from_str(&seen[0].2).unwrap();
            let sub = &body["subscription"];
            assert_eq!(sub["eventNotifyUri"], crate::nkef::sba::CLOSED_NOTIFY_URI);
            let events = sub["eventList"].as_array().unwrap();
            assert_eq!(events.len(), 2);
            assert!(events.iter().all(|e| e["immediateFlag"] == true), "{body}");
            assert_eq!(
                (seen[1].0.as_str(), seen[1].1.as_str()),
                ("DELETE", "/namf-evts/v1/subscriptions/488")
            );
        }
        // No AMF URI: no SBA client, no sba domain.
        let mut task = NkefTask::new(GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY).0);
        let text = task
            .handle_query("route", Consumer::Internal)
            .await
            .unwrap();
        assert!(
            text.starts_with("NKEF bundle consumer=internal domains=none "),
            "{text}"
        );
    }
}
