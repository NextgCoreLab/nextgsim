//! GTP-U restart counter (TS 23.007, TS 29.281 §8.2)
//!
//! The restart counter is a node's answer to "did I restart since we last spoke?".
//! It is held in non-volatile storage, incremented once per restart, and advertised
//! in the Recovery IE of every Echo Response. A peer that observes it change knows
//! this node lost its state and purges the contexts it held for it (TS 23.007).
//!
//! A hardcoded constant satisfies the encoder and defeats the mechanism: every
//! restart looks like no restart, and both ends keep forwarding into tunnels that no
//! longer exist. That is what this module replaces.

use std::io;
use std::path::Path;

/// A node's GTP-U restart counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartCounter {
    value: u8,
}

impl RestartCounter {
    /// Load the counter from `path`, advance it for this process start, and store it
    /// back.
    ///
    /// Semantics, per TS 23.007: a node that has never started before is at 0, and
    /// each subsequent start increments. So the FIRST call on a fresh path yields 0
    /// and the next yields 1 — the value changes across a restart, which is the only
    /// property a peer relies on. Incrementing on the first start instead would be
    /// harmless but would make "0" unreachable, and 0 is what a peer sees from a node
    /// that has genuinely never restarted.
    ///
    /// Wraps at 255 -> 0 (`wrapping_add`): the counter is one octet on the wire, and
    /// a peer compares values for inequality rather than ordering.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`io::Error`] if the path exists but cannot be read, or
    /// if the new value cannot be written. A caller that cannot persist should prefer
    /// [`Self::in_memory`] and log it: advertising a counter that silently resets to
    /// the same value every start is worse than admitting there is no storage.
    pub fn load_and_advance(path: &Path) -> io::Result<Self> {
        let value = match std::fs::read_to_string(path) {
            Ok(text) => text
                .trim()
                .parse::<u8>()
                .map(|v| v.wrapping_add(1))
                // An unparseable file is a restart we cannot count from. Start over
                // at 0 rather than refusing to boot: the peer sees a change either
                // way, which is what the counter is for.
                .unwrap_or(0),
            Err(e) if e.kind() == io::ErrorKind::NotFound => 0,
            Err(e) => return Err(e),
        };
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, value.to_string())?;
        Ok(Self { value })
    }

    /// A counter with no storage behind it, fixed at 0.
    ///
    /// The honest representation of "no restart-counter path is configured": the value
    /// never changes, so a peer correctly concludes nothing ever restarted, because
    /// this node cannot tell it otherwise.
    pub fn in_memory() -> Self {
        Self { value: 0 }
    }

    /// The value to advertise in the Recovery IE.
    pub fn value(self) -> u8 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "nextgsim-gtpu-restart-{}-{tag}-{nanos}",
            std::process::id()
        ))
    }

    /// The property a peer depends on: the advertised value CHANGES across a restart.
    #[test]
    fn the_counter_advances_across_a_simulated_restart() {
        let path = temp_path("advance");
        let first = RestartCounter::load_and_advance(&path).expect("first start");
        assert_eq!(first.value(), 0, "a node that never restarted advertises 0");

        // A "restart" is just another process start against the same storage.
        let second = RestartCounter::load_and_advance(&path).expect("second start");
        assert_eq!(second.value(), 1);
        assert_ne!(
            first.value(),
            second.value(),
            "a peer detects a restart only by the value changing"
        );

        let third = RestartCounter::load_and_advance(&path).expect("third start");
        assert_eq!(third.value(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_counter_wraps_rather_than_overflowing() {
        let path = temp_path("wrap");
        std::fs::write(&path, "255").expect("seed");
        let counter = RestartCounter::load_and_advance(&path).expect("start");
        assert_eq!(
            counter.value(),
            0,
            "one octet on the wire, so 255 wraps to 0"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unparseable_file_starts_over_rather_than_refusing_to_boot() {
        let path = temp_path("garbage");
        std::fs::write(&path, "not-a-number").expect("seed");
        let counter = RestartCounter::load_and_advance(&path).expect("start");
        assert_eq!(counter.value(), 0);
        assert_eq!(
            std::fs::read_to_string(&path).expect("read").trim(),
            "0",
            "the repaired value must be persisted, or every start reads garbage again"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn without_storage_the_counter_is_a_fixed_zero() {
        assert_eq!(RestartCounter::in_memory().value(), 0);
    }
}
