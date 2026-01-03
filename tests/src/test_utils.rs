//! Test utility functions for integration tests
//!
//! Provides common utilities for test setup, logging, and assertions.

use std::future::Future;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tracing_subscriber::{fmt, EnvFilter};

/// Result type for integration tests
pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Initialize logging for tests with optional filter
///
/// Uses RUST_LOG environment variable if set, otherwise defaults to "info"
pub fn init_test_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    
    let _ = fmt()
        .with_env_filter(filter)
        .with_test_writer()
        .try_init();
}

/// Wait for a condition to become true with timeout
///
/// # Arguments
/// * `condition` - Async function that returns true when condition is met
/// * `timeout_duration` - Maximum time to wait
/// * `poll_interval` - How often to check the condition
///
/// # Returns
/// * `Ok(())` if condition became true within timeout
/// * `Err` if timeout elapsed
pub async fn wait_for_condition<F, Fut>(
    mut condition: F,
    timeout_duration: Duration,
    poll_interval: Duration,
) -> TestResult
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let result = timeout(timeout_duration, async {
        loop {
            if condition().await {
                return;
            }
            sleep(poll_interval).await;
        }
    })
    .await;

    match result {
        Ok(()) => Ok(()),
        Err(_) => Err("Condition not met within timeout".into()),
    }
}

/// Default timeout for test operations
pub const DEFAULT_TEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Default poll interval for condition checks
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Generate a unique test port to avoid conflicts
pub fn get_test_port(base: u16, offset: u16) -> u16 {
    base + offset
}

/// SCTP port base for tests
pub const TEST_SCTP_PORT_BASE: u16 = 38412;

/// GTP-U port base for tests
pub const TEST_GTP_PORT_BASE: u16 = 2152;

/// CLI port base for tests
pub const TEST_CLI_PORT_BASE: u16 = 4997;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_wait_for_condition_success() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();
        
        // Set flag after a short delay
        tokio::spawn(async move {
            sleep(Duration::from_millis(50)).await;
            flag_clone.store(true, Ordering::SeqCst);
        });
        
        let result = wait_for_condition(
            || async { flag.load(Ordering::SeqCst) },
            Duration::from_secs(1),
            Duration::from_millis(10),
        )
        .await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wait_for_condition_timeout() {
        let result = wait_for_condition(
            || async { false },
            Duration::from_millis(100),
            Duration::from_millis(10),
        )
        .await;
        
        assert!(result.is_err());
    }

    #[test]
    fn test_get_test_port() {
        assert_eq!(get_test_port(38412, 0), 38412);
        assert_eq!(get_test_port(38412, 1), 38413);
    }
}
