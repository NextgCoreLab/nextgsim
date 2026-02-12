//! Joint Source-Channel Coding (JSCC)
//!
//! Combines source coding (compression) and channel coding (error protection)
//! into a single step. The encoder maps raw features directly to channel symbols,
//! adapting the compression ratio based on the instantaneous channel quality (SNR).
//!
//! When ONNX models are available the encoder / decoder run learned transforms.
//! Otherwise, a lightweight analytical fallback is used that performs adaptive
//! dimensionality reduction followed by power-normalised symbol mapping.

use std::path::Path;

use tracing::{debug, info};

use nextgsim_ai::config::ExecutionProvider;
use nextgsim_ai::error::ModelError;
use nextgsim_ai::inference::{InferenceEngine, OnnxEngine};
use nextgsim_ai::tensor::TensorData;

use crate::{ChannelQuality, SemanticTask};

/// Error type for JSCC operations
#[derive(Debug, thiserror::Error)]
pub enum JsccError {
    /// Model loading failed
    #[error("Failed to load JSCC model: {0}")]
    ModelLoad(#[from] ModelError),
    /// Inference failed
    #[error("JSCC inference failed: {0}")]
    Inference(#[from] nextgsim_ai::error::InferenceError),
    /// Invalid configuration
    #[error("Invalid JSCC configuration: {reason}")]
    InvalidConfig {
        /// Description of what is wrong
        reason: String,
    },
}

/// Configuration for the JSCC codec.
#[derive(Debug, Clone)]
pub struct JsccConfig {
    /// Base number of channel symbols to produce (before SNR adaptation)
    pub base_symbols: usize,
    /// Minimum number of channel symbols (even at the worst channel)
    pub min_symbols: usize,
    /// Maximum number of channel symbols (at the best channel)
    pub max_symbols: usize,
    /// SNR threshold (dB) below which the codec uses minimum symbols
    pub snr_low_db: f32,
    /// SNR threshold (dB) above which the codec uses maximum symbols
    pub snr_high_db: f32,
    /// Power constraint for channel symbols (average power per symbol)
    pub power_constraint: f32,
}

impl Default for JsccConfig {
    fn default() -> Self {
        Self {
            base_symbols: 64,
            min_symbols: 8,
            max_symbols: 256,
            snr_low_db: 0.0,
            snr_high_db: 25.0,
            power_constraint: 1.0,
        }
    }
}

impl JsccConfig {
    /// Computes the number of channel symbols to use for the given SNR.
    ///
    /// Linearly interpolates between `min_symbols` and `max_symbols`
    /// in the `[snr_low_db, snr_high_db]` range.
    pub fn symbols_for_snr(&self, snr_db: f32) -> usize {
        if snr_db <= self.snr_low_db {
            return self.min_symbols;
        }
        if snr_db >= self.snr_high_db {
            return self.max_symbols;
        }
        let t = (snr_db - self.snr_low_db) / (self.snr_high_db - self.snr_low_db);
        let symbols = self.min_symbols as f32 + t * (self.max_symbols - self.min_symbols) as f32;
        (symbols.round() as usize).clamp(self.min_symbols, self.max_symbols)
    }
}

/// JSCC encoder: maps raw features + channel state to channel symbols.
pub struct JsccEncoder {
    /// ONNX engine for the learned encoder (optional)
    engine: OnnxEngine,
    /// Whether the model is loaded
    model_loaded: bool,
    /// Codec configuration
    config: JsccConfig,
}

impl JsccEncoder {
    /// Creates a new JSCC encoder with the given configuration.
    ///
    /// # Errors
    /// Returns `JsccError::ModelLoad` if the ONNX engine cannot be initialised.
    pub fn new(config: JsccConfig) -> Result<Self, JsccError> {
        let engine = OnnxEngine::new(ExecutionProvider::Cpu)?;
        Ok(Self {
            engine,
            model_loaded: false,
            config,
        })
    }

    /// Creates a JSCC encoder with default configuration.
    ///
    /// # Errors
    /// Returns `JsccError::ModelLoad` if the ONNX engine cannot be initialised.
    pub fn with_defaults() -> Result<Self, JsccError> {
        Self::new(JsccConfig::default())
    }

    /// Loads an ONNX encoder model.
    ///
    /// The model is expected to accept two inputs:
    ///   - `features`: `[1, feature_dim]` f32
    ///   - `snr`:      `[1, 1]` f32 (channel SNR in dB)
    ///     and produce one output:
    ///   - `symbols`:  `[1, num_symbols]` f32
    ///
    /// # Errors
    /// Returns `JsccError::ModelLoad` if the model cannot be loaded.
    pub fn load_model(&mut self, path: &Path) -> Result<(), JsccError> {
        info!("Loading JSCC encoder model from {:?}", path);
        self.engine.load_model(path)?;
        self.model_loaded = true;
        info!("JSCC encoder model loaded");
        Ok(())
    }

    /// Returns whether the ONNX model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &JsccConfig {
        &self.config
    }

    /// Encodes raw features into channel symbols, adapting to the channel.
    ///
    /// # Errors
    /// Returns `JsccError` on failure.
    pub fn encode(
        &self,
        data: &[f32],
        channel: &ChannelQuality,
        task: SemanticTask,
    ) -> Result<JsccSymbols, JsccError> {
        if self.model_loaded {
            self.encode_neural(data, channel, task)
        } else {
            debug!("No JSCC encoder model, using analytical fallback");
            Ok(self.encode_fallback(data, channel, task))
        }
    }

    /// Runs the learned JSCC encoder.
    fn encode_neural(
        &self,
        data: &[f32],
        channel: &ChannelQuality,
        task: SemanticTask,
    ) -> Result<JsccSymbols, JsccError> {
        // Prepare a single concatenated input: [features..., snr]
        let mut input_data = data.to_vec();
        input_data.push(channel.snr_db);
        let input = TensorData::float32(
            input_data.clone(),
            vec![1i64, input_data.len() as i64],
        );
        let output = self.engine.infer(&input)?;
        let symbols = output
            .as_f32_slice()
            .ok_or_else(|| JsccError::InvalidConfig {
                reason: "JSCC encoder model did not produce f32 output".to_string(),
            })?
            .to_vec();

        Ok(JsccSymbols {
            symbols,
            snr_db: channel.snr_db,
            task_id: crate::codec::task_to_id(task),
            original_len: data.len(),
        })
    }

    /// Analytical JSCC fallback.
    ///
    /// 1. Adaptively selects the number of symbols based on SNR.
    /// 2. Performs mean-pooling dimensionality reduction.
    /// 3. Applies power normalisation so that average symbol power = `power_constraint`.
    fn encode_fallback(
        &self,
        data: &[f32],
        channel: &ChannelQuality,
        task: SemanticTask,
    ) -> JsccSymbols {
        let num_symbols = self.config.symbols_for_snr(channel.snr_db);
        let stride = (data.len() / num_symbols).max(1);

        let mut symbols = Vec::with_capacity(num_symbols);
        for i in 0..num_symbols {
            let start = i * stride;
            let end = ((i + 1) * stride).min(data.len());
            if start < data.len() {
                let chunk = &data[start..end];
                let mean: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
                symbols.push(mean);
            }
        }

        // Power normalisation
        let avg_power: f32 = if symbols.is_empty() {
            0.0
        } else {
            symbols.iter().map(|s| s * s).sum::<f32>() / symbols.len() as f32
        };

        if avg_power > 0.0 {
            let scale = (self.config.power_constraint / avg_power).sqrt();
            for s in &mut symbols {
                *s *= scale;
            }
        }

        JsccSymbols {
            symbols,
            snr_db: channel.snr_db,
            task_id: crate::codec::task_to_id(task),
            original_len: data.len(),
        }
    }
}

/// JSCC decoder: reconstructs features from received channel symbols.
pub struct JsccDecoder {
    /// ONNX engine for the learned decoder (optional)
    engine: OnnxEngine,
    /// Whether the model is loaded
    model_loaded: bool,
}

impl JsccDecoder {
    /// Creates a new JSCC decoder.
    ///
    /// # Errors
    /// Returns `JsccError::ModelLoad` if the ONNX engine cannot be initialised.
    pub fn new() -> Result<Self, JsccError> {
        let engine = OnnxEngine::new(ExecutionProvider::Cpu)?;
        Ok(Self {
            engine,
            model_loaded: false,
        })
    }

    /// Loads an ONNX decoder model.
    ///
    /// The model is expected to accept two inputs:
    ///   - `symbols`: `[1, num_symbols]` f32 (received channel symbols)
    ///   - `snr`:     `[1, 1]` f32 (channel SNR in dB)
    ///     and produce one output:
    ///   - `features`: `[1, output_dim]` f32
    ///
    /// # Errors
    /// Returns `JsccError::ModelLoad` if the model cannot be loaded.
    pub fn load_model(&mut self, path: &Path) -> Result<(), JsccError> {
        info!("Loading JSCC decoder model from {:?}", path);
        self.engine.load_model(path)?;
        self.model_loaded = true;
        info!("JSCC decoder model loaded");
        Ok(())
    }

    /// Returns whether the ONNX model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Decodes received channel symbols back to reconstructed features.
    ///
    /// # Errors
    /// Returns `JsccError` on failure.
    pub fn decode(&self, received: &JsccSymbols) -> Result<Vec<f32>, JsccError> {
        if self.model_loaded {
            self.decode_neural(received)
        } else {
            debug!("No JSCC decoder model, using analytical fallback");
            Ok(self.decode_fallback(received))
        }
    }

    /// Runs the learned JSCC decoder.
    fn decode_neural(&self, received: &JsccSymbols) -> Result<Vec<f32>, JsccError> {
        let mut input_data = received.symbols.clone();
        input_data.push(received.snr_db);
        let input = TensorData::float32(
            input_data.clone(),
            vec![1i64, input_data.len() as i64],
        );
        let output = self.engine.infer(&input)?;
        let decoded = output
            .as_f32_slice()
            .ok_or_else(|| JsccError::InvalidConfig {
                reason: "JSCC decoder model did not produce f32 output".to_string(),
            })?
            .to_vec();
        Ok(decoded)
    }

    /// Nearest-neighbour upsampling fallback.
    fn decode_fallback(&self, received: &JsccSymbols) -> Vec<f32> {
        let output_len = received.original_len;
        if received.symbols.is_empty() || output_len == 0 {
            return vec![0.0; output_len];
        }

        let stride = output_len / received.symbols.len().max(1);
        let mut output = Vec::with_capacity(output_len);

        for &sym in &received.symbols {
            for _ in 0..stride {
                output.push(sym);
            }
        }

        while output.len() < output_len {
            output.push(received.symbols.last().copied().unwrap_or(0.0));
        }

        output.truncate(output_len);
        output
    }
}

/// Channel symbols produced by the JSCC encoder.
///
/// This struct travels over the (simulated) physical channel.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JsccSymbols {
    /// Complex-valued channel symbols (represented as f32 for simplicity)
    pub symbols: Vec<f32>,
    /// Channel SNR in dB at the time of encoding
    pub snr_db: f32,
    /// Semantic task ID
    pub task_id: u32,
    /// Original data length (needed by the fallback decoder)
    pub original_len: usize,
}

impl JsccSymbols {
    /// Returns the number of channel uses (symbols).
    pub fn num_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// Returns the bandwidth ratio (channel uses / original length).
    pub fn bandwidth_ratio(&self) -> f32 {
        if self.original_len == 0 {
            0.0
        } else {
            self.symbols.len() as f32 / self.original_len as f32
        }
    }

    /// Applies channel corruption (AWGN, fading) - A18.4
    pub fn apply_channel(&mut self, channel_model: &ChannelModel) {
        match channel_model {
            ChannelModel::AWGN { snr_db } => {
                self.apply_awgn(*snr_db);
            }
            ChannelModel::Rayleigh { snr_db } => {
                self.apply_rayleigh_fading(*snr_db);
            }
            ChannelModel::Rician { snr_db, k_factor } => {
                self.apply_rician_fading(*snr_db, *k_factor);
            }
        }
    }

    /// Applies Additive White Gaussian Noise (AWGN)
    fn apply_awgn(&mut self, snr_db: f32) {
        let mut rng = rand::thread_rng();

        // Compute signal power
        let signal_power = self.symbols.iter().map(|s| s * s).sum::<f32>() / self.symbols.len() as f32;

        // Compute noise power from SNR
        let snr_linear = 10.0f32.powf(snr_db / 10.0);
        let noise_power = signal_power / snr_linear;
        let noise_std = noise_power.sqrt();

        // Add Gaussian noise
        for symbol in &mut self.symbols {
            let noise = sample_gaussian(&mut rng, noise_std);
            *symbol += noise;
        }
    }

    /// Applies Rayleigh fading + AWGN
    fn apply_rayleigh_fading(&mut self, snr_db: f32) {
        let mut rng = rand::thread_rng();

        // Apply Rayleigh fading coefficient to each symbol
        for symbol in &mut self.symbols {
            // Rayleigh fading: |h| where h ~ CN(0, 1)
            let h_real = sample_gaussian(&mut rng, 1.0 / std::f32::consts::SQRT_2);
            let h_imag = sample_gaussian(&mut rng, 1.0 / std::f32::consts::SQRT_2);
            let h_magnitude = (h_real * h_real + h_imag * h_imag).sqrt();

            *symbol *= h_magnitude;
        }

        // Add AWGN
        self.apply_awgn(snr_db);
    }

    /// Applies Rician fading + AWGN
    fn apply_rician_fading(&mut self, snr_db: f32, k_factor: f32) {
        let mut rng = rand::thread_rng();

        // Rician fading: line-of-sight + scattered components
        let los_power = k_factor / (1.0 + k_factor);
        let scatter_power = 1.0 / (1.0 + k_factor);

        for symbol in &mut self.symbols {
            // Line-of-sight component
            let los = los_power.sqrt();

            // Scattered component (Rayleigh)
            let h_real = sample_gaussian(&mut rng, scatter_power.sqrt() / std::f32::consts::SQRT_2);
            let h_imag = sample_gaussian(&mut rng, scatter_power.sqrt() / std::f32::consts::SQRT_2);
            let scatter_magnitude = (h_real * h_real + h_imag * h_imag).sqrt();

            let h_magnitude = los + scatter_magnitude;
            *symbol *= h_magnitude;
        }

        // Add AWGN
        self.apply_awgn(snr_db);
    }
}

/// Channel model for simulation (A18.4)
#[derive(Debug, Clone, Copy)]
pub enum ChannelModel {
    /// Additive White Gaussian Noise
    AWGN {
        /// Signal-to-noise ratio in dB
        snr_db: f32,
    },
    /// Rayleigh fading (no line-of-sight)
    Rayleigh {
        /// Signal-to-noise ratio in dB
        snr_db: f32,
    },
    /// Rician fading (with line-of-sight)
    Rician {
        /// Signal-to-noise ratio in dB
        snr_db: f32,
        /// K-factor (ratio of line-of-sight to scattered power)
        k_factor: f32,
    },
}

/// Samples from N(0, sigma^2) using Box-Muller transform
fn sample_gaussian<R: rand::Rng>(rng: &mut R, sigma: f32) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.gen();
    sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Combined JSCC codec holding both encoder and decoder.
pub struct JsccCodec {
    /// The JSCC encoder
    pub encoder: JsccEncoder,
    /// The JSCC decoder
    pub decoder: JsccDecoder,
}

impl JsccCodec {
    /// Creates a new JSCC codec with the given configuration.
    ///
    /// # Errors
    /// Returns `JsccError` if engine initialisation fails.
    pub fn new(config: JsccConfig) -> Result<Self, JsccError> {
        Ok(Self {
            encoder: JsccEncoder::new(config)?,
            decoder: JsccDecoder::new()?,
        })
    }

    /// Creates a codec with default configuration.
    ///
    /// # Errors
    /// Returns `JsccError` if engine initialisation fails.
    pub fn with_defaults() -> Result<Self, JsccError> {
        Self::new(JsccConfig::default())
    }

    /// Loads encoder and decoder ONNX models.
    ///
    /// # Errors
    /// Returns `JsccError` if either model fails to load.
    pub fn load_models(
        &mut self,
        encoder_path: &Path,
        decoder_path: &Path,
    ) -> Result<(), JsccError> {
        self.encoder.load_model(encoder_path)?;
        self.decoder.load_model(decoder_path)?;
        Ok(())
    }

    /// Encodes then decodes (end-to-end round trip) for testing / simulation.
    ///
    /// # Errors
    /// Returns `JsccError` on failure.
    pub fn round_trip(
        &self,
        data: &[f32],
        channel: &ChannelQuality,
        task: SemanticTask,
    ) -> Result<Vec<f32>, JsccError> {
        let symbols = self.encoder.encode(data, channel, task)?;
        self.decoder.decode(&symbols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChannelQuality;

    #[test]
    fn test_jscc_config_symbols_for_snr() {
        let config = JsccConfig {
            min_symbols: 8,
            max_symbols: 256,
            snr_low_db: 0.0,
            snr_high_db: 20.0,
            ..Default::default()
        };

        assert_eq!(config.symbols_for_snr(-5.0), 8);
        assert_eq!(config.symbols_for_snr(0.0), 8);
        assert_eq!(config.symbols_for_snr(30.0), 256);

        let mid = config.symbols_for_snr(10.0);
        assert!(mid > 8 && mid < 256);
    }

    #[test]
    fn test_jscc_encoder_fallback() {
        let encoder = JsccEncoder::with_defaults().expect("Failed to create encoder");
        assert!(!encoder.is_model_loaded());

        let data: Vec<f32> = (0..128).map(|i| i as f32 / 127.0).collect();
        let channel = ChannelQuality::new(15.0, 1000.0, 0.01);

        let symbols = encoder
            .encode(&data, &channel, SemanticTask::ImageClassification)
            .expect("Encoding failed");

        assert!(symbols.num_symbols() > 0);
        assert!(symbols.bandwidth_ratio() > 0.0);
        assert!(symbols.bandwidth_ratio() <= 1.0);
    }

    #[test]
    fn test_jscc_decoder_fallback() {
        let decoder = JsccDecoder::new().expect("Failed to create decoder");
        let symbols = JsccSymbols {
            symbols: vec![0.1, 0.2, 0.3, 0.4],
            snr_db: 15.0,
            task_id: 0,
            original_len: 32,
        };

        let decoded = decoder.decode(&symbols).expect("Decoding failed");
        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn test_jscc_codec_roundtrip() {
        let codec = JsccCodec::with_defaults().expect("Failed to create codec");
        let data: Vec<f32> = (0..256).map(|i| (i as f32) / 255.0).collect();
        let channel = ChannelQuality::new(20.0, 1000.0, 0.005);

        let reconstructed = codec
            .round_trip(&data, &channel, SemanticTask::SensorFusion)
            .expect("Round trip failed");

        assert_eq!(reconstructed.len(), data.len());
    }

    #[test]
    fn test_jscc_channel_adaptation() {
        let encoder = JsccEncoder::with_defaults().expect("Failed to create encoder");
        let data: Vec<f32> = (0..256).map(|i| (i as f32) / 255.0).collect();

        let good = ChannelQuality::new(20.0, 1000.0, 0.005);
        let poor = ChannelQuality::new(2.0, 100.0, 0.15);

        let good_symbols = encoder
            .encode(&data, &good, SemanticTask::ImageClassification)
            .expect("Encoding failed");
        let poor_symbols = encoder
            .encode(&data, &poor, SemanticTask::ImageClassification)
            .expect("Encoding failed");

        // Good channel should allocate more symbols than poor channel
        assert!(good_symbols.num_symbols() >= poor_symbols.num_symbols());
    }

    // Tests for A18.4: Channel simulation

    #[test]
    fn test_awgn_channel() {
        let encoder = JsccEncoder::with_defaults().expect("Failed to create encoder");
        let data: Vec<f32> = vec![1.0; 64];
        let channel = ChannelQuality::new(15.0, 500.0, 0.01);

        let mut symbols = encoder
            .encode(&data, &channel, SemanticTask::SensorFusion)
            .expect("Encoding failed");

        let original_symbols = symbols.symbols.clone();

        // Apply AWGN channel
        symbols.apply_channel(&ChannelModel::AWGN { snr_db: 10.0 });

        // Symbols should be different after noise
        assert_ne!(symbols.symbols, original_symbols);
    }

    #[test]
    fn test_rayleigh_fading() {
        let encoder = JsccEncoder::with_defaults().expect("Failed to create encoder");
        let data: Vec<f32> = vec![1.0; 64];
        let channel = ChannelQuality::new(15.0, 500.0, 0.01);

        let mut symbols = encoder
            .encode(&data, &channel, SemanticTask::ImageClassification)
            .expect("Encoding failed");

        let original_symbols = symbols.symbols.clone();

        // Apply Rayleigh fading
        symbols.apply_channel(&ChannelModel::Rayleigh { snr_db: 10.0 });

        // Symbols should be different
        assert_ne!(symbols.symbols, original_symbols);
    }

    #[test]
    fn test_rician_fading() {
        let encoder = JsccEncoder::with_defaults().expect("Failed to create encoder");
        let data: Vec<f32> = vec![1.0; 64];
        let channel = ChannelQuality::new(15.0, 500.0, 0.01);

        let mut symbols = encoder
            .encode(&data, &channel, SemanticTask::ImageClassification)
            .expect("Encoding failed");

        let original_symbols = symbols.symbols.clone();

        // Apply Rician fading with K=3 (strong line-of-sight)
        symbols.apply_channel(&ChannelModel::Rician {
            snr_db: 15.0,
            k_factor: 3.0,
        });

        // Symbols should be different
        assert_ne!(symbols.symbols, original_symbols);
    }
}
