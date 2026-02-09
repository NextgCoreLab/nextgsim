//! Multi-modal semantic communication traits
//!
//! Defines generic `SemanticEncode` and `SemanticDecode` traits that are
//! parameterised over the data type `T`, enabling the same pipeline to
//! handle images, audio, video, sensor vectors, and custom modalities.
//!
//! Concrete implementations are provided for 1-D feature vectors (`Vec<f32>`).
//! Marker types (`ImageData`, `AudioData`, `VideoData`) are defined so
//! downstream crates can implement the traits for richer data types without
//! breaking backward compatibility.

use crate::codec::CodecError;
use crate::{ChannelQuality, SemanticFeatures, SemanticTask};

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Trait for encoding a typed source into semantic features.
///
/// `T` is the source data type (e.g. `Vec<f32>`, `ImageData`, `AudioData`).
pub trait SemanticEncode<T> {
    /// Encodes `source` for the given `task`, producing compressed semantic features.
    ///
    /// # Errors
    /// Returns `CodecError` if encoding fails.
    fn encode(&self, source: &T, task: SemanticTask) -> Result<SemanticFeatures, CodecError>;

    /// Channel-adaptive encoding: the compression level is adjusted based
    /// on the current channel quality.
    ///
    /// The default implementation calls `encode` and then prunes features.
    /// Implementations may override for tighter integration.
    ///
    /// # Errors
    /// Returns `CodecError` if encoding fails.
    fn encode_adaptive(
        &self,
        source: &T,
        task: SemanticTask,
        channel: &ChannelQuality,
    ) -> Result<SemanticFeatures, CodecError> {
        let features = self.encode(source, task)?;
        let keep = 1.0 / channel.recommended_compression();
        Ok(features.prune(keep.clamp(0.1, 1.0)))
    }
}

/// Trait for decoding semantic features back into a typed output.
///
/// `T` is the reconstructed data type.
pub trait SemanticDecode<T> {
    /// Decodes compressed features into an instance of `T`.
    ///
    /// # Errors
    /// Returns `CodecError` if decoding fails.
    fn decode(&self, features: &SemanticFeatures) -> Result<T, CodecError>;
}

// ---------------------------------------------------------------------------
// 1-D vector implementation (backward-compatible with existing pipeline)
// ---------------------------------------------------------------------------

/// Encoder for 1-D feature vectors (`Vec<f32>`).
///
/// This wraps the [`NeuralEncoder`](crate::codec::NeuralEncoder) and provides
/// the `SemanticEncode<Vec<f32>>` trait implementation.
pub struct VectorEncoder {
    inner: crate::codec::NeuralEncoder,
}

impl VectorEncoder {
    /// Creates a new vector encoder with the given target compressed dimension.
    ///
    /// # Errors
    /// Returns `CodecError` if the ONNX engine fails to initialise.
    pub fn new(target_dim: usize) -> Result<Self, CodecError> {
        Ok(Self {
            inner: crate::codec::NeuralEncoder::new(target_dim)?,
        })
    }

    /// Returns a mutable reference to the inner neural encoder (e.g. for
    /// loading an ONNX model).
    pub fn inner_mut(&mut self) -> &mut crate::codec::NeuralEncoder {
        &mut self.inner
    }
}

impl SemanticEncode<Vec<f32>> for VectorEncoder {
    fn encode(&self, source: &Vec<f32>, task: SemanticTask) -> Result<SemanticFeatures, CodecError> {
        self.inner.encode(source, task)
    }
}

/// Decoder for 1-D feature vectors (`Vec<f32>`).
///
/// This wraps the [`NeuralDecoder`](crate::codec::NeuralDecoder).
pub struct VectorDecoder {
    inner: crate::codec::NeuralDecoder,
}

impl VectorDecoder {
    /// Creates a new vector decoder.
    ///
    /// # Errors
    /// Returns `CodecError` if the ONNX engine fails to initialise.
    pub fn new() -> Result<Self, CodecError> {
        Ok(Self {
            inner: crate::codec::NeuralDecoder::new()?,
        })
    }

    /// Returns a mutable reference to the inner neural decoder (e.g. for
    /// loading an ONNX model).
    pub fn inner_mut(&mut self) -> &mut crate::codec::NeuralDecoder {
        &mut self.inner
    }
}

impl SemanticDecode<Vec<f32>> for VectorDecoder {
    fn decode(&self, features: &SemanticFeatures) -> Result<Vec<f32>, CodecError> {
        self.inner.decode(features)
    }
}

// ---------------------------------------------------------------------------
// Marker types for future modalities
// ---------------------------------------------------------------------------

/// Marker type for image data.
///
/// Downstream crates can implement `SemanticEncode<ImageData>` and
/// `SemanticDecode<ImageData>` for image-specific pipelines.
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Raw pixel values (e.g. RGB, flattened row-major)
    pub pixels: Vec<f32>,
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Number of channels (e.g. 3 for RGB)
    pub channels: usize,
}

impl ImageData {
    /// Creates a new image data container.
    pub fn new(pixels: Vec<f32>, width: usize, height: usize, channels: usize) -> Self {
        Self {
            pixels,
            width,
            height,
            channels,
        }
    }

    /// Returns the total number of elements.
    pub fn num_elements(&self) -> usize {
        self.width * self.height * self.channels
    }

    /// Returns the flat pixel slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.pixels
    }
}

/// Marker type for audio data.
///
/// Downstream crates can implement `SemanticEncode<AudioData>` and
/// `SemanticDecode<AudioData>` for audio-specific pipelines.
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Audio samples (mono, normalised to [-1, 1])
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
}

impl AudioData {
    /// Creates new audio data.
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Returns the duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 || self.channels == 0 {
            0.0
        } else {
            self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
        }
    }

    /// Returns the flat sample slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.samples
    }
}

/// Marker type for video data.
///
/// Downstream crates can implement `SemanticEncode<VideoData>` and
/// `SemanticDecode<VideoData>` for video-specific pipelines.
#[derive(Debug, Clone)]
pub struct VideoData {
    /// Sequence of frames (each frame is an `ImageData`).
    pub frames: Vec<ImageData>,
    /// Frames per second.
    pub fps: f32,
}

impl VideoData {
    /// Creates new video data.
    pub fn new(frames: Vec<ImageData>, fps: f32) -> Self {
        Self { frames, fps }
    }

    /// Returns the number of frames.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Returns the duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        if self.fps <= 0.0 {
            0.0
        } else {
            self.frames.len() as f32 / self.fps
        }
    }
}

// ---------------------------------------------------------------------------
// Blanket encoder / decoder for ImageData using the flat pixel vector
// ---------------------------------------------------------------------------

impl SemanticEncode<ImageData> for VectorEncoder {
    fn encode(&self, source: &ImageData, task: SemanticTask) -> Result<SemanticFeatures, CodecError> {
        self.inner.encode(&source.pixels, task)
    }
}

impl SemanticDecode<ImageData> for VectorDecoder {
    fn decode(&self, features: &SemanticFeatures) -> Result<ImageData, CodecError> {
        let pixels = self.inner.decode(features)?;
        // Attempt to reconstruct dimensions from original_dims
        let (width, height, channels) = match *features.original_dims.as_slice() {
            [w, h, c] => (w, h, c),
            [total] => {
                // Guess: treat as a square single-channel image
                let side = (total as f64).sqrt() as usize;
                (side, side.max(1), 1)
            }
            _ => (pixels.len(), 1, 1),
        };
        Ok(ImageData::new(pixels, width, height, channels))
    }
}

// ---------------------------------------------------------------------------
// Video Temporal Coding (A18.6)
// ---------------------------------------------------------------------------

/// Video temporal encoder that exploits temporal redundancy
pub struct VideoTemporalEncoder {
    /// Target compressed dimension per frame
    target_dim: usize,
}

impl VideoTemporalEncoder {
    /// Creates a new video temporal encoder
    pub fn new(target_dim: usize) -> Self {
        Self { target_dim }
    }

    /// Encodes video using temporal prediction
    pub fn encode(&self, video: &VideoData) -> SemanticFeatures {
        if video.frames.is_empty() {
            return SemanticFeatures::new(5, vec![], vec![0]);
        }

        let mut compressed_features = Vec::new();

        // Encode first frame as I-frame (intra)
        let first_frame_pixels = &video.frames[0].pixels;
        let i_frame = self.encode_i_frame(first_frame_pixels);
        compressed_features.extend(i_frame);

        // Encode subsequent frames as P-frames (predicted)
        for i in 1..video.frames.len() {
            let prev_pixels = &video.frames[i - 1].pixels;
            let curr_pixels = &video.frames[i].pixels;

            let p_frame = self.encode_p_frame(prev_pixels, curr_pixels);
            compressed_features.extend(p_frame);
        }

        let total_pixels: usize = video.frames.iter().map(ImageData::num_elements).sum();

        SemanticFeatures::new(5, compressed_features, vec![total_pixels])
    }

    /// Encodes an I-frame (intra-coded)
    fn encode_i_frame(&self, pixels: &[f32]) -> Vec<f32> {
        // Simple mean-pooling compression
        let stride = (pixels.len() / self.target_dim).max(1);
        let mut features = Vec::with_capacity(self.target_dim);

        for i in 0..self.target_dim {
            let start = i * stride;
            let end = ((i + 1) * stride).min(pixels.len());
            if start < pixels.len() {
                let chunk = &pixels[start..end];
                let mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
                features.push(mean);
            }
        }

        features
    }

    /// Encodes a P-frame (predicted from previous frame)
    fn encode_p_frame(&self, prev_pixels: &[f32], curr_pixels: &[f32]) -> Vec<f32> {
        // Compute motion vectors (simplified: pixel difference)
        let motion = self.estimate_motion(prev_pixels, curr_pixels);

        // Compress motion vectors
        let stride = (motion.len() / (self.target_dim / 2).max(1)).max(1);
        let mut features = Vec::with_capacity(self.target_dim / 2);

        for i in 0..(self.target_dim / 2) {
            let start = i * stride;
            let end = ((i + 1) * stride).min(motion.len());
            if start < motion.len() {
                let chunk = &motion[start..end];
                let mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
                features.push(mean);
            }
        }

        features
    }

    /// Estimates motion between two frames
    fn estimate_motion(&self, prev: &[f32], curr: &[f32]) -> Vec<f32> {
        let len = prev.len().min(curr.len());
        prev[..len]
            .iter()
            .zip(curr[..len].iter())
            .map(|(p, c)| c - p)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Speech/NLP-Specific Codecs (A18.7)
// ---------------------------------------------------------------------------

/// Speech-specific semantic codec
pub struct SpeechCodec {
    /// Target mel-frequency cepstral coefficients (MFCCs)
    target_mfcc: usize,
}

impl SpeechCodec {
    /// Creates a new speech codec
    pub fn new(target_mfcc: usize) -> Self {
        Self { target_mfcc }
    }

    /// Encodes audio for speech recognition
    pub fn encode(&self, audio: &AudioData) -> SemanticFeatures {
        // Simplified MFCC-like encoding: spectral features
        let features = self.extract_spectral_features(&audio.samples);

        SemanticFeatures::new(2, features, vec![audio.samples.len()])
    }

    /// Extracts simplified spectral features
    fn extract_spectral_features(&self, samples: &[f32]) -> Vec<f32> {
        // Simplified: divide into frames and compute energy per frame
        let frame_size = 512;
        let hop_size = 256;
        let num_frames = (samples.len() / hop_size).max(1);

        let mut features = Vec::new();

        for i in 0..num_frames {
            let start = i * hop_size;
            let end = (start + frame_size).min(samples.len());

            if start < samples.len() {
                let frame = &samples[start..end];

                // Compute frame energy (simplified MFCC component)
                let energy = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
                features.push(energy.sqrt());
            }

            if features.len() >= self.target_mfcc {
                break;
            }
        }

        // Pad if needed
        while features.len() < self.target_mfcc {
            features.push(0.0);
        }

        features.truncate(self.target_mfcc);
        features
    }
}

/// NLP-specific semantic codec
pub struct NLPCodec {
    /// Target semantic dimension
    target_dim: usize,
}

impl NLPCodec {
    /// Creates a new NLP codec
    pub fn new(target_dim: usize) -> Self {
        Self { target_dim }
    }

    /// Encodes token embeddings for text understanding
    pub fn encode(&self, token_embeddings: &[f32], task: SemanticTask) -> SemanticFeatures {
        // Compress token embeddings using attention pooling
        let features = self.attention_pool(token_embeddings);

        let task_id = match task {
            SemanticTask::TextUnderstanding => 3,
            SemanticTask::Custom(id) => id,
            _ => 3,
        };

        SemanticFeatures::new(task_id, features, vec![token_embeddings.len()])
    }

    /// Attention-based pooling of token embeddings
    fn attention_pool(&self, embeddings: &[f32]) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![0.0; self.target_dim];
        }

        // Simplified: weighted average based on magnitude
        let stride = (embeddings.len() / self.target_dim).max(1);
        let mut features = Vec::with_capacity(self.target_dim);

        for i in 0..self.target_dim {
            let start = i * stride;
            let end = ((i + 1) * stride).min(embeddings.len());

            if start < embeddings.len() {
                let chunk = &embeddings[start..end];

                // Weighted by magnitude (attention proxy)
                let total_weight: f32 = chunk.iter().map(|e| e.abs()).sum();
                let weighted_sum: f32 = chunk.iter().map(|e| e * e.abs()).sum();

                let pooled = if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    0.0
                };

                features.push(pooled);
            }
        }

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_encoder_trait() {
        let encoder = VectorEncoder::new(16).expect("Failed to create encoder");
        let data = vec![0.1f32; 128];
        let features = encoder
            .encode(&data, SemanticTask::SensorFusion)
            .expect("Encoding failed");
        assert!(features.num_features() > 0);
    }

    #[test]
    fn test_vector_decoder_trait() {
        let decoder = VectorDecoder::new().expect("Failed to create decoder");
        let features = SemanticFeatures::new(0, vec![0.5; 8], vec![64]);
        let decoded: Vec<f32> = decoder.decode(&features).expect("Decoding failed");
        assert_eq!(decoded.len(), 64);
    }

    #[test]
    fn test_adaptive_encoding_trait() {
        let encoder = VectorEncoder::new(32).expect("Failed to create encoder");
        let data = vec![0.5f32; 256];
        let channel = ChannelQuality::new(10.0, 500.0, 0.05);
        let features = encoder
            .encode_adaptive(&data, SemanticTask::ImageClassification, &channel)
            .expect("Adaptive encoding failed");
        // After pruning, should have fewer features than 32
        assert!(features.num_features() <= 32);
    }

    #[test]
    fn test_image_data_basics() {
        let img = ImageData::new(vec![0.0; 3 * 4 * 3], 4, 3, 3);
        assert_eq!(img.num_elements(), 36);
        assert_eq!(img.as_slice().len(), 36);
    }

    #[test]
    fn test_audio_data_duration() {
        let audio = AudioData::new(vec![0.0; 16000], 16000, 1);
        assert!((audio.duration_secs() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_video_data_duration() {
        let frame = ImageData::new(vec![0.0; 12], 2, 2, 3);
        let video = VideoData::new(vec![frame; 30], 30.0);
        assert_eq!(video.num_frames(), 30);
        assert!((video.duration_secs() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_image_encode_via_trait() {
        let encoder = VectorEncoder::new(8).expect("Failed to create encoder");
        let img = ImageData::new(vec![0.5; 64], 8, 8, 1);
        let features = SemanticEncode::<ImageData>::encode(&encoder, &img, SemanticTask::ImageClassification)
            .expect("Image encoding failed");
        assert!(features.num_features() > 0);
    }

    // Tests for A18.6: Video temporal coding

    #[test]
    fn test_video_temporal_encoder() {
        let frames = vec![
            ImageData::new(vec![0.5; 64], 8, 8, 1),
            ImageData::new(vec![0.6; 64], 8, 8, 1),
            ImageData::new(vec![0.7; 64], 8, 8, 1),
        ];
        let video = VideoData::new(frames, 30.0);

        let encoder = VideoTemporalEncoder::new(16);
        let compressed = encoder.encode(&video);

        assert!(compressed.num_features() > 0);
        assert_eq!(compressed.task_id, 5); // VideoAnalytics
    }

    #[test]
    fn test_video_motion_estimation() {
        let encoder = VideoTemporalEncoder::new(16);

        let frame1 = vec![1.0, 2.0, 3.0, 4.0];
        let frame2 = vec![1.1, 2.1, 3.1, 4.1];

        let motion = encoder.estimate_motion(&frame1, &frame2);

        assert_eq!(motion.len(), 4);
        // Motion should be small difference
        for &m in &motion {
            assert!(m.abs() < 0.5);
        }
    }

    // Tests for A18.7: Speech/NLP-specific codecs

    #[test]
    fn test_speech_encoder() {
        let audio = AudioData::new(vec![0.1; 16000], 16000, 1);
        let encoder = SpeechCodec::new(64);

        let compressed = encoder.encode(&audio);

        assert!(compressed.num_features() > 0);
        assert_eq!(compressed.task_id, 2); // SpeechRecognition
    }

    #[test]
    fn test_nlp_encoder() {
        let text = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Token embeddings
        let encoder = NLPCodec::new(3);

        let compressed = encoder.encode(&text, SemanticTask::TextUnderstanding);

        assert!(compressed.num_features() > 0);
        assert_eq!(compressed.task_id, 3); // TextUnderstanding
    }
}
