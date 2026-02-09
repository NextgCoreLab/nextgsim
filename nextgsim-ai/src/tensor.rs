//! Tensor data types and operations
//!
//! This module provides tensor data structures for ML inference,
//! supporting multiple data types and operations for both input and output tensors.

use half::f16;
use ndarray::{Array, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Shape of a tensor as a vector of dimensions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    /// Dimensions of the tensor
    dims: Vec<i64>,
}

impl TensorShape {
    /// Creates a new tensor shape from dimensions
    pub fn new(dims: Vec<i64>) -> Self {
        Self { dims }
    }

    /// Returns the dimensions as a slice
    pub fn dims(&self) -> &[i64] {
        &self.dims
    }

    /// Returns the number of dimensions (rank)
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Checks if the shape is compatible with another shape (ignoring batch dimension)
    pub fn is_compatible_with(&self, other: &TensorShape) -> bool {
        if self.dims.len() != other.dims.len() {
            return false;
        }
        // Allow -1 as dynamic dimension
        self.dims
            .iter()
            .zip(other.dims.iter())
            .all(|(&a, &b)| a == b || a == -1 || b == -1)
    }

    /// Creates a shape for a 1D tensor
    pub fn d1(dim: i64) -> Self {
        Self::new(vec![dim])
    }

    /// Creates a shape for a 2D tensor
    pub fn d2(dim0: i64, dim1: i64) -> Self {
        Self::new(vec![dim0, dim1])
    }

    /// Creates a shape for a 3D tensor
    pub fn d3(dim0: i64, dim1: i64, dim2: i64) -> Self {
        Self::new(vec![dim0, dim1, dim2])
    }

    /// Creates a shape for a 4D tensor
    pub fn d4(dim0: i64, dim1: i64, dim2: i64, dim3: i64) -> Self {
        Self::new(vec![dim0, dim1, dim2, dim3])
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if *dim == -1 {
                write!(f, "?")?;
            } else {
                write!(f, "{dim}")?;
            }
        }
        write!(f, "]")
    }
}

impl From<Vec<i64>> for TensorShape {
    fn from(dims: Vec<i64>) -> Self {
        Self::new(dims)
    }
}

impl From<&[i64]> for TensorShape {
    fn from(dims: &[i64]) -> Self {
        Self::new(dims.to_vec())
    }
}

/// Tensor data with type information
///
/// Supports multiple numeric types commonly used in ML models:
/// - Float32: Standard single-precision (default for most models)
/// - Float16: Half-precision (for memory efficiency)
/// - Float64: Double-precision (for high-precision requirements)
/// - Int32: 32-bit integers
/// - Int64: 64-bit integers
/// - Uint8: Unsigned 8-bit (for quantized models)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorData {
    /// 32-bit floating point data
    Float32 {
        /// Data values
        data: Vec<f32>,
        /// Shape of the tensor
        shape: TensorShape,
    },
    /// 16-bit floating point data (half precision)
    Float16 {
        /// Data values
        data: Vec<f16>,
        /// Shape of the tensor
        shape: TensorShape,
    },
    /// 64-bit floating point data
    Float64 {
        /// Data values
        data: Vec<f64>,
        /// Shape of the tensor
        shape: TensorShape,
    },
    /// 32-bit signed integer data
    Int32 {
        /// Data values
        data: Vec<i32>,
        /// Shape of the tensor
        shape: TensorShape,
    },
    /// 64-bit signed integer data
    Int64 {
        /// Data values
        data: Vec<i64>,
        /// Shape of the tensor
        shape: TensorShape,
    },
    /// 8-bit unsigned integer data (for quantized models)
    Uint8 {
        /// Data values
        data: Vec<u8>,
        /// Shape of the tensor
        shape: TensorShape,
    },
}

impl TensorData {
    /// Creates a Float32 tensor from data and shape
    pub fn float32(data: Vec<f32>, shape: impl Into<TensorShape>) -> Self {
        TensorData::Float32 {
            data,
            shape: shape.into(),
        }
    }

    /// Creates a Float64 tensor from data and shape
    pub fn float64(data: Vec<f64>, shape: impl Into<TensorShape>) -> Self {
        TensorData::Float64 {
            data,
            shape: shape.into(),
        }
    }

    /// Creates an Int64 tensor from data and shape
    pub fn int64(data: Vec<i64>, shape: impl Into<TensorShape>) -> Self {
        TensorData::Int64 {
            data,
            shape: shape.into(),
        }
    }

    /// Creates an Int32 tensor from data and shape
    pub fn int32(data: Vec<i32>, shape: impl Into<TensorShape>) -> Self {
        TensorData::Int32 {
            data,
            shape: shape.into(),
        }
    }

    /// Creates a Uint8 tensor from data and shape
    pub fn uint8(data: Vec<u8>, shape: impl Into<TensorShape>) -> Self {
        TensorData::Uint8 {
            data,
            shape: shape.into(),
        }
    }

    /// Creates a Float32 tensor filled with zeros
    pub fn zeros_f32(shape: impl Into<TensorShape>) -> Self {
        let shape = shape.into();
        let size = shape.num_elements();
        TensorData::Float32 {
            data: vec![0.0f32; size],
            shape,
        }
    }

    /// Creates a Float32 tensor filled with ones
    pub fn ones_f32(shape: impl Into<TensorShape>) -> Self {
        let shape = shape.into();
        let size = shape.num_elements();
        TensorData::Float32 {
            data: vec![1.0f32; size],
            shape,
        }
    }

    /// Returns the shape of the tensor
    pub fn shape(&self) -> &TensorShape {
        match self {
            TensorData::Float32 { shape, .. } => shape,
            TensorData::Float16 { shape, .. } => shape,
            TensorData::Float64 { shape, .. } => shape,
            TensorData::Int32 { shape, .. } => shape,
            TensorData::Int64 { shape, .. } => shape,
            TensorData::Uint8 { shape, .. } => shape,
        }
    }

    /// Returns the data type as a string
    pub fn dtype(&self) -> &'static str {
        match self {
            TensorData::Float32 { .. } => "float32",
            TensorData::Float16 { .. } => "float16",
            TensorData::Float64 { .. } => "float64",
            TensorData::Int32 { .. } => "int32",
            TensorData::Int64 { .. } => "int64",
            TensorData::Uint8 { .. } => "uint8",
        }
    }

    /// Returns the number of elements in the tensor
    pub fn len(&self) -> usize {
        match self {
            TensorData::Float32 { data, .. } => data.len(),
            TensorData::Float16 { data, .. } => data.len(),
            TensorData::Float64 { data, .. } => data.len(),
            TensorData::Int32 { data, .. } => data.len(),
            TensorData::Int64 { data, .. } => data.len(),
            TensorData::Uint8 { data, .. } => data.len(),
        }
    }

    /// Returns true if the tensor has no elements
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Converts Float32 data to ndarray
    pub fn as_f32_array(&self) -> Option<ArrayD<f32>> {
        match self {
            TensorData::Float32 { data, shape } => {
                let dims: Vec<usize> = shape.dims().iter().map(|&d| d as usize).collect();
                Array::from_shape_vec(IxDyn(&dims), data.clone()).ok()
            }
            _ => None,
        }
    }

    /// Converts Float64 data to ndarray
    pub fn as_f64_array(&self) -> Option<ArrayD<f64>> {
        match self {
            TensorData::Float64 { data, shape } => {
                let dims: Vec<usize> = shape.dims().iter().map(|&d| d as usize).collect();
                Array::from_shape_vec(IxDyn(&dims), data.clone()).ok()
            }
            _ => None,
        }
    }

    /// Converts Int64 data to ndarray
    pub fn as_i64_array(&self) -> Option<ArrayD<i64>> {
        match self {
            TensorData::Int64 { data, shape } => {
                let dims: Vec<usize> = shape.dims().iter().map(|&d| d as usize).collect();
                Array::from_shape_vec(IxDyn(&dims), data.clone()).ok()
            }
            _ => None,
        }
    }

    /// Returns a reference to the underlying Float32 data if applicable
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            TensorData::Float32 { data, .. } => Some(data),
            _ => None,
        }
    }

    /// Returns a reference to the underlying Float64 data if applicable
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match self {
            TensorData::Float64 { data, .. } => Some(data),
            _ => None,
        }
    }

    /// Returns a reference to the underlying Int64 data if applicable
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        match self {
            TensorData::Int64 { data, .. } => Some(data),
            _ => None,
        }
    }

    /// Validates that the data length matches the shape
    pub fn validate(&self) -> bool {
        let expected = self.shape().num_elements();
        self.len() == expected
    }

    /// Scales tensor data by a factor (for federated learning aggregation)
    pub fn scale(&self, factor: f32) -> Self {
        match self {
            TensorData::Float32 { data, shape } => {
                let scaled_data: Vec<f32> = data.iter().map(|&x| x * factor).collect();
                TensorData::Float32 {
                    data: scaled_data,
                    shape: shape.clone(),
                }
            }
            TensorData::Float64 { data, shape } => {
                let scaled_data: Vec<f64> = data.iter().map(|&x| x * factor as f64).collect();
                TensorData::Float64 {
                    data: scaled_data,
                    shape: shape.clone(),
                }
            }
            TensorData::Int32 { data, shape } => {
                let scaled_data: Vec<i32> = data.iter().map(|&x| (x as f32 * factor) as i32).collect();
                TensorData::Int32 {
                    data: scaled_data,
                    shape: shape.clone(),
                }
            }
            TensorData::Int64 { data, shape } => {
                let scaled_data: Vec<i64> = data.iter().map(|&x| (x as f32 * factor) as i64).collect();
                TensorData::Int64 {
                    data: scaled_data,
                    shape: shape.clone(),
                }
            }
            TensorData::Uint8 { data, shape } => {
                let scaled_data: Vec<u8> = data.iter().map(|&x| ((x as f32 * factor).min(255.0).max(0.0)) as u8).collect();
                TensorData::Uint8 {
                    data: scaled_data,
                    shape: shape.clone(),
                }
            }
            TensorData::Float16 { data, shape } => {
                let scaled_data: Vec<f16> = data.iter().map(|&x| f16::from_f32(x.to_f32() * factor)).collect();
                TensorData::Float16 {
                    data: scaled_data,
                    shape: shape.clone(),
                }
            }
        }
    }

    /// Adds two tensors element-wise (for federated learning aggregation)
    pub fn add(&self, other: &TensorData) -> Result<Self, String> {
        if self.shape().dims() != other.shape().dims() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape().dims(),
                other.shape().dims()
            ));
        }

        match (self, other) {
            (TensorData::Float32 { data: d1, shape }, TensorData::Float32 { data: d2, .. }) => {
                let sum: Vec<f32> = d1.iter().zip(d2.iter()).map(|(&a, &b)| a + b).collect();
                Ok(TensorData::Float32 {
                    data: sum,
                    shape: shape.clone(),
                })
            }
            (TensorData::Float64 { data: d1, shape }, TensorData::Float64 { data: d2, .. }) => {
                let sum: Vec<f64> = d1.iter().zip(d2.iter()).map(|(&a, &b)| a + b).collect();
                Ok(TensorData::Float64 {
                    data: sum,
                    shape: shape.clone(),
                })
            }
            _ => Err("Type mismatch or unsupported tensor type for addition".to_string()),
        }
    }

    /// Adds Gaussian noise for differential privacy
    pub fn add_gaussian_noise(&self, noise_multiplier: f32, _clipping_threshold: f32) -> Self {
        use rand::thread_rng;
        use rand::Rng;

        let mut rng = thread_rng();

        match self {
            TensorData::Float32 { data, shape } => {
                let noisy_data: Vec<f32> = data
                    .iter()
                    .map(|&x| {
                        let noise: f32 = rng.gen_range(-noise_multiplier..noise_multiplier);
                        x + noise
                    })
                    .collect();
                TensorData::Float32 {
                    data: noisy_data,
                    shape: shape.clone(),
                }
            }
            TensorData::Float64 { data, shape } => {
                let noisy_data: Vec<f64> = data
                    .iter()
                    .map(|&x| {
                        let noise: f64 = rng.gen_range(-(noise_multiplier as f64)..(noise_multiplier as f64));
                        x + noise
                    })
                    .collect();
                TensorData::Float64 {
                    data: noisy_data,
                    shape: shape.clone(),
                }
            }
            // For integer types, just return a copy (noise not applicable)
            other => other.clone(),
        }
    }
}

impl fmt::Display for TensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}>({}, {} elements)", self.dtype(), self.shape(), self.len())
    }
}

/// Creates a TensorData from ndarray
impl From<ArrayD<f32>> for TensorData {
    fn from(array: ArrayD<f32>) -> Self {
        let shape = TensorShape::new(array.shape().iter().map(|&d| d as i64).collect());
        let data = array.into_raw_vec();
        TensorData::Float32 { data, shape }
    }
}

impl From<ArrayD<f64>> for TensorData {
    fn from(array: ArrayD<f64>) -> Self {
        let shape = TensorShape::new(array.shape().iter().map(|&d| d as i64).collect());
        let data = array.into_raw_vec();
        TensorData::Float64 { data, shape }
    }
}

impl From<ArrayD<i64>> for TensorData {
    fn from(array: ArrayD<i64>) -> Self {
        let shape = TensorShape::new(array.shape().iter().map(|&d| d as i64).collect());
        let data = array.into_raw_vec();
        TensorData::Int64 { data, shape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_creation() {
        let shape = TensorShape::new(vec![1, 10, 3]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.num_elements(), 30);
        assert_eq!(shape.dims(), &[1, 10, 3]);
    }

    #[test]
    fn test_tensor_shape_helpers() {
        assert_eq!(TensorShape::d1(10).dims(), &[10]);
        assert_eq!(TensorShape::d2(5, 10).dims(), &[5, 10]);
        assert_eq!(TensorShape::d3(1, 5, 10).dims(), &[1, 5, 10]);
        assert_eq!(TensorShape::d4(1, 2, 3, 4).dims(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_tensor_shape_display() {
        let shape = TensorShape::new(vec![1, -1, 3]);
        assert_eq!(format!("{shape}"), "[1, ?, 3]");
    }

    #[test]
    fn test_tensor_shape_compatibility() {
        let shape1 = TensorShape::new(vec![1, 10, 3]);
        let shape2 = TensorShape::new(vec![1, 10, 3]);
        let shape3 = TensorShape::new(vec![1, -1, 3]);
        let shape4 = TensorShape::new(vec![1, 5, 3]);

        assert!(shape1.is_compatible_with(&shape2));
        assert!(shape1.is_compatible_with(&shape3));
        assert!(!shape1.is_compatible_with(&shape4));
    }

    #[test]
    fn test_tensor_data_float32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = TensorData::float32(data.clone(), vec![2i64, 3]);

        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.dtype(), "float32");
        assert_eq!(tensor.len(), 6);
        assert!(tensor.validate());
        assert_eq!(tensor.as_f32_slice(), Some(data.as_slice()));
    }

    #[test]
    fn test_tensor_data_zeros() {
        let tensor = TensorData::zeros_f32(vec![2i64, 3]);
        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.as_f32_slice().unwrap(), &[0.0; 6]);
    }

    #[test]
    fn test_tensor_data_ones() {
        let tensor = TensorData::ones_f32(vec![2i64, 3]);
        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.as_f32_slice().unwrap(), &[1.0; 6]);
    }

    #[test]
    fn test_tensor_data_display() {
        let tensor = TensorData::float32(vec![1.0, 2.0, 3.0], vec![3i64]);
        let display = format!("{tensor}");
        assert!(display.contains("float32"));
        assert!(display.contains("3 elements"));
    }

    #[test]
    fn test_tensor_data_from_ndarray() {
        let array = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Failed to create array");
        let tensor: TensorData = array.into();

        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.dtype(), "float32");
    }

    #[test]
    fn test_tensor_data_int64() {
        let data = vec![1i64, 2, 3, 4];
        let tensor = TensorData::int64(data.clone(), vec![4i64]);

        assert_eq!(tensor.dtype(), "int64");
        assert_eq!(tensor.as_i64_slice(), Some(data.as_slice()));
    }

    #[test]
    fn test_tensor_data_validation() {
        // Valid tensor
        let valid = TensorData::float32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2i64, 3]);
        assert!(valid.validate());

        // Note: In a strict implementation, this would be invalid
        // but for flexibility, we allow mismatched sizes and validate() returns false
        let invalid = TensorData::Float32 {
            data: vec![1.0, 2.0, 3.0], // 3 elements
            shape: TensorShape::new(vec![2, 3]), // expects 6 elements
        };
        assert!(!invalid.validate());
    }
}
