//! Milenage algorithm implementation (3GPP TS 35.206)
//!
//! Milenage is the 3GPP authentication and key generation algorithm used in
//! UMTS, LTE, and 5G networks. It provides the following functions:
//! - f1: Network authentication (MAC-A)
//! - f1*: Re-synchronization authentication (MAC-S)
//! - f2: User authentication (RES/XRES)
//! - f3: Cipher key derivation (CK)
//! - f4: Integrity key derivation (IK)
//! - f5: Anonymity key derivation (AK)
//! - f5*: Re-synchronization anonymity key (AK for AUTS)
//!
//! Reference: 3GPP TS 35.206 V17.0.0

use crate::aes::{Aes128Block, xor_block, BLOCK_SIZE};

/// Key size in bytes (128 bits)
pub const KEY_SIZE: usize = 16;

/// OP/OPc size in bytes (128 bits)
pub const OP_SIZE: usize = 16;

/// RAND size in bytes (128 bits)
pub const RAND_SIZE: usize = 16;

/// SQN size in bytes (48 bits)
pub const SQN_SIZE: usize = 6;

/// AMF size in bytes (16 bits)
pub const AMF_SIZE: usize = 2;

/// MAC size in bytes (64 bits)
pub const MAC_SIZE: usize = 8;

/// RES size in bytes (64 bits)
pub const RES_SIZE: usize = 8;

/// CK size in bytes (128 bits)
pub const CK_SIZE: usize = 16;

/// IK size in bytes (128 bits)
pub const IK_SIZE: usize = 16;

/// AK size in bytes (48 bits)
pub const AK_SIZE: usize = 6;

/// Milenage constants for rotation and XOR operations
/// c1 = 0x00...00 (all zeros)
const C1: [u8; BLOCK_SIZE] = [0x00; BLOCK_SIZE];
/// c2 = 0x00...01
const C2: [u8; BLOCK_SIZE] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
];
/// c3 = 0x00...02
const C3: [u8; BLOCK_SIZE] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
];
/// c4 = 0x00...04
const C4: [u8; BLOCK_SIZE] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04,
];
/// c5 = 0x00...08
const C5: [u8; BLOCK_SIZE] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08,
];

/// Rotation amounts for each function
const R1: usize = 64;  // r1 = 64 bits
const R2: usize = 0;   // r2 = 0 bits
const R3: usize = 32;  // r3 = 32 bits
const R4: usize = 64;  // r4 = 64 bits
const R5: usize = 96;  // r5 = 96 bits

/// Rotate a 128-bit block left by `bits` positions
fn rotate_left(block: &[u8; BLOCK_SIZE], bits: usize) -> [u8; BLOCK_SIZE] {
    if bits == 0 || bits >= 128 {
        return *block;
    }
    
    let byte_shift = bits / 8;
    let bit_shift = bits % 8;
    let mut result = [0u8; BLOCK_SIZE];
    
    for i in 0..BLOCK_SIZE {
        let src_idx = (i + byte_shift) % BLOCK_SIZE;
        let next_idx = (i + byte_shift + 1) % BLOCK_SIZE;
        
        if bit_shift == 0 {
            result[i] = block[src_idx];
        } else {
            result[i] = (block[src_idx] << bit_shift) | (block[next_idx] >> (8 - bit_shift));
        }
    }
    
    result
}

/// Compute OPc from OP and K
///
/// OPc = OP XOR E_K(OP)
///
/// # Arguments
/// * `k` - 128-bit subscriber key
/// * `op` - 128-bit operator variant algorithm configuration field
///
/// # Returns
/// 128-bit OPc value
pub fn compute_opc(k: &[u8; KEY_SIZE], op: &[u8; OP_SIZE]) -> [u8; OP_SIZE] {
    let cipher = Aes128Block::new(k);
    let encrypted = cipher.encrypt_block_copy(op);
    
    let mut opc = [0u8; OP_SIZE];
    for i in 0..OP_SIZE {
        opc[i] = op[i] ^ encrypted[i];
    }
    opc
}

/// Milenage algorithm context
pub struct Milenage {
    cipher: Aes128Block,
    opc: [u8; OP_SIZE],
}

impl Milenage {
    /// Create a new Milenage instance with K and OPc
    ///
    /// # Arguments
    /// * `k` - 128-bit subscriber key
    /// * `opc` - 128-bit OPc (pre-computed from OP)
    pub fn new(k: &[u8; KEY_SIZE], opc: &[u8; OP_SIZE]) -> Self {
        Self {
            cipher: Aes128Block::new(k),
            opc: *opc,
        }
    }

    /// Create a new Milenage instance with K and OP (computes OPc internally)
    ///
    /// # Arguments
    /// * `k` - 128-bit subscriber key
    /// * `op` - 128-bit OP
    pub fn new_with_op(k: &[u8; KEY_SIZE], op: &[u8; OP_SIZE]) -> Self {
        let opc = compute_opc(k, op);
        Self::new(k, &opc)
    }

    /// Compute TEMP = E_K(RAND XOR OPc)
    fn compute_temp(&self, rand: &[u8; RAND_SIZE]) -> [u8; BLOCK_SIZE] {
        let mut temp = [0u8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            temp[i] = rand[i] ^ self.opc[i];
        }
        self.cipher.encrypt_block(&mut temp);
        temp
    }

    /// Compute OUT1 for f1/f1* (MAC-A/MAC-S)
    /// 
    /// Per 3GPP TS 35.206 and reference implementation:
    /// OUT1 = E_K(TEMP XOR rot(IN1 XOR OPc, r1) XOR c1) XOR OPc
    fn compute_out1(&self, rand: &[u8; RAND_SIZE], sqn: &[u8; SQN_SIZE], amf: &[u8; AMF_SIZE]) -> [u8; BLOCK_SIZE] {
        let temp = self.compute_temp(rand);
        
        // IN1 = SQN || AMF || SQN || AMF
        let mut in1 = [0u8; BLOCK_SIZE];
        in1[0..6].copy_from_slice(sqn);
        in1[6..8].copy_from_slice(amf);
        in1[8..14].copy_from_slice(sqn);
        in1[14..16].copy_from_slice(amf);
        
        // Step 1: IN1 XOR OPc
        let mut block = [0u8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            block[i] = in1[i] ^ self.opc[i];
        }
        
        // Step 2: rotate by r1 bits
        let rotated = rotate_left(&block, R1);
        
        // Step 3: XOR with TEMP and c1
        let mut block = rotated;
        xor_block(&mut block, &temp);
        xor_block(&mut block, &C1);
        
        // Step 4: Encrypt
        self.cipher.encrypt_block(&mut block);
        
        // Step 5: XOR with OPc
        xor_block(&mut block, &self.opc);
        
        block
    }

    /// f1 - Network authentication function
    ///
    /// Computes MAC-A = f1(K, RAND, SQN, AMF)
    ///
    /// # Arguments
    /// * `rand` - 128-bit random challenge
    /// * `sqn` - 48-bit sequence number
    /// * `amf` - 16-bit authentication management field
    ///
    /// # Returns
    /// 64-bit MAC-A
    pub fn f1(&self, rand: &[u8; RAND_SIZE], sqn: &[u8; SQN_SIZE], amf: &[u8; AMF_SIZE]) -> [u8; MAC_SIZE] {
        let out1 = self.compute_out1(rand, sqn, amf);
        let mut mac_a = [0u8; MAC_SIZE];
        mac_a.copy_from_slice(&out1[0..8]);
        mac_a
    }

    /// f1* - Re-synchronization authentication function
    ///
    /// Computes MAC-S = f1*(K, RAND, SQN, AMF)
    ///
    /// # Arguments
    /// * `rand` - 128-bit random challenge
    /// * `sqn` - 48-bit sequence number
    /// * `amf` - 16-bit authentication management field
    ///
    /// # Returns
    /// 64-bit MAC-S
    pub fn f1_star(&self, rand: &[u8; RAND_SIZE], sqn: &[u8; SQN_SIZE], amf: &[u8; AMF_SIZE]) -> [u8; MAC_SIZE] {
        let out1 = self.compute_out1(rand, sqn, amf);
        let mut mac_s = [0u8; MAC_SIZE];
        mac_s.copy_from_slice(&out1[8..16]);
        mac_s
    }

    /// Compute OUT2 for f2/f5 (RES/AK)
    ///
    /// Per 3GPP TS 35.206:
    /// OUT2 = E_K(rotate(TEMP XOR OPc, r2) XOR c2) XOR OPc
    fn compute_out2(&self, rand: &[u8; RAND_SIZE]) -> [u8; BLOCK_SIZE] {
        let temp = self.compute_temp(rand);
        
        // Step 1: TEMP XOR OPc
        let mut block = [0u8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            block[i] = temp[i] ^ self.opc[i];
        }
        
        // Step 2: rotate by r2 bits
        let rotated = rotate_left(&block, R2);
        
        // Step 3: XOR with c2
        let mut block = rotated;
        xor_block(&mut block, &C2);
        
        // Step 4: Encrypt
        self.cipher.encrypt_block(&mut block);
        
        // Step 5: XOR with OPc
        xor_block(&mut block, &self.opc);
        
        block
    }

    /// f2 - User authentication function
    ///
    /// Computes RES = f2(K, RAND)
    ///
    /// # Arguments
    /// * `rand` - 128-bit random challenge
    ///
    /// # Returns
    /// 64-bit RES (can be extended to 128 bits if needed)
    pub fn f2(&self, rand: &[u8; RAND_SIZE]) -> [u8; RES_SIZE] {
        let out2 = self.compute_out2(rand);
        let mut res = [0u8; RES_SIZE];
        res.copy_from_slice(&out2[8..16]);
        res
    }

    /// f5 - Anonymity key derivation function
    ///
    /// Computes AK = f5(K, RAND)
    ///
    /// # Arguments
    /// * `rand` - 128-bit random challenge
    ///
    /// # Returns
    /// 48-bit AK
    pub fn f5(&self, rand: &[u8; RAND_SIZE]) -> [u8; AK_SIZE] {
        let out2 = self.compute_out2(rand);
        let mut ak = [0u8; AK_SIZE];
        ak.copy_from_slice(&out2[0..6]);
        ak
    }

    /// Compute OUT3 for f3 (CK)
    ///
    /// Per 3GPP TS 35.206:
    /// OUT3 = E_K(rotate(TEMP XOR OPc, r3) XOR c3) XOR OPc
    fn compute_out3(&self, rand: &[u8; RAND_SIZE]) -> [u8; BLOCK_SIZE] {
        let temp = self.compute_temp(rand);
        
        // Step 1: TEMP XOR OPc
        let mut block = [0u8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            block[i] = temp[i] ^ self.opc[i];
        }
        
        // Step 2: rotate by r3 bits
        let rotated = rotate_left(&block, R3);
        
        // Step 3: XOR with c3
        let mut block = rotated;
        xor_block(&mut block, &C3);
        
        // Step 4: Encrypt
        self.cipher.encrypt_block(&mut block);
        
        // Step 5: XOR with OPc
        xor_block(&mut block, &self.opc);
        
        block
    }

    /// f3 - Cipher key derivation function
    ///
    /// Computes CK = f3(K, RAND)
    ///
    /// # Arguments
    /// * `rand` - 128-bit random challenge
    ///
    /// # Returns
    /// 128-bit CK
    pub fn f3(&self, rand: &[u8; RAND_SIZE]) -> [u8; CK_SIZE] {
        self.compute_out3(rand)
    }

    /// Compute OUT4 for f4 (IK)
    ///
    /// Per 3GPP TS 35.206:
    /// OUT4 = E_K(rotate(TEMP XOR OPc, r4) XOR c4) XOR OPc
    fn compute_out4(&self, rand: &[u8; RAND_SIZE]) -> [u8; BLOCK_SIZE] {
        let temp = self.compute_temp(rand);
        
        // Step 1: TEMP XOR OPc
        let mut block = [0u8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            block[i] = temp[i] ^ self.opc[i];
        }
        
        // Step 2: rotate by r4 bits
        let rotated = rotate_left(&block, R4);
        
        // Step 3: XOR with c4
        let mut block = rotated;
        xor_block(&mut block, &C4);
        
        // Step 4: Encrypt
        self.cipher.encrypt_block(&mut block);
        
        // Step 5: XOR with OPc
        xor_block(&mut block, &self.opc);
        
        block
    }

    /// f4 - Integrity key derivation function
    ///
    /// Computes IK = f4(K, RAND)
    ///
    /// # Arguments
    /// * `rand` - 128-bit random challenge
    ///
    /// # Returns
    /// 128-bit IK
    pub fn f4(&self, rand: &[u8; RAND_SIZE]) -> [u8; IK_SIZE] {
        self.compute_out4(rand)
    }

    /// Compute OUT5 for f5* (AK for resync)
    ///
    /// Per 3GPP TS 35.206:
    /// OUT5 = E_K(rotate(TEMP XOR OPc, r5) XOR c5) XOR OPc
    fn compute_out5(&self, rand: &[u8; RAND_SIZE]) -> [u8; BLOCK_SIZE] {
        let temp = self.compute_temp(rand);
        
        // Step 1: TEMP XOR OPc
        let mut block = [0u8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            block[i] = temp[i] ^ self.opc[i];
        }
        
        // Step 2: rotate by r5 bits
        let rotated = rotate_left(&block, R5);
        
        // Step 3: XOR with c5
        let mut block = rotated;
        xor_block(&mut block, &C5);
        
        // Step 4: Encrypt
        self.cipher.encrypt_block(&mut block);
        
        // Step 5: XOR with OPc
        xor_block(&mut block, &self.opc);
        
        block
    }

    /// f5* - Re-synchronization anonymity key derivation function
    ///
    /// Computes AK = f5*(K, RAND)
    ///
    /// # Arguments
    /// * `rand` - 128-bit random challenge
    ///
    /// # Returns
    /// 48-bit AK for re-synchronization
    pub fn f5_star(&self, rand: &[u8; RAND_SIZE]) -> [u8; AK_SIZE] {
        let out5 = self.compute_out5(rand);
        let mut ak = [0u8; AK_SIZE];
        ak.copy_from_slice(&out5[0..6]);
        ak
    }

    /// Compute all authentication vectors at once
    ///
    /// This is more efficient than calling individual functions when all
    /// values are needed.
    ///
    /// # Arguments
    /// * `rand` - 128-bit random challenge
    /// * `sqn` - 48-bit sequence number
    /// * `amf` - 16-bit authentication management field
    ///
    /// # Returns
    /// Tuple of (MAC-A, RES, CK, IK, AK)
    pub fn compute_all(
        &self,
        rand: &[u8; RAND_SIZE],
        sqn: &[u8; SQN_SIZE],
        amf: &[u8; AMF_SIZE],
    ) -> ([u8; MAC_SIZE], [u8; RES_SIZE], [u8; CK_SIZE], [u8; IK_SIZE], [u8; AK_SIZE]) {
        let mac_a = self.f1(rand, sqn, amf);
        let res = self.f2(rand);
        let ck = self.f3(rand);
        let ik = self.f4(rand);
        let ak = self.f5(rand);
        
        (mac_a, res, ck, ik, ak)
    }
}

/// Convenience function to compute OPc
pub fn milenage_opc(k: &[u8; KEY_SIZE], op: &[u8; OP_SIZE]) -> [u8; OP_SIZE] {
    compute_opc(k, op)
}

/// Convenience function to compute f1 (MAC-A)
pub fn milenage_f1(
    k: &[u8; KEY_SIZE],
    opc: &[u8; OP_SIZE],
    rand: &[u8; RAND_SIZE],
    sqn: &[u8; SQN_SIZE],
    amf: &[u8; AMF_SIZE],
) -> [u8; MAC_SIZE] {
    Milenage::new(k, opc).f1(rand, sqn, amf)
}

/// Convenience function to compute f1* (MAC-S)
pub fn milenage_f1_star(
    k: &[u8; KEY_SIZE],
    opc: &[u8; OP_SIZE],
    rand: &[u8; RAND_SIZE],
    sqn: &[u8; SQN_SIZE],
    amf: &[u8; AMF_SIZE],
) -> [u8; MAC_SIZE] {
    Milenage::new(k, opc).f1_star(rand, sqn, amf)
}

/// Convenience function to compute f2, f3, f4, f5
pub fn milenage_f2345(
    k: &[u8; KEY_SIZE],
    opc: &[u8; OP_SIZE],
    rand: &[u8; RAND_SIZE],
) -> ([u8; RES_SIZE], [u8; CK_SIZE], [u8; IK_SIZE], [u8; AK_SIZE]) {
    let m = Milenage::new(k, opc);
    (m.f2(rand), m.f3(rand), m.f4(rand), m.f5(rand))
}

/// Convenience function to compute f5* (AK for resync)
pub fn milenage_f5_star(
    k: &[u8; KEY_SIZE],
    opc: &[u8; OP_SIZE],
    rand: &[u8; RAND_SIZE],
) -> [u8; AK_SIZE] {
    Milenage::new(k, opc).f5_star(rand)
}


#[cfg(test)]
mod tests {
    use super::*;

    /// 3GPP TS 35.207 Test Set 1
    #[test]
    fn test_milenage_3gpp_test_set_1() {
        let k: [u8; 16] = [
            0x46, 0x5b, 0x5c, 0xe8, 0xb1, 0x99, 0xb4, 0x9f,
            0xaa, 0x5f, 0x0a, 0x2e, 0xe2, 0x38, 0xa6, 0xbc,
        ];
        let rand: [u8; 16] = [
            0x23, 0x55, 0x3c, 0xbe, 0x96, 0x37, 0xa8, 0x9d,
            0x21, 0x8a, 0xe6, 0x4d, 0xae, 0x47, 0xbf, 0x35,
        ];
        let sqn: [u8; 6] = [0xff, 0x9b, 0xb4, 0xd0, 0xb6, 0x07];
        let amf: [u8; 2] = [0xb9, 0xb9];
        let op: [u8; 16] = [
            0xcd, 0xc2, 0x02, 0xd5, 0x12, 0x3e, 0x20, 0xf6,
            0x2b, 0x6d, 0x67, 0x6a, 0xc7, 0x2c, 0xb3, 0x18,
        ];

        // Expected OPc
        let expected_opc: [u8; 16] = [
            0xcd, 0x63, 0xcb, 0x71, 0x95, 0x4a, 0x9f, 0x4e,
            0x48, 0xa5, 0x99, 0x4e, 0x37, 0xa0, 0x2b, 0xaf,
        ];

        // Expected outputs
        let expected_f1: [u8; 8] = [0x4a, 0x9f, 0xfa, 0xc3, 0x54, 0xdf, 0xaf, 0xb3];
        let expected_f1_star: [u8; 8] = [0x01, 0xcf, 0xaf, 0x9e, 0xc4, 0xe8, 0x71, 0xe9];
        let expected_f2: [u8; 8] = [0xa5, 0x42, 0x11, 0xd5, 0xe3, 0xba, 0x50, 0xbf];
        let expected_f3: [u8; 16] = [
            0xb4, 0x0b, 0xa9, 0xa3, 0xc5, 0x8b, 0x2a, 0x05,
            0xbb, 0xf0, 0xd9, 0x87, 0xb2, 0x1b, 0xf8, 0xcb,
        ];
        let expected_f4: [u8; 16] = [
            0xf7, 0x69, 0xbc, 0xd7, 0x51, 0x04, 0x46, 0x04,
            0x12, 0x76, 0x72, 0x71, 0x1c, 0x6d, 0x34, 0x41,
        ];
        let expected_f5: [u8; 6] = [0xaa, 0x68, 0x9c, 0x64, 0x83, 0x70];
        let expected_f5_star: [u8; 6] = [0x45, 0x1e, 0x8b, 0xec, 0xa4, 0x3b];

        // Compute OPc
        let opc = compute_opc(&k, &op);
        assert_eq!(opc, expected_opc, "OPc mismatch");

        // Create Milenage instance
        let m = Milenage::new(&k, &opc);

        // Test f1 (MAC-A)
        let f1 = m.f1(&rand, &sqn, &amf);
        assert_eq!(f1, expected_f1, "f1 (MAC-A) mismatch");

        // Test f1* (MAC-S)
        let f1_star = m.f1_star(&rand, &sqn, &amf);
        assert_eq!(f1_star, expected_f1_star, "f1* (MAC-S) mismatch");

        // Test f2 (RES)
        let f2 = m.f2(&rand);
        assert_eq!(f2, expected_f2, "f2 (RES) mismatch");

        // Test f3 (CK)
        let f3 = m.f3(&rand);
        assert_eq!(f3, expected_f3, "f3 (CK) mismatch");

        // Test f4 (IK)
        let f4 = m.f4(&rand);
        assert_eq!(f4, expected_f4, "f4 (IK) mismatch");

        // Test f5 (AK)
        let f5 = m.f5(&rand);
        assert_eq!(f5, expected_f5, "f5 (AK) mismatch");

        // Test f5* (AK for resync)
        let f5_star = m.f5_star(&rand);
        assert_eq!(f5_star, expected_f5_star, "f5* (AK resync) mismatch");
    }


    /// 3GPP TS 35.207 Test Set 2
    #[test]
    fn test_milenage_3gpp_test_set_2() {
        let k: [u8; 16] = [
            0x46, 0x5b, 0x5c, 0xe8, 0xb1, 0x99, 0xb4, 0x9f,
            0xaa, 0x5f, 0x0a, 0x2e, 0xe2, 0x38, 0xa6, 0xbc,
        ];
        let rand: [u8; 16] = [
            0x23, 0x55, 0x3c, 0xbe, 0x96, 0x37, 0xa8, 0x9d,
            0x21, 0x8a, 0xe6, 0x4d, 0xae, 0x47, 0xbf, 0x35,
        ];
        let sqn: [u8; 6] = [0xff, 0x9b, 0xb4, 0xd0, 0xb6, 0x07];
        let amf: [u8; 2] = [0xb9, 0xb9];
        let op: [u8; 16] = [
            0xcd, 0xc2, 0x02, 0xd5, 0x12, 0x3e, 0x20, 0xf6,
            0x2b, 0x6d, 0x67, 0x6a, 0xc7, 0x2c, 0xb3, 0x18,
        ];

        // Expected OPc
        let expected_opc: [u8; 16] = [
            0xcd, 0x63, 0xcb, 0x71, 0x95, 0x4a, 0x9f, 0x4e,
            0x48, 0xa5, 0x99, 0x4e, 0x37, 0xa0, 0x2b, 0xaf,
        ];

        // Expected outputs (Test Set 1 values)
        let expected_f1: [u8; 8] = [0x4a, 0x9f, 0xfa, 0xc3, 0x54, 0xdf, 0xaf, 0xb3];
        let expected_f1_star: [u8; 8] = [0x01, 0xcf, 0xaf, 0x9e, 0xc4, 0xe8, 0x71, 0xe9];
        let expected_f2: [u8; 8] = [0xa5, 0x42, 0x11, 0xd5, 0xe3, 0xba, 0x50, 0xbf];
        let expected_f3: [u8; 16] = [
            0xb4, 0x0b, 0xa9, 0xa3, 0xc5, 0x8b, 0x2a, 0x05,
            0xbb, 0xf0, 0xd9, 0x87, 0xb2, 0x1b, 0xf8, 0xcb,
        ];
        let expected_f4: [u8; 16] = [
            0xf7, 0x69, 0xbc, 0xd7, 0x51, 0x04, 0x46, 0x04,
            0x12, 0x76, 0x72, 0x71, 0x1c, 0x6d, 0x34, 0x41,
        ];
        let expected_f5: [u8; 6] = [0xaa, 0x68, 0x9c, 0x64, 0x83, 0x70];
        let expected_f5_star: [u8; 6] = [0x45, 0x1e, 0x8b, 0xec, 0xa4, 0x3b];

        let opc = compute_opc(&k, &op);
        assert_eq!(opc, expected_opc, "OPc mismatch");

        let m = Milenage::new(&k, &opc);

        assert_eq!(m.f1(&rand, &sqn, &amf), expected_f1, "f1 mismatch");
        assert_eq!(m.f1_star(&rand, &sqn, &amf), expected_f1_star, "f1* mismatch");
        assert_eq!(m.f2(&rand), expected_f2, "f2 mismatch");
        assert_eq!(m.f3(&rand), expected_f3, "f3 mismatch");
        assert_eq!(m.f4(&rand), expected_f4, "f4 mismatch");
        assert_eq!(m.f5(&rand), expected_f5, "f5 mismatch");
        assert_eq!(m.f5_star(&rand), expected_f5_star, "f5* mismatch");
    }


    /// 3GPP TS 35.207 Test Set 3
    #[test]
    fn test_milenage_3gpp_test_set_3() {
        let k: [u8; 16] = [
            0xfe, 0xc8, 0x6b, 0xa6, 0xeb, 0x70, 0x7e, 0xd0,
            0x89, 0x05, 0x75, 0x7b, 0x1b, 0xb4, 0x4b, 0x8f,
        ];
        let rand: [u8; 16] = [
            0x9f, 0x7c, 0x8d, 0x02, 0x1a, 0xcc, 0xf4, 0xdb,
            0x21, 0x3c, 0xcf, 0xf0, 0xc7, 0xf7, 0x1a, 0x6a,
        ];
        let sqn: [u8; 6] = [0x9d, 0x02, 0x77, 0x59, 0x5f, 0xfc];
        let amf: [u8; 2] = [0x72, 0x5c];
        let op: [u8; 16] = [
            0xdb, 0xc5, 0x9a, 0xdc, 0xb6, 0xf9, 0xa0, 0xef,
            0x73, 0x54, 0x77, 0xb7, 0xfa, 0xdf, 0x83, 0x74,
        ];

        let expected_opc: [u8; 16] = [
            0x10, 0x06, 0x02, 0x0f, 0x0a, 0x47, 0x8b, 0xf6,
            0xb6, 0x99, 0xf1, 0x5c, 0x06, 0x2e, 0x42, 0xb3,
        ];

        let expected_f1: [u8; 8] = [0x9c, 0xab, 0xc3, 0xe9, 0x9b, 0xaf, 0x72, 0x81];
        let expected_f1_star: [u8; 8] = [0x95, 0x81, 0x4b, 0xa2, 0xb3, 0x04, 0x43, 0x24];
        let expected_f2: [u8; 8] = [0x80, 0x11, 0xc4, 0x8c, 0x0c, 0x21, 0x4e, 0xd2];
        let expected_f3: [u8; 16] = [
            0x5d, 0xbd, 0xbb, 0x29, 0x54, 0xe8, 0xf3, 0xcd,
            0xe6, 0x65, 0xb0, 0x46, 0x17, 0x9a, 0x50, 0x98,
        ];
        let expected_f4: [u8; 16] = [
            0x59, 0xa9, 0x2d, 0x3b, 0x47, 0x6a, 0x04, 0x43,
            0x48, 0x70, 0x55, 0xcf, 0x88, 0xb2, 0x30, 0x7b,
        ];
        let expected_f5: [u8; 6] = [0x33, 0x48, 0x4d, 0xc2, 0x13, 0x6b];
        let expected_f5_star: [u8; 6] = [0xde, 0xac, 0xdd, 0x84, 0x8c, 0xc6];

        let opc = compute_opc(&k, &op);
        assert_eq!(opc, expected_opc, "OPc mismatch");

        let m = Milenage::new(&k, &opc);

        assert_eq!(m.f1(&rand, &sqn, &amf), expected_f1, "f1 mismatch");
        assert_eq!(m.f1_star(&rand, &sqn, &amf), expected_f1_star, "f1* mismatch");
        assert_eq!(m.f2(&rand), expected_f2, "f2 mismatch");
        assert_eq!(m.f3(&rand), expected_f3, "f3 mismatch");
        assert_eq!(m.f4(&rand), expected_f4, "f4 mismatch");
        assert_eq!(m.f5(&rand), expected_f5, "f5 mismatch");
        assert_eq!(m.f5_star(&rand), expected_f5_star, "f5* mismatch");
    }


    /// 3GPP TS 35.207 Test Set 4
    #[test]
    fn test_milenage_3gpp_test_set_4() {
        let k: [u8; 16] = [
            0x9e, 0x59, 0x44, 0xae, 0xa9, 0x4b, 0x81, 0x16,
            0x5c, 0x82, 0xfb, 0xf9, 0xf3, 0x2d, 0xb7, 0x51,
        ];
        let rand: [u8; 16] = [
            0xce, 0x83, 0xdb, 0xc5, 0x4a, 0xc0, 0x27, 0x4a,
            0x15, 0x7c, 0x17, 0xf8, 0x0d, 0x01, 0x7b, 0xd6,
        ];
        let sqn: [u8; 6] = [0x0b, 0x60, 0x4a, 0x81, 0xec, 0xa8];
        let amf: [u8; 2] = [0x9e, 0x09];
        let op: [u8; 16] = [
            0x22, 0x30, 0x14, 0xc5, 0x80, 0x66, 0x94, 0xc0,
            0x07, 0xca, 0x1e, 0xee, 0xf5, 0x7f, 0x00, 0x4f,
        ];

        let expected_opc: [u8; 16] = [
            0xa6, 0x4a, 0x50, 0x7a, 0xe1, 0xa2, 0xa9, 0x8b,
            0xb8, 0x8e, 0xb4, 0x21, 0x01, 0x35, 0xdc, 0x87,
        ];

        let expected_f1: [u8; 8] = [0x74, 0xa5, 0x82, 0x20, 0xcb, 0xa8, 0x4c, 0x49];
        let expected_f1_star: [u8; 8] = [0xac, 0x2c, 0xc7, 0x4a, 0x96, 0x87, 0x18, 0x37];
        let expected_f2: [u8; 8] = [0xf3, 0x65, 0xcd, 0x68, 0x3c, 0xd9, 0x2e, 0x96];
        let expected_f3: [u8; 16] = [
            0xe2, 0x03, 0xed, 0xb3, 0x97, 0x15, 0x74, 0xf5,
            0xa9, 0x4b, 0x0d, 0x61, 0xb8, 0x16, 0x34, 0x5d,
        ];
        let expected_f4: [u8; 16] = [
            0x0c, 0x45, 0x24, 0xad, 0xea, 0xc0, 0x41, 0xc4,
            0xdd, 0x83, 0x0d, 0x20, 0x85, 0x4f, 0xc4, 0x6b,
        ];
        let expected_f5: [u8; 6] = [0xf0, 0xb9, 0xc0, 0x8a, 0xd0, 0x2e];
        let expected_f5_star: [u8; 6] = [0x60, 0x85, 0xa8, 0x6c, 0x6f, 0x63];

        let opc = compute_opc(&k, &op);
        assert_eq!(opc, expected_opc, "OPc mismatch");

        let m = Milenage::new(&k, &opc);

        assert_eq!(m.f1(&rand, &sqn, &amf), expected_f1, "f1 mismatch");
        assert_eq!(m.f1_star(&rand, &sqn, &amf), expected_f1_star, "f1* mismatch");
        assert_eq!(m.f2(&rand), expected_f2, "f2 mismatch");
        assert_eq!(m.f3(&rand), expected_f3, "f3 mismatch");
        assert_eq!(m.f4(&rand), expected_f4, "f4 mismatch");
        assert_eq!(m.f5(&rand), expected_f5, "f5 mismatch");
        assert_eq!(m.f5_star(&rand), expected_f5_star, "f5* mismatch");
    }


    /// 3GPP TS 35.207 Test Set 5
    #[test]
    fn test_milenage_3gpp_test_set_5() {
        let k: [u8; 16] = [
            0x4a, 0xb1, 0xde, 0xb0, 0x5c, 0xa6, 0xce, 0xb0,
            0x51, 0xfc, 0x98, 0xe7, 0x7d, 0x02, 0x6a, 0x84,
        ];
        let rand: [u8; 16] = [
            0x74, 0xb0, 0xcd, 0x60, 0x31, 0xa1, 0xc8, 0x33,
            0x9b, 0x2b, 0x6c, 0xe2, 0xb8, 0xc4, 0xa1, 0x86,
        ];
        let sqn: [u8; 6] = [0xe8, 0x80, 0xa1, 0xb5, 0x80, 0xb6];
        let amf: [u8; 2] = [0x9f, 0x07];
        let op: [u8; 16] = [
            0x2d, 0x16, 0xc5, 0xcd, 0x1f, 0xdf, 0x6b, 0x22,
            0x38, 0x35, 0x84, 0xe3, 0xbe, 0xf2, 0xa8, 0xd8,
        ];

        let expected_opc: [u8; 16] = [
            0xdc, 0xf0, 0x7c, 0xbd, 0x51, 0x85, 0x52, 0x90,
            0xb9, 0x2a, 0x07, 0xa9, 0x89, 0x1e, 0x52, 0x3e,
        ];

        let expected_f1: [u8; 8] = [0x49, 0xe7, 0x85, 0xdd, 0x12, 0x62, 0x6e, 0xf2];
        let expected_f1_star: [u8; 8] = [0x9e, 0x85, 0x79, 0x03, 0x36, 0xbb, 0x3f, 0xa2];
        let expected_f2: [u8; 8] = [0x58, 0x60, 0xfc, 0x1b, 0xce, 0x35, 0x1e, 0x7e];
        let expected_f3: [u8; 16] = [
            0x76, 0x57, 0x76, 0x6b, 0x37, 0x3d, 0x1c, 0x21,
            0x38, 0xf3, 0x07, 0xe3, 0xde, 0x92, 0x42, 0xf9,
        ];
        let expected_f4: [u8; 16] = [
            0x1c, 0x42, 0xe9, 0x60, 0xd8, 0x9b, 0x8f, 0xa9,
            0x9f, 0x27, 0x44, 0xe0, 0x70, 0x8c, 0xcb, 0x53,
        ];
        let expected_f5: [u8; 6] = [0x31, 0xe1, 0x1a, 0x60, 0x91, 0x18];
        let expected_f5_star: [u8; 6] = [0xfe, 0x25, 0x55, 0xe5, 0x4a, 0xa9];

        let opc = compute_opc(&k, &op);
        assert_eq!(opc, expected_opc, "OPc mismatch");

        let m = Milenage::new(&k, &opc);

        assert_eq!(m.f1(&rand, &sqn, &amf), expected_f1, "f1 mismatch");
        assert_eq!(m.f1_star(&rand, &sqn, &amf), expected_f1_star, "f1* mismatch");
        assert_eq!(m.f2(&rand), expected_f2, "f2 mismatch");
        assert_eq!(m.f3(&rand), expected_f3, "f3 mismatch");
        assert_eq!(m.f4(&rand), expected_f4, "f4 mismatch");
        assert_eq!(m.f5(&rand), expected_f5, "f5 mismatch");
        assert_eq!(m.f5_star(&rand), expected_f5_star, "f5* mismatch");
    }


    /// 3GPP TS 35.207 Test Set 6
    #[test]
    fn test_milenage_3gpp_test_set_6() {
        let k: [u8; 16] = [
            0x6c, 0x38, 0xa1, 0x16, 0xac, 0x28, 0x0c, 0x45,
            0x4f, 0x59, 0x33, 0x2e, 0xe3, 0x5c, 0x8c, 0x4f,
        ];
        let rand: [u8; 16] = [
            0xee, 0x64, 0x66, 0xbc, 0x96, 0x20, 0x2c, 0x5a,
            0x55, 0x7a, 0xbb, 0xef, 0xf8, 0xba, 0xbf, 0x63,
        ];
        let sqn: [u8; 6] = [0x41, 0x4b, 0x98, 0x22, 0x21, 0x81];
        let amf: [u8; 2] = [0x44, 0x64];
        let op: [u8; 16] = [
            0x1b, 0xa0, 0x0a, 0x1a, 0x7c, 0x67, 0x00, 0xac,
            0x8c, 0x3f, 0xf3, 0xe9, 0x6a, 0xd0, 0x87, 0x25,
        ];

        let expected_opc: [u8; 16] = [
            0x38, 0x03, 0xef, 0x53, 0x63, 0xb9, 0x47, 0xc6,
            0xaa, 0xa2, 0x25, 0xe5, 0x8f, 0xae, 0x39, 0x34,
        ];

        let expected_f1: [u8; 8] = [0x07, 0x8a, 0xdf, 0xb4, 0x88, 0x24, 0x1a, 0x57];
        let expected_f1_star: [u8; 8] = [0x80, 0x24, 0x6b, 0x8d, 0x01, 0x86, 0xbc, 0xf1];
        let expected_f2: [u8; 8] = [0x16, 0xc8, 0x23, 0x3f, 0x05, 0xa0, 0xac, 0x28];
        let expected_f3: [u8; 16] = [
            0x3f, 0x8c, 0x75, 0x87, 0xfe, 0x8e, 0x4b, 0x23,
            0x3a, 0xf6, 0x76, 0xae, 0xde, 0x30, 0xba, 0x3b,
        ];
        let expected_f4: [u8; 16] = [
            0xa7, 0x46, 0x6c, 0xc1, 0xe6, 0xb2, 0xa1, 0x33,
            0x7d, 0x49, 0xd3, 0xb6, 0x6e, 0x95, 0xd7, 0xb4,
        ];
        let expected_f5: [u8; 6] = [0x45, 0xb0, 0xf6, 0x9a, 0xb0, 0x6c];
        let expected_f5_star: [u8; 6] = [0x1f, 0x53, 0xcd, 0x2b, 0x11, 0x13];

        let opc = compute_opc(&k, &op);
        assert_eq!(opc, expected_opc, "OPc mismatch");

        let m = Milenage::new(&k, &opc);

        assert_eq!(m.f1(&rand, &sqn, &amf), expected_f1, "f1 mismatch");
        assert_eq!(m.f1_star(&rand, &sqn, &amf), expected_f1_star, "f1* mismatch");
        assert_eq!(m.f2(&rand), expected_f2, "f2 mismatch");
        assert_eq!(m.f3(&rand), expected_f3, "f3 mismatch");
        assert_eq!(m.f4(&rand), expected_f4, "f4 mismatch");
        assert_eq!(m.f5(&rand), expected_f5, "f5 mismatch");
        assert_eq!(m.f5_star(&rand), expected_f5_star, "f5* mismatch");
    }

    /// Test rotate_left function
    #[test]
    fn test_rotate_left() {
        let block: [u8; 16] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
        ];

        // Rotate by 0 should return same block
        assert_eq!(rotate_left(&block, 0), block);

        // Rotate by 64 bits (8 bytes)
        let rotated_64 = rotate_left(&block, 64);
        assert_eq!(rotated_64[0], 0x09);
        assert_eq!(rotated_64[8], 0x01);

        // Rotate by 32 bits (4 bytes)
        let rotated_32 = rotate_left(&block, 32);
        assert_eq!(rotated_32[0], 0x05);
        assert_eq!(rotated_32[12], 0x01);
    }

    /// Test convenience functions
    #[test]
    fn test_convenience_functions() {
        let k: [u8; 16] = [
            0x46, 0x5b, 0x5c, 0xe8, 0xb1, 0x99, 0xb4, 0x9f,
            0xaa, 0x5f, 0x0a, 0x2e, 0xe2, 0x38, 0xa6, 0xbc,
        ];
        let rand: [u8; 16] = [
            0x23, 0x55, 0x3c, 0xbe, 0x96, 0x37, 0xa8, 0x9d,
            0x21, 0x8a, 0xe6, 0x4d, 0xae, 0x47, 0xbf, 0x35,
        ];
        let sqn: [u8; 6] = [0xff, 0x9b, 0xb4, 0xd0, 0xb6, 0x07];
        let amf: [u8; 2] = [0xb9, 0xb9];
        let op: [u8; 16] = [
            0xcd, 0xc2, 0x02, 0xd5, 0x12, 0x3e, 0x20, 0xf6,
            0x2b, 0x6d, 0x67, 0x6a, 0xc7, 0x2c, 0xb3, 0x18,
        ];

        let opc = milenage_opc(&k, &op);
        let f1 = milenage_f1(&k, &opc, &rand, &sqn, &amf);
        let f1_star = milenage_f1_star(&k, &opc, &rand, &sqn, &amf);
        let (f2, f3, f4, f5) = milenage_f2345(&k, &opc, &rand);
        let f5_star = milenage_f5_star(&k, &opc, &rand);

        // Verify against Test Set 1 expected values
        assert_eq!(f1, [0x4a, 0x9f, 0xfa, 0xc3, 0x54, 0xdf, 0xaf, 0xb3]);
        assert_eq!(f1_star, [0x01, 0xcf, 0xaf, 0x9e, 0xc4, 0xe8, 0x71, 0xe9]);
        assert_eq!(f2, [0xa5, 0x42, 0x11, 0xd5, 0xe3, 0xba, 0x50, 0xbf]);
        assert_eq!(f5, [0xaa, 0x68, 0x9c, 0x64, 0x83, 0x70]);
        assert_eq!(f5_star, [0x45, 0x1e, 0x8b, 0xec, 0xa4, 0x3b]);
        assert_eq!(f3.len(), 16);
        assert_eq!(f4.len(), 16);
    }

    /// Test compute_all function
    #[test]
    fn test_compute_all() {
        let k: [u8; 16] = [
            0x46, 0x5b, 0x5c, 0xe8, 0xb1, 0x99, 0xb4, 0x9f,
            0xaa, 0x5f, 0x0a, 0x2e, 0xe2, 0x38, 0xa6, 0xbc,
        ];
        let opc: [u8; 16] = [
            0xcd, 0x63, 0xcb, 0x71, 0x95, 0x4a, 0x9f, 0x4e,
            0x48, 0xa5, 0x99, 0x4e, 0x37, 0xa0, 0x2b, 0xaf,
        ];
        let rand: [u8; 16] = [
            0x23, 0x55, 0x3c, 0xbe, 0x96, 0x37, 0xa8, 0x9d,
            0x21, 0x8a, 0xe6, 0x4d, 0xae, 0x47, 0xbf, 0x35,
        ];
        let sqn: [u8; 6] = [0xff, 0x9b, 0xb4, 0xd0, 0xb6, 0x07];
        let amf: [u8; 2] = [0xb9, 0xb9];

        let m = Milenage::new(&k, &opc);
        let (mac_a, res, ck, ik, ak) = m.compute_all(&rand, &sqn, &amf);

        assert_eq!(mac_a, [0x4a, 0x9f, 0xfa, 0xc3, 0x54, 0xdf, 0xaf, 0xb3]);
        assert_eq!(res, [0xa5, 0x42, 0x11, 0xd5, 0xe3, 0xba, 0x50, 0xbf]);
        assert_eq!(ak, [0xaa, 0x68, 0x9c, 0x64, 0x83, 0x70]);
        assert_eq!(ck.len(), 16);
        assert_eq!(ik.len(), 16);
    }
}

#[cfg(test)]
mod debug_tests {
    
    use crate::aes::Aes128Block;

    #[test]
    fn test_aes_encryption_for_opc() {
        // Test Set 1 values
        let k: [u8; 16] = [
            0x46, 0x5b, 0x5c, 0xe8, 0xb1, 0x99, 0xb4, 0x9f,
            0xaa, 0x5f, 0x0a, 0x2e, 0xe2, 0x38, 0xa6, 0xbc,
        ];
        let op: [u8; 16] = [
            0xcd, 0xc2, 0x02, 0xd5, 0x12, 0x3e, 0x20, 0xf6,
            0x2b, 0x6d, 0x67, 0x6a, 0xc7, 0x2c, 0xb3, 0x18,
        ];
        
        let cipher = Aes128Block::new(&k);
        let encrypted = cipher.encrypt_block_copy(&op);
        
        // Expected E_K(OP) from OpenSSL: 00a1c9a48774bfb863c8fe24f08c98b7
        let expected_encrypted: [u8; 16] = [
            0x00, 0xa1, 0xc9, 0xa4, 0x87, 0x74, 0xbf, 0xb8,
            0x63, 0xc8, 0xfe, 0x24, 0xf0, 0x8c, 0x98, 0xb7,
        ];
        
        eprintln!("K:         {k:02x?}");
        eprintln!("OP:        {op:02x?}");
        eprintln!("E_K(OP):   {encrypted:02x?}");
        eprintln!("Expected:  {expected_encrypted:02x?}");
        
        assert_eq!(encrypted, expected_encrypted, "AES encryption mismatch");
    }
}
