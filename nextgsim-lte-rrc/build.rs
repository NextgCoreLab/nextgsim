//! Build script for E-UTRA RRC code generation from the TS 36.331 ASN.1 schema.
//!
//! Same generator as `nextgsim-rrc` (NR RRC): RRC is encoded with the BASIC-PER
//! Unaligned variant (TS 36.331 §8.1).

use std::env;
use std::path::{Path, PathBuf};

use asn1_compiler::generator::{Codec, Derive, Visibility};
use asn1_compiler::Asn1Compiler;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("value expected"));
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("value expected"));
    let schema = manifest_dir
        .parent()
        .unwrap_or(Path::new("."))
        .join("tools/lte-rrc-19.4.0.asn");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", schema.display());

    let output_file = out_dir.join("lte_rrc.rs");
    let mut compiler = Asn1Compiler::new(
        output_file.to_str().expect("value expected"),
        &Visibility::Public,
        vec![Codec::Uper],
        vec![Derive::Debug, Derive::Clone, Derive::PartialEq],
    );
    if let Err(e) = compiler.compile_files(&[schema.to_str().expect("value expected")]) {
        panic!("Failed to compile E-UTRA RRC ASN.1 schema: {e}");
    }
}
