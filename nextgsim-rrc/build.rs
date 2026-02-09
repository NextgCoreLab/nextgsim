//! Build script for RRC code generation from ASN.1 schema

use std::env;
use std::path::PathBuf;

use asn1_compiler::generator::{Codec, Derive, Visibility};
use asn1_compiler::Asn1Compiler;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // Get the manifest directory (where Cargo.toml is)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let schema_path = manifest_dir.parent().unwrap().join("tools/rrc-15.6.0.asn1");

    // Rerun if schema changes
    println!("cargo:rerun-if-changed={}", schema_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let output_file = out_dir.join("rrc.rs");

    // Generate Rust code from RRC ASN.1 schema
    // RRC uses Unaligned PER (UPER)
    let mut compiler = Asn1Compiler::new(
        output_file.to_str().expect("Invalid output path"),
        &Visibility::Public,
        vec![Codec::Uper],
        vec![Derive::Debug, Derive::Clone, Derive::PartialEq],
    );

    let schema_str = schema_path.to_str().expect("Invalid schema path");
    if let Err(e) = compiler.compile_files(&[schema_str]) {
        panic!("Failed to compile RRC ASN.1 schema: {e}");
    }
}
