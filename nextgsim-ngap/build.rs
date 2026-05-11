//! Build script for NGAP code generation from ASN.1 schema

use std::env;
use std::path::{Path, PathBuf};

use asn1_compiler::generator::{Codec, Derive, Visibility};
use asn1_compiler::Asn1Compiler;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("value expected"));

    // Get the manifest directory (where Cargo.toml is)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("value expected"));
    let schema_path = manifest_dir
        .parent()
        .unwrap_or(Path::new("."))
        .join("tools/ngap-17.9.asn");

    // Rerun if schema changes
    println!("cargo:rerun-if-changed={}", schema_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let output_file = out_dir.join("ngap.rs");

    // Generate Rust code from NGAP ASN.1 schema
    // NGAP uses Aligned PER (APER)
    let mut compiler = Asn1Compiler::new(
        output_file.to_str().expect("value expected"),
        &Visibility::Public,
        vec![Codec::Aper],
        vec![Derive::Debug, Derive::Clone, Derive::PartialEq],
    );

    let schema_str = schema_path.to_str().expect("value expected");
    if let Err(e) = compiler.compile_files(&[schema_str]) {
        panic!("Failed to compile NGAP ASN.1 schema: {e}");
    }
}
