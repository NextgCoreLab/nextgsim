//! Build script for S1AP code generation from the TS 36.413 ASN.1 schema.
//!
//! Same generator and settings as `nextgsim-ngap`: S1AP, like NGAP, is encoded
//! with the BASIC-PER Aligned variant (TS 36.413 §9.4).

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
        .join("tools/s1ap-19.2.0.asn");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", schema.display());

    let output_file = out_dir.join("s1ap.rs");
    let mut compiler = Asn1Compiler::new(
        output_file.to_str().expect("value expected"),
        &Visibility::Public,
        vec![Codec::Aper],
        vec![Derive::Debug, Derive::Clone, Derive::PartialEq],
    );
    if let Err(e) = compiler.compile_files(&[schema.to_str().expect("value expected")]) {
        panic!("Failed to compile S1AP ASN.1 schema: {e}");
    }
}
