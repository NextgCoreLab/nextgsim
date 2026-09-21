//! Build script for NGAP and NRPPa code generation from ASN.1 schemas

use std::env;
use std::path::{Path, PathBuf};

use asn1_compiler::generator::{Codec, Derive, Visibility};
use asn1_compiler::Asn1Compiler;

/// Compile one ASN.1 schema into `OUT_DIR/<output>` with the APER codec.
///
/// Both schemas want identical generator settings, so the only per-schema
/// inputs are the source file and the generated module name.
fn generate(schema_path: &Path, out_dir: &Path, output: &str, label: &str) {
    println!("cargo:rerun-if-changed={}", schema_path.display());

    let output_file = out_dir.join(output);
    let mut compiler = Asn1Compiler::new(
        output_file.to_str().expect("value expected"),
        &Visibility::Public,
        vec![Codec::Aper],
        vec![Derive::Debug, Derive::Clone, Derive::PartialEq],
    );

    let schema_str = schema_path.to_str().expect("value expected");
    if let Err(e) = compiler.compile_files(&[schema_str]) {
        panic!("Failed to compile {label} ASN.1 schema: {e}");
    }
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("value expected"));

    // Get the manifest directory (where Cargo.toml is)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("value expected"));
    let tools = manifest_dir
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    println!("cargo:rerun-if-changed=build.rs");

    // NGAP and NRPPa both use Aligned PER (APER). They are generated into two
    // separate modules rather than compiled together because they are distinct
    // protocols that merely share a transport: NGAP carries NRPPa as an opaque
    // `NRPPa-PDU ::= OCTET STRING` (TS 38.413 §9.3.3.17), so nothing in the NGAP
    // schema references an NRPPa type. Compiling them as one unit would also
    // collide on the names both define (`Criticality`, `ProcedureCode`,
    // `ProtocolIE-ID`, `CriticalityDiagnostics`, ...) with different
    // constraints.
    generate(
        &tools.join("tools/ngap-17.9.asn"),
        &out_dir,
        "ngap.rs",
        "NGAP",
    );
    generate(
        &tools.join("tools/nrppa-19.2.0.asn1"),
        &out_dir,
        "nrppa.rs",
        "NRPPa",
    );
}
