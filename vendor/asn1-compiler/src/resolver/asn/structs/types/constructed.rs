//! Structs for the resolved Base Types

use crate::resolver::asn::structs::types::{constraints::Asn1ConstraintValueSet, Asn1ResolvedType};

#[derive(Debug, Clone)]
pub(crate) enum ResolvedConstructedType {
    Choice {
        name: Option<String>,
        root_components: Vec<ResolvedComponent>,
        additions: Option<Vec<ResolvedComponent>>,
    },
    Sequence {
        name: Option<String>,
        extensible: bool,
        components: Vec<ResolvedSeqComponent>,
        additions: Option<Vec<ResolvedSeqAddition>>,
    },
    SequenceOf {
        name: Option<String>,
        ty: Box<Asn1ResolvedType>,
        size_values: Option<Asn1ConstraintValueSet>,
    },
}

#[derive(Debug, Clone)]
pub(crate) enum ClassFieldComponentType {
    FixedTypeValue,
    Type,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedComponent {
    pub(crate) id: String,
    pub(crate) ty: Asn1ResolvedType,
}

/// One SEQUENCE extension addition (X.691 §19.7): either a lone component
/// after the extension marker, or an extension addition group `[[ ... ]]`.
/// Each is ONE bit of the extension-additions presence bitmap and ONE open
/// type on the wire (§19.8, §19.9); a group's components are encoded inside
/// it as a non-extensible SEQUENCE.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedSeqAddition {
    pub(crate) components: Vec<ResolvedSeqComponent>,
    pub(crate) is_group: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedSeqComponent {
    pub(crate) component: ResolvedComponent,
    pub(crate) optional: bool,
    pub(crate) class_field_type: Option<ClassFieldComponentType>,
    pub(crate) key_field: bool,
    // #214: a non-key fixed-type-value class field named `...criticality`
    // (`&criticality` in every 3GPP *AP IE class). Marked so the codec can report
    // an unknown IE's criticality.
    pub(crate) criticality_field: bool,
    // FIXME : Handle default
    // pub(crate) default: Option<Asn1ResolvedType>
}
