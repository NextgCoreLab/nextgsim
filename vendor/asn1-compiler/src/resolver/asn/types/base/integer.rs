//! Functionality for handling Resolved ASN.1 INTEGER Types

use anyhow::Result;

use crate::parser::asn::structs::types::{base::Asn1TypeInteger, Asn1Type};
use crate::resolver::asn::structs::types::base::Asn1ResolvedInteger;
use crate::resolver::Resolver;

impl Asn1ResolvedInteger {
    // Resolves the parsed integer to it's Resolved variant with right bit width and signedness.
    pub(super) fn resolve_integer(
        ty: &Asn1Type,
        _i: &Asn1TypeInteger,
        resolver: &mut Resolver,
    ) -> Result<Asn1ResolvedInteger> {
        let mut base = Asn1ResolvedInteger::default();

        if ty.constraints.is_none() {
            return Ok(base);
        } else {
            let constraints = ty.constraints.as_ref().unwrap();
            if constraints.is_empty() {
                return Ok(base);
            }
        }

        // Get the Values that are expected
        let value_set = ty.get_integer_valueset_from_constraint(resolver)?;

        // The Rust type must hold every value the type can carry, which for an
        // extensible INTEGER includes its extension additions
        // (`INTEGER (0..4095, ..., 4096..2000000)`): X.691 §12.1 encodes those as
        // unconstrained whole numbers, so a root-sized type could neither build
        // nor decode them (#231). The PER-visible bounds stay the ROOT's -- the
        // generator reads `root_values` for lb/ub. A bare `...` has no visible
        // additions, so such a type keeps its root width; a received value that
        // does not fit is refused on decode rather than wrapped.
        let (min, max) = {
            let root = &value_set.root_values;
            let added = value_set.additional_values.as_ref();
            let min = [root.min(), added.and_then(|a| a.min())]
                .iter()
                .flatten()
                .copied()
                .min();
            let max = [root.max(), added.and_then(|a| a.max())]
                .iter()
                .flatten()
                .copied()
                .max();
            (min, max)
        };
        if let Some(x) = min {
            base.signed = x < 0
        }

        let bit_width = if base.signed {
            let min = min.unwrap();
            let max = max.unwrap();
            let bits_needed_max = 128 - max.abs().leading_zeros();
            let bits_needed_min = 128 - min.abs().leading_zeros();
            std::cmp::max(bits_needed_min, bits_needed_max)
        } else if min.is_none() {
            8_u32
        } else {
            let max = max.unwrap();
            128 - max.leading_zeros()
        };

        base.bits = if bit_width <= 8 {
            8
        } else if bit_width <= 16 {
            16
        } else if bit_width <= 32 {
            32
        } else if bit_width <= 64 {
            64
        } else {
            128
        };

        // TODO: If we have named values, They should be added to Global list of resolved definitions.
        // if i.named_values.is_some() {}

        let _ = base.resolved_constraints.replace(value_set);
        Ok(base)
    }
}
