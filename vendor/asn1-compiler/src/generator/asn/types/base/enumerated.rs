//! Mainly 'generator' code for `Asn1ResolvedEnumerated`

use proc_macro2::{Ident, TokenStream};
use quote::quote;

use anyhow::Result;

use crate::generator::Generator;
use crate::resolver::asn::structs::types::base::Asn1ResolvedEnumerated;

impl Asn1ResolvedEnumerated {
    pub(crate) fn generate(&self, name: &str, generator: &mut Generator) -> Result<TokenStream> {
        let struct_name = generator.to_type_ident(name);
        let inner_type = generator.to_inner_type(self.bits, self.signed);

        let named_values = self.generate_named_values(generator)?;

        let mut ty_attributes = quote! { type = "ENUMERATED" };
        if self.extensible {
            ty_attributes.extend(quote! { , extensible = true });
        }
        let ub = format!("{}", self.named_root_values.len() - 1);
        ty_attributes.extend(quote! { , lb = "0" });
        ty_attributes.extend(quote! { , ub =  #ub  });

        let vis = generator.get_visibility_tokens();
        let dir = generator.generate_derive_tokens();

        let struct_tokens = quote! {
            #dir
            #[asn(#ty_attributes)]
            #vis struct #struct_name(#vis #inner_type);

            impl #struct_name {
                #named_values
            }
        };

        Ok(struct_tokens)
    }

    fn generate_named_values(&self, generator: &Generator) -> Result<TokenStream> {
        // Extension additions are numbered on from the root (#213): the first
        // addition is `root_len`, matching how 3GPP specifications and
        // hand-written codecs number them (NGAP CauseRadioNetwork
        // n26-interface-not-available is 45, after 45 root values 0..=44). The
        // derive converts such a value to its X.691 §14.3 within-additions index
        // on encode and back on decode. Before #213 additions got no constant at
        // all, so they could not even be named.
        let root_len = self.named_root_values.len() as i128;
        let ext_values: Vec<(String, i128)> = self
            .named_ext_values
            .iter()
            .map(|(name, idx)| (name.clone(), root_len + idx))
            .collect();

        if let Some((name, value)) = ext_values.last() {
            let max = if self.signed {
                (1_i128 << (self.bits - 1)) - 1
            } else {
                (1_i128 << self.bits) - 1
            };
            if *value > max {
                return Err(anyhow::anyhow!(
                    "ENUMERATED extension value '{}' = {} does not fit the generated {}-bit type",
                    name,
                    value,
                    self.bits
                ));
            }
        }

        let mut tokens = TokenStream::new();
        for (name, value) in self.named_root_values.iter().chain(ext_values.iter()) {
            let const_name = generator.to_const_ident(name);
            let value_literal = generator.to_suffixed_literal(self.bits, self.signed, *value);
            let ty = generator.to_inner_type(self.bits, self.signed);

            let vis = generator.get_visibility_tokens();

            let const_tokens = quote! {
                #vis const #const_name: #ty =  #value_literal ;
            };
            tokens.extend(const_tokens);
        }

        Ok(tokens)
    }

    pub(crate) fn generate_ident_and_aux_type(
        &self,
        generator: &mut Generator,
        input: Option<&String>,
    ) -> Result<Ident> {
        let unique_name = if let Some(unique_name) = input {
            unique_name.to_string()
        } else {
            generator.get_unique_name("ENUMERATED")
        };

        let item = self.generate(&unique_name, generator)?;
        generator.aux_items.push(item);

        Ok(generator.to_type_ident(&unique_name))
    }
}
