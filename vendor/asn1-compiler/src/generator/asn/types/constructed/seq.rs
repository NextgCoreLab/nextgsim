//! Implementation of Code Generation for ASN.1 `SEQUENCE` Type.

use crate::generator::Generator;
use crate::resolver::asn::structs::types::{
    constructed::ResolvedConstructedType, Asn1ResolvedType,
};
use anyhow::Result;
use proc_macro2::TokenStream;
use quote::quote;

impl ResolvedConstructedType {
    pub(crate) fn generate_sequence(
        &self,
        name: &str,
        generator: &mut Generator,
    ) -> Result<TokenStream> {
        if let ResolvedConstructedType::Sequence {
            ref components,
            ref extensible,
            ref additions,
            ..
        } = self
        {
            let type_name = generator.to_type_ident(name);

            let extensible = if *extensible {
                quote! { true }
            } else {
                quote! { false }
            };

            let vis = generator.get_visibility_tokens();

            let mut comp_tokens = TokenStream::new();
            let mut optional_fields = 0;
            for c in components {
                let comp_field_ident = generator.to_value_ident(&c.component.id);
                let comp_ty_suffix = generator.to_type_ident(&c.component.id);
                let input_comp_ty_ident = format!("{}{}", name, comp_ty_suffix);
                let comp_ty_ident = Asn1ResolvedType::generate_name_maybe_aux_type(
                    &c.component.ty,
                    generator,
                    Some(&input_comp_ty_ident),
                )?;
                let mut fld_attrs = vec![];

                let fld_tokens = if c.optional {
                    let idx: proc_macro2::TokenStream =
                        format!("{}", optional_fields).parse().unwrap();
                    fld_attrs.push(quote! { optional_idx = #idx,  });

                    optional_fields += 1;

                    quote! { #vis #comp_field_ident: Option<#comp_ty_ident>, }
                } else {
                    quote! { #vis #comp_field_ident: #comp_ty_ident, }
                };

                if c.key_field {
                    fld_attrs.push(quote! { key_field = true })
                }
                if c.criticality_field {
                    fld_attrs.push(quote! { criticality_field = true })
                }

                let fld_attr_tokens = if !fld_attrs.is_empty() {
                    quote! { #[asn(#(#fld_attrs),*)] }
                } else {
                    quote! {}
                };

                comp_tokens.extend(quote! {
                    #fld_attr_tokens #fld_tokens
                });
            }

            // #253: SEQUENCE extension additions (X.691 §19.7-§19.9) live in one
            // `extensions` field of a generated `<Type>_ExtAdditions` struct.
            // It derives `Default`, so code building a SEQUENCE spells
            // `extensions: Default::default()` once instead of one `None` per
            // addition. Every addition member is `Option<..>`, tagged with its
            // index in the extension-additions presence bitmap; the members of
            // one extension addition group `[[ ... ]]` share the index, and the
            // group's OPTIONAL members are numbered for its own optional bitmap.
            // A group is present on the wire when any member is `Some`; its
            // mandatory members must then all be `Some`.
            let mut ext_struct_tokens = TokenStream::new();
            if let Some(additions) = additions {
                let ext_type_name = generator.to_type_ident(&format!("{}-ExtAdditions", name));
                let mut ext_fields = TokenStream::new();
                for (ext_idx, addition) in additions.iter().enumerate() {
                    let ext_idx: proc_macro2::TokenStream = format!("{}", ext_idx).parse().unwrap();
                    let mut group_optionals = 0;
                    for c in &addition.components {
                        let comp_field_ident = generator.to_value_ident(&c.component.id);
                        let comp_ty_suffix = generator.to_type_ident(&c.component.id);
                        let input_comp_ty_ident = format!("{}{}", name, comp_ty_suffix);
                        let comp_ty_ident = Asn1ResolvedType::generate_name_maybe_aux_type(
                            &c.component.ty,
                            generator,
                            Some(&input_comp_ty_ident),
                        )?;
                        let mut fld_attrs = vec![quote! { extension_idx = #ext_idx }];
                        if addition.is_group {
                            fld_attrs.push(quote! { extension_group = true });
                            if c.optional {
                                let gidx: proc_macro2::TokenStream =
                                    format!("{}", group_optionals).parse().unwrap();
                                fld_attrs.push(quote! { group_optional_idx = #gidx });
                                group_optionals += 1;
                            }
                        }
                        ext_fields.extend(quote! {
                            #[asn(#(#fld_attrs),*)]
                            #vis #comp_field_ident: Option<#comp_ty_ident>,
                        });
                    }
                }
                comp_tokens.extend(quote! {
                    #[asn(extension_additions = true)]
                    #vis extensions: #ext_type_name,
                });
                let dir = generator.generate_derive_tokens();
                ext_struct_tokens = quote! {
                    #dir
                    #[derive(Default)]
                    #[asn(type = "SEQUENCE-EXTENSIONS")]
                    #vis struct #ext_type_name {
                        #ext_fields
                    }
                };
            }

            let mut ty_tokens = quote! { type = "SEQUENCE", extensible = #extensible };

            if optional_fields > 0 {
                let optflds: proc_macro2::TokenStream =
                    format!("{}", optional_fields).parse().unwrap();
                ty_tokens.extend(quote! { , optional_fields = #optflds });
            }

            let dir = generator.generate_derive_tokens();
            Ok(quote! {
                #dir
                #[asn(#ty_tokens)]
                #vis struct #type_name {
                    #comp_tokens
                }

                #ext_struct_tokens
            })
        } else {
            Ok(TokenStream::new())
        }
    }
}
