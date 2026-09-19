//! `APER` Code generation for ASN.1 Choice Type

use proc_macro::TokenStream;
use quote::quote;

use crate::attrs::{parse_fld_meta_as_codec_params, TyCodecParams};

pub(super) fn generate_per_codec_for_asn_choice(
    ast: &syn::DeriveInput,
    params: &TyCodecParams,
    aligned: bool,
) -> proc_macro::TokenStream {
    let name = &ast.ident;

    let (
        codec_path,
        codec_encode_fn,
        codec_decode_fn,
        ty_encode_path,
        ty_decode_path,
        codec_new_perdata_path,
        open_type_length_encode_path,
        open_type_length_decode_path,
    ) = if aligned {
        (
            quote!(asn1_codecs::aper::AperCodec),
            quote!(aper_encode),
            quote!(aper_decode),
            quote!(asn1_codecs::aper::encode::encode_choice_idx),
            quote!(asn1_codecs::aper::decode::decode_choice_idx),
            quote!(asn1_codecs::PerCodecData::new_aper),
            quote!(asn1_codecs::aper::encode::encode_length_determinent),
            quote!(asn1_codecs::aper::decode::decode_length_determinent),
        )
    } else {
        (
            quote!(asn1_codecs::uper::UperCodec),
            quote!(uper_encode),
            quote!(uper_decode),
            quote!(asn1_codecs::uper::encode::encode_choice_idx),
            quote!(asn1_codecs::uper::decode::decode_choice_idx),
            quote!(asn1_codecs::PerCodecData::new_uper),
            quote!(asn1_codecs::uper::encode::encode_length_determinent),
            quote!(asn1_codecs::uper::decode::decode_length_determinent),
        )
    };
    // Aligned PER aligns before the open-type octets; unaligned PER must not (see
    // `append_open_type_unaligned`).
    let open_type_append_fn = if aligned {
        quote!(asn1_codecs::PerCodecData::append_aligned)
    } else {
        quote!(asn1_codecs::PerCodecData::append_open_type_unaligned)
    };
    let open_type_offset_fn = quote!(asn1_codecs::PerCodecData::decode_offset_bits);
    let open_type_resync_fn = quote!(asn1_codecs::PerCodecData::advance_to_open_type_end);
    let lb = params.lb.as_ref().unwrap().value().parse::<i128>().unwrap();
    let ub = params.ub.as_ref().unwrap().value().parse::<i128>().unwrap();
    let ext = params.ext.as_ref();

    let variant_tokens = generate_choice_variant_decode_tokens_using_attrs(
        ast,
        lb,
        ub,
        ext,
        codec_encode_fn.clone(),
        codec_decode_fn.clone(),
        ty_encode_path,
        codec_new_perdata_path,
        open_type_length_encode_path,
        open_type_append_fn,
    );
    if variant_tokens.is_err() {
        return variant_tokens.err().unwrap().to_compile_error().into();
    }
    let (root_decode_tokens, ext_decode_tokens, variant_encode_tokens) = variant_tokens.unwrap();

    let tokens = quote! {

        impl #codec_path for #name {
            type Output = Self;

            fn #codec_decode_fn(data: &mut asn1_codecs::PerCodecData) -> Result<Self::Output, asn1_codecs::PerCodecError> {
                log::trace!(concat!("decode: ", stringify!(#name)));

                let (idx, extended) = #ty_decode_path(data, #lb, #ub, #ext)?;
                // Root and extension indices are SEPARATE number spaces (X.691
                // §23.6): an extension addition is numbered from 0 in declaration
                // order and does NOT continue the root's numbering. So matching on
                // `idx` alone is ambiguous -- in the vendored RRC schema EVERY
                // extensible CHOICE has an extension key that collides with a root
                // key, and a decoder that ignored `extended` would silently return
                // the wrong variant rather than fail. Hence two match blocks.
                if !extended {
                    match idx {
                        #(#root_decode_tokens)*
                        _ => Err(asn1_codecs::PerCodecError::new(
                                asn1_codecs::PerCodecErrorCause::Generic,
                                format!("Index {} is not a valid Choice Index", idx).as_str()))
                    }
                } else {
                    // §23.8: the extension arm's value is wrapped in an open type,
                    // length-prefixed and octet-aligned, so that a decoder which
                    // does not know this extension can skip exactly that many
                    // octets. Read the length, decode the value from the octets it
                    // frames, then resynchronise -- the inner value may not fill
                    // the final octet, and the bits after it are padding that
                    // belongs to the frame rather than to the next field.
                    let length = #open_type_length_decode_path(data, None, None, false)?;
                    let start = #open_type_offset_fn(data);
                    let decoded = match idx {
                        #(#ext_decode_tokens)*
                        _ => Err(asn1_codecs::PerCodecError::new(
                                asn1_codecs::PerCodecErrorCause::Generic,
                                format!("Extension index {} is not a valid Choice addition", idx).as_str()))
                    }?;
                    #open_type_resync_fn(data, start, length)?;
                    Ok(decoded)
                }
            }

            fn #codec_encode_fn(&self, data: &mut asn1_codecs::PerCodecData) -> Result<(), asn1_codecs::PerCodecError> {
                log::trace!(concat!("encode: ", stringify!(#name)));

                match self {
                    #(#variant_encode_tokens)*
                }
            }
        }
    };

    TokenStream::from(tokens)
}

/// Returns `(root_decode_arms, extension_decode_arms, encode_arms)`.
///
/// The decode arms are returned as two lists rather than one because root and
/// extension indices are separate number spaces (X.691 §23.6) and their keys
/// collide in practice -- see the comment at the `match` sites.
#[allow(clippy::too_many_arguments)]
fn generate_choice_variant_decode_tokens_using_attrs(
    ast: &syn::DeriveInput,
    lb: i128,
    ub: i128,
    ext: Option<&syn::LitBool>,
    codec_encode_fn: proc_macro2::TokenStream,
    codec_decode_fn: proc_macro2::TokenStream,
    choice_encode_path: proc_macro2::TokenStream,
    codec_new_perdata_path: proc_macro2::TokenStream,
    open_type_length_encode_path: proc_macro2::TokenStream,
    open_type_append_fn: proc_macro2::TokenStream,
) -> Result<
    (
        Vec<proc_macro2::TokenStream>,
        Vec<proc_macro2::TokenStream>,
        Vec<proc_macro2::TokenStream>,
    ),
    syn::Error,
> {
    let mut root_decode_tokens = vec![];
    let mut ext_decode_tokens = vec![];
    let mut encode_tokens = vec![];

    let mut errors = vec![];
    if let syn::Data::Enum(ref data) = ast.data {
        for variant in &data.variants {
            let codec_params = parse_fld_meta_as_codec_params(&variant.attrs);
            match codec_params {
                Err(e) => errors.push(e),
                Ok(cp) => {
                    let key = cp.key.as_ref();
                    if key.is_none() {
                        errors.push(syn::Error::new_spanned(
                        variant,
                        "Missing Key for the variant. Please provide `#[asn(key = <int>)]` attribute.",
                    ));
                        continue;
                    }
                    let extended = cp.extended.as_ref();
                    let is_extension = extended.map(syn::LitBool::value).unwrap_or(false);
                    let variant_ident = &variant.ident;
                    if let syn::Fields::Unnamed(ref fields) = variant.fields {
                        if fields.unnamed.len() == 1 {
                            let ty = &fields.unnamed.first().as_ref().unwrap().ty;
                            let variant_decode_token = quote! {
                                #key => Ok(Self::#variant_ident(#ty::#codec_decode_fn(data)?)),
                            };
                            let variant_encode_token = if is_extension {
                                // §23.8: an extension arm's value is an open type
                                // -- encoded into its own buffer, then emitted as
                                // an octet count followed by the octet-aligned
                                // bytes. Encoding into `inner` first is what makes
                                // the length knowable before the value is written.
                                // This is the same shape `open.rs` already uses for
                                // an ASN.1 OPEN TYPE, deliberately, so there is one
                                // framing implementation and not two.
                                quote! {
                                    Self::#variant_ident(ref v) => {
                                        #choice_encode_path(data, #lb, #ub, #ext, #key, #extended)?;
                                        let mut inner = #codec_new_perdata_path();
                                        v.#codec_encode_fn(&mut inner)?;
                                        let length = inner.length_in_bytes();
                                        #open_type_length_encode_path(data, None, None, false, length)?;
                                        #open_type_append_fn(data, &mut inner);
                                        Ok(())
                                    }
                                }
                            } else {
                                quote! {
                                    Self::#variant_ident(ref v) => {
                                        #choice_encode_path(data, #lb, #ub, #ext, #key, #extended)?;
                                        v.#codec_encode_fn(data)
                                    }
                                }
                            };
                            if is_extension {
                                ext_decode_tokens.push(variant_decode_token);
                            } else {
                                root_decode_tokens.push(variant_decode_token);
                            }
                            encode_tokens.push(variant_encode_token);
                        } else {
                            errors.push(syn::Error::new_spanned(
                                variant,
                                "Unsupported variant type".to_string(),
                            ));
                        }
                    }
                }
            }
        }
    }

    if let Some((first, others)) = errors.split_first_mut() {
        for e in others {
            first.combine(e.clone())
        }
        Err(first.clone())
    } else {
        Ok((root_decode_tokens, ext_decode_tokens, encode_tokens))
    }
}
