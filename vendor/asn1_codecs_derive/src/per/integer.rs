//! `APER` Code generation for ASN.1 INTEGER Type

use quote::quote;

use crate::{attrs::TyCodecParams, utils};

pub(super) fn generate_per_codec_for_asn_integer(
    ast: &syn::DeriveInput,
    params: &TyCodecParams,
    aligned: bool,
) -> proc_macro::TokenStream {
    let name = &ast.ident;

    let (codec_path, codec_encode_fn, codec_decode_fn, ty_encode_path, ty_decode_path) = if aligned
    {
        (
            quote!(asn1_codecs::aper::AperCodec),
            quote!(aper_encode),
            quote!(aper_decode),
            quote!(asn1_codecs::aper::encode::encode_integer),
            quote!(asn1_codecs::aper::decode::decode_integer),
        )
    } else {
        (
            quote!(asn1_codecs::uper::UperCodec),
            quote!(uper_encode),
            quote!(uper_decode),
            quote!(asn1_codecs::uper::encode::encode_integer),
            quote!(asn1_codecs::uper::decode::decode_integer),
        )
    };
    let ty = if let syn::Data::Struct(ref d) = &ast.data {
        match d.fields {
            syn::Fields::Unnamed(ref f) => {
                if f.unnamed.len() == 1 {
                    let first = f.unnamed.first().unwrap();
                    Some(first.ty.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    } else {
        None
    };

    if ty.is_none() {
        return syn::Error::new_spanned(ast, format!("{} Should be a Unit Struct.", name))
            .to_compile_error()
            .into();
    }

    let (lb, ub, ext) = utils::get_bounds_extensible_from_params(params);

    let tokens = quote! {

        impl #codec_path for #name {
            type Output = Self;

            fn #codec_decode_fn(data: &mut asn1_codecs::PerCodecData) -> Result<Self::Output, asn1_codecs::PerCodecError> {
                log::trace!(concat!("decode: ", stringify!(#name)));

                let decoded = #ty_decode_path(data, #lb, #ub, #ext)?;
                // An extension value (X.691 §12.1) is an unconstrained whole number,
                // so it can exceed the Rust type; refuse it rather than wrap (#231).
                let value = <#ty as ::core::convert::TryFrom<i128>>::try_from(decoded.0).map_err(|_| {
                    asn1_codecs::PerCodecError::new(
                        asn1_codecs::PerCodecErrorCause::Generic,
                        format!(concat!("decode: ", stringify!(#name), " value {} does not fit the type"), decoded.0),
                    )
                })?;
                Ok(Self(value))
            }
            fn #codec_encode_fn(&self, data: &mut asn1_codecs::PerCodecData) -> Result<(), asn1_codecs::PerCodecError> {
                log::trace!(concat!("encode: ", stringify!(#name)));

                // X.691 §12.1: a value of an extensible INTEGER outside the root
                // takes the extension encoding (#231).
                let value = self.0 as i128;
                let lb: Option<i128> = #lb;
                let ub: Option<i128> = #ub;
                let extended = #ext
                    && (lb.is_some_and(|l| value < l) || ub.is_some_and(|u| value > u));
                #ty_encode_path(data, lb, ub, #ext, value, extended)
            }
        }
    };

    tokens.into()
}
