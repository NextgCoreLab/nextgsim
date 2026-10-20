//! `APER` Code generation for ASN.1 `SEQUENCE` type

use quote::quote;

use crate::attrs::{parse_fld_meta_as_codec_params, TyCodecParams};

struct FieldTokens {
    decode_tokens: Vec<proc_macro2::TokenStream>,
    encode_tokens: Vec<proc_macro2::TokenStream>,
    hdr_encode_tokens: Vec<proc_macro2::TokenStream>,
    has_key_field: bool,
    // Every field, in declaration order, for the final `Self { .. }`.
    field_idents: Vec<syn::Ident>,
    // #253: the SEQUENCE's extension additions, in extension-bitmap order.
    additions: Vec<ExtAddition>,
    // #253: the `extension_additions` container field, if the type has one.
    ext_container: Option<(syn::Ident, syn::Ident)>,
}

struct ExtPaths {
    codec_path: proc_macro2::TokenStream,
    codec_encode_fn: proc_macro2::TokenStream,
    codec_decode_fn: proc_macro2::TokenStream,
    ty_encode_path: proc_macro2::TokenStream,
    ty_decode_path: proc_macro2::TokenStream,
    open_encode_path: proc_macro2::TokenStream,
    open_len_decode_path: proc_macro2::TokenStream,
    new_perdata_path: proc_macro2::TokenStream,
    aligned: syn::LitBool,
    flavour: &'static str,
}

impl ExtPaths {
    // Per-codec names: a type derives both APER and UPER, so the inherent
    // methods of a `SEQUENCE-EXTENSIONS` struct must not collide.
    fn ext_fn_names(&self) -> (syn::Ident, syn::Ident, syn::Ident) {
        let span = proc_macro2::Span::call_site();
        (
            syn::Ident::new(&format!("__{}_ext_additions_present", self.flavour), span),
            syn::Ident::new(&format!("__{}_encode_ext_additions", self.flavour), span),
            syn::Ident::new(&format!("__{}_decode_ext_additions", self.flavour), span),
        )
    }
}

/// #253: decode / presence / encode tokens for a SEQUENCE's extension
/// additions (X.691 §19.7-§19.9). The decode tokens bind every member to its
/// `__asn_fld_` local and expect `extensions_present` in scope; the presence
/// tokens define `ext_presence` from `self`; the encode tokens write the
/// presence bitmap and each present addition as an open type.
fn build_addition_tokens(
    additions: &[ExtAddition],
    paths: &ExtPaths,
) -> (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
) {
    let codec_encode_fn = &paths.codec_encode_fn;
    let codec_decode_fn = &paths.codec_decode_fn;
    let ty_encode_path = &paths.ty_encode_path;
    let ty_decode_path = &paths.ty_decode_path;
    let open_encode_path = &paths.open_encode_path;
    let open_len_decode_path = &paths.open_len_decode_path;
    let new_perdata_path = &paths.new_perdata_path;
    let aligned = &paths.aligned;
    let count = additions.len();
    let mut init = vec![];
    let mut arms = vec![];
    let mut presence = vec![];
    let mut encodes = vec![];
    for (k, add) in additions.iter().enumerate() {
        let kk = syn::Index::from(k);
        let ids: Vec<&syn::Ident> = add.members.iter().map(|m| &m.ident).collect();
        for id in &ids {
            let local = local_ident(id);
            init.push(quote! { let mut #local = None; });
        }
        presence.push(quote! {
            if false #(|| self.#ids.is_some())* {
                ext_presence.set(#kk, true);
            }
        });
        if !add.is_group {
            let m = &add.members[0];
            let (id, ty) = (&m.ident, &m.ty);
            let local = local_ident(id);
            arms.push(quote! {
                #kk => { #local = Some(#ty::#codec_decode_fn(data)?); }
            });
            encodes.push(quote! {
                if ext_presence[#kk] {
                    let mut inner = #new_perdata_path();
                    self.#id.as_ref().unwrap().#codec_encode_fn(&mut inner)?;
                    #open_encode_path(data, &mut inner)?;
                }
            });
        } else {
            let gopt = add
                .members
                .iter()
                .filter(|m| m.group_optional_idx.is_some())
                .count();
            let mut dec = vec![];
            let mut gbm_set = vec![];
            let mut enc = vec![];
            for m in &add.members {
                let (id, ty) = (&m.ident, &m.ty);
                let local = local_ident(id);
                match &m.group_optional_idx {
                    Some(j) => {
                        dec.push(quote! {
                            if gbm[#j] { #local = Some(#ty::#codec_decode_fn(data)?); }
                        });
                        gbm_set.push(quote! {
                            if self.#id.is_some() { gbm.set(#j, true); }
                        });
                        enc.push(quote! {
                            if let Some(v) = self.#id.as_ref() { v.#codec_encode_fn(&mut inner)?; }
                        });
                    }
                    None => {
                        dec.push(quote! {
                            #local = Some(#ty::#codec_decode_fn(data)?);
                        });
                        let msg = format!(
                                "mandatory member `{}` of extension addition group {} is absent while the group is present",
                                id, k
                            );
                        enc.push(quote! {
                            match self.#id.as_ref() {
                                Some(v) => v.#codec_encode_fn(&mut inner)?,
                                None => return Err(asn1_codecs::PerCodecError::new(
                                    asn1_codecs::PerCodecErrorCause::Generic, #msg)),
                            }
                        });
                    }
                }
            }
            let gopt_lit = syn::Index::from(gopt);
            arms.push(quote! {
                #kk => {
                    let (gbm, _) = #ty_decode_path(data, false, #gopt_lit)?;
                    let _ = &gbm;
                    #(#dec)*
                }
            });
            encodes.push(quote! {
                if ext_presence[#kk] {
                    let mut inner = #new_perdata_path();
                    let mut gbm = bitvec::bitvec![u8, bitvec::prelude::Msb0; 0; #gopt_lit];
                    #(#gbm_set)*
                    #ty_encode_path(&mut inner, false, &gbm, false)?;
                    #(#enc)*
                    #open_encode_path(data, &mut inner)?;
                }
            });
        }
    }
    let count_lit = syn::Index::from(count);
    (
        quote! {
            #(#init)*
            if extensions_present {
                let ext_bitmap = asn1_codecs::decode_sequence_extension_additions_bitmap(data, #aligned)?;
                for (ext_i, present) in ext_bitmap.iter().by_vals().enumerate() {
                    if !present {
                        continue;
                    }
                    // Each present addition is an open type (§19.9); an
                    // index past the known ones is a newer peer's, skipped.
                    let ext_len = #open_len_decode_path(data)?;
                    let ext_start = data.decode_offset_bits();
                    match ext_i {
                        #(#arms)*
                        _ => {}
                    }
                    data.advance_to_open_type_end(ext_start, ext_len)?;
                }
            }
        },
        quote! {
            let mut ext_presence = bitvec::bitvec![u8, bitvec::prelude::Msb0; 0; #count_lit];
            #(#presence)*
        },
        quote! {
            if ext_presence.any() {
                asn1_codecs::encode_sequence_extension_additions_bitmap(data, &ext_presence, #aligned)?;
                #(#encodes)*
            }
        },
    )
}

/// #253: the impl for a `SEQUENCE-EXTENSIONS` struct -- the container a
/// generated SEQUENCE keeps its extension additions in. Not a codec type of
/// its own: it gives the enclosing SEQUENCE's codec three inherent methods.
pub(super) fn generate_per_codec_for_asn_sequence_extensions(
    ast: &syn::DeriveInput,
    _params: &TyCodecParams,
    aligned: bool,
) -> proc_macro::TokenStream {
    let name = &ast.ident;
    let paths = ext_paths(aligned);
    let field_tokens = generate_seq_field_codec_tokens_using_attrs(
        ast,
        paths.codec_encode_fn.clone(),
        paths.codec_decode_fn.clone(),
        true,
    );
    let field_tokens = match field_tokens {
        Ok(f) => f,
        Err(e) => return e.to_compile_error().into(),
    };
    if !field_tokens.decode_tokens.is_empty() || field_tokens.ext_container.is_some() {
        return syn::Error::new_spanned(
            ast,
            "A SEQUENCE-EXTENSIONS struct holds only extension addition fields.",
        )
        .to_compile_error()
        .into();
    }
    let additions = field_tokens.additions;
    let field_idents = field_tokens.field_idents;
    let field_locals: Vec<syn::Ident> = field_idents.iter().map(local_ident).collect();
    let (present_fn, encode_fn, decode_fn) = paths.ext_fn_names();
    let ids: Vec<&syn::Ident> = field_idents.iter().collect();
    let (dec, pres, enc) = build_addition_tokens(&additions, &paths);
    let codec_path = &paths.codec_path;

    let tokens = quote! {
        impl #name {
            #[doc(hidden)]
            pub fn #present_fn(&self) -> bool {
                false #(|| self.#ids.is_some())*
            }

            #[doc(hidden)]
            pub fn #encode_fn(&self, data: &mut asn1_codecs::PerCodecData) -> Result<(), asn1_codecs::PerCodecError> {
                // An inherent impl: bring the codec trait's methods into scope.
                #[allow(unused_imports)]
                use #codec_path as _;
                #pres
                #enc
                Ok(())
            }

            #[doc(hidden)]
            pub fn #decode_fn(data: &mut asn1_codecs::PerCodecData) -> Result<Self, asn1_codecs::PerCodecError> {
                #[allow(unused_imports)]
                use #codec_path as _;
                let extensions_present = true;
                #dec
                Ok(Self { #(#field_idents: #field_locals),* })
            }
        }
    };
    tokens.into()
}

fn ext_paths(aligned: bool) -> ExtPaths {
    if aligned {
        ExtPaths {
            codec_path: quote!(asn1_codecs::aper::AperCodec),
            codec_encode_fn: quote!(aper_encode),
            codec_decode_fn: quote!(aper_decode),
            ty_encode_path: quote!(asn1_codecs::aper::encode::encode_sequence_header),
            ty_decode_path: quote!(asn1_codecs::aper::decode::decode_sequence_header),
            open_encode_path: quote!(asn1_codecs::aper::encode::encode_open_type),
            open_len_decode_path: quote!(asn1_codecs::aper::decode::decode_open_type_length),
            new_perdata_path: quote!(asn1_codecs::PerCodecData::new_aper),
            aligned: syn::LitBool::new(true, proc_macro2::Span::call_site()),
            flavour: "aper",
        }
    } else {
        ExtPaths {
            codec_path: quote!(asn1_codecs::uper::UperCodec),
            codec_encode_fn: quote!(uper_encode),
            codec_decode_fn: quote!(uper_decode),
            ty_encode_path: quote!(asn1_codecs::uper::encode::encode_sequence_header),
            ty_decode_path: quote!(asn1_codecs::uper::decode::decode_sequence_header),
            open_encode_path: quote!(asn1_codecs::uper::encode::encode_open_type),
            open_len_decode_path: quote!(asn1_codecs::uper::decode::decode_open_type_length),
            new_perdata_path: quote!(asn1_codecs::PerCodecData::new_uper),
            aligned: syn::LitBool::new(false, proc_macro2::Span::call_site()),
            flavour: "uper",
        }
    }
}

fn local_ident(id: &syn::Ident) -> syn::Ident {
    syn::Ident::new(&format!("__asn_fld_{}", id), id.span())
}

struct ExtMember {
    ident: syn::Ident,
    ty: syn::Ident,
    group_optional_idx: Option<syn::LitInt>,
}

struct ExtAddition {
    is_group: bool,
    members: Vec<ExtMember>,
}

pub(super) fn generate_per_codec_for_asn_sequence(
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
            quote!(asn1_codecs::aper::encode::encode_sequence_header),
            quote!(asn1_codecs::aper::decode::decode_sequence_header),
        )
    } else {
        (
            quote!(asn1_codecs::uper::UperCodec),
            quote!(uper_encode),
            quote!(uper_decode),
            quote!(asn1_codecs::uper::encode::encode_sequence_header),
            quote!(asn1_codecs::uper::decode::decode_sequence_header),
        )
    };
    let (open_encode_path, open_len_decode_path, new_perdata_path) = if aligned {
        (
            quote!(asn1_codecs::aper::encode::encode_open_type),
            quote!(asn1_codecs::aper::decode::decode_open_type_length),
            quote!(asn1_codecs::PerCodecData::new_aper),
        )
    } else {
        (
            quote!(asn1_codecs::uper::encode::encode_open_type),
            quote!(asn1_codecs::uper::decode::decode_open_type_length),
            quote!(asn1_codecs::PerCodecData::new_uper),
        )
    };

    let ext = params.ext.as_ref();
    let opt_count = if params.optional_fields.is_some() {
        params.optional_fields.as_ref().unwrap().clone()
    } else {
        syn::LitInt::new("0", proc_macro2::Span::call_site())
    };

    let aligned = syn::LitBool::new(aligned, proc_macro2::Span::call_site());

    let extensible = ext.is_some_and(|e| e.value());
    let field_tokens = generate_seq_field_codec_tokens_using_attrs(
        ast,
        codec_encode_fn.clone(),
        codec_decode_fn.clone(),
        extensible,
    );
    if field_tokens.is_err() {
        return field_tokens.err().unwrap().to_compile_error().into();
    }
    let field_tokens = field_tokens.unwrap();
    let fld_decode_tokens = field_tokens.decode_tokens;
    let hdr_encode_tokens = field_tokens.hdr_encode_tokens;
    let fld_encode_tokens = field_tokens.encode_tokens;
    let has_key_field = field_tokens.has_key_field;
    let field_idents = field_tokens.field_idents;
    let field_locals: Vec<syn::Ident> = field_idents.iter().map(local_ident).collect();
    let additions = field_tokens.additions;

    // #253: SEQUENCE extension additions (X.691 §19.7-§19.9). A generated
    // SEQUENCE carries them in one `extension_additions` container field (a
    // `SEQUENCE-EXTENSIONS` struct deriving `Default`); hand-written types may
    // also flatten them. With none known the old behaviour stands: a peer's
    // additions are skipped by length.
    let paths = ExtPaths {
        codec_path: codec_path.clone(),
        codec_encode_fn: codec_encode_fn.clone(),
        codec_decode_fn: codec_decode_fn.clone(),
        ty_encode_path: ty_encode_path.clone(),
        ty_decode_path: ty_decode_path.clone(),
        open_encode_path,
        open_len_decode_path,
        new_perdata_path,
        aligned: aligned.clone(),
        flavour: if aligned.value() { "aper" } else { "uper" },
    };
    let (ext_decode_tokens, ext_presence_tokens, ext_encode_tokens, ext_present_expr) =
        if let Some((id, ty)) = field_tokens.ext_container.as_ref() {
            let local = local_ident(id);
            let (present_fn, encode_fn, decode_fn) = paths.ext_fn_names();
            (
                quote! {
                    let #local = if extensions_present {
                        #ty::#decode_fn(data)?
                    } else {
                        <#ty as Default>::default()
                    };
                },
                quote! { let ext_present = self.#id.#present_fn(); },
                quote! {
                    if ext_present {
                        self.#id.#encode_fn(data)?;
                    }
                },
                quote! { ext_present },
            )
        } else if additions.is_empty() {
            (
                quote! {
                    if extensions_present {
                        asn1_codecs::decode_sequence_extensions_skip_bits(data, #aligned)?;
                    }
                },
                quote! {},
                quote! {},
                quote! { false },
            )
        } else {
            let (d, p, e) = build_addition_tokens(&additions, &paths);
            (d, p, e, quote! { ext_presence.any() })
        };

    let inner_body = quote! {
        let (bitmap, extensions_present) = #ty_decode_path(data, #ext, #opt_count)?;
        let _ = &bitmap;
        #(#fld_decode_tokens)*
        #ext_decode_tokens
        Ok(Self { #(#field_idents: #field_locals),* })
    };
    // #214: only a protocol-IE SEQUENCE (with a key field) may pass a skipped
    // unknown open type up to its SEQUENCE OF; any other SEQUENCE is a nested
    // position the skip cannot resume from.
    let decode_body = if has_key_field {
        inner_body
    } else {
        quote! {
            let result: Result<Self::Output, asn1_codecs::PerCodecError> = (|| { #inner_body })();
            result.map_err(asn1_codecs::PerCodecError::into_unskippable)
        }
    };

    let tokens = quote! {
        impl #codec_path for #name {
            type Output = Self;

            fn #codec_decode_fn(data: &mut asn1_codecs::PerCodecData) -> Result<Self::Output, asn1_codecs::PerCodecError> {
                log::trace!(concat!("decode: ", stringify!(#name)));

                #decode_body
            }

            fn #codec_encode_fn(&self, data: &mut asn1_codecs::PerCodecData) -> Result<(), asn1_codecs::PerCodecError> {
                log::trace!(concat!("encode: ", stringify!(#name)));

                let mut bitmap = bitvec::bitvec![u8, bitvec::prelude::Msb0; 0; #opt_count];

                #(#hdr_encode_tokens)*

                #ext_presence_tokens

                #ty_encode_path(data, #ext, &bitmap, #ext_present_expr)?;

                #(#fld_encode_tokens)*

                #ext_encode_tokens

                Ok(())
            }
        }
    };

    tokens.into()
}

fn generate_seq_field_codec_tokens_using_attrs(
    ast: &syn::DeriveInput,
    codec_encode_fn: proc_macro2::TokenStream,
    codec_decode_fn: proc_macro2::TokenStream,
    extensible: bool,
) -> Result<FieldTokens, syn::Error> {
    let mut decode_fields: Vec<(syn::Ident, proc_macro2::TokenStream)> = vec![];
    let mut field_idents: Vec<syn::Ident> = vec![];
    let mut additions: Vec<ExtAddition> = vec![];
    let mut ext_container: Option<(syn::Ident, syn::Ident)> = None;
    let mut has_key_field = false;
    let mut encode_tokens = vec![];
    let mut hdr_encode_tokens = vec![];

    let mut errors: Vec<syn::Error> = vec![];
    if let syn::Data::Struct(ref data) = ast.data {
        if let syn::Fields::Named(ref fields) = data.fields {
            for field in &fields.named {
                let codec_params = parse_fld_meta_as_codec_params(&field.attrs);
                match codec_params {
                    Err(e) => errors.push(e),
                    Ok(cp) => {
                        let field_type = get_field_type(field);
                        if field_type.ty.is_none() {
                            errors.push(syn::Error::new_spanned(
                                field,
                                "Field Type is not in supported Format!",
                            ));
                            continue;
                        } else {
                            let ty_ident = field_type.ty.unwrap();
                            let optional = field_type.is_optional;
                            let id = field.ident.as_ref().unwrap();
                            field_idents.push(id.clone());
                            // #253: the container of a generated SEQUENCE's
                            // extension additions.
                            if cp.extension_additions.as_ref().is_some_and(|e| e.value()) {
                                if optional || ext_container.is_some() {
                                    errors.push(syn::Error::new_spanned(
                                        field,
                                        "One non-Option `extension_additions` field is allowed.",
                                    ));
                                    continue;
                                }
                                ext_container = Some((id.clone(), ty_ident));
                                continue;
                            }
                            // #253: an extension addition field is decoded and
                            // encoded after the root, by the extension machinery.
                            if let Some(ext_idx) = cp.extension_idx.as_ref() {
                                if !optional {
                                    errors.push(syn::Error::new_spanned(
                                        field,
                                        "An extension addition field must be an `Option<..>`.",
                                    ));
                                    continue;
                                }
                                let k: usize = match ext_idx.base10_parse() {
                                    Ok(k) => k,
                                    Err(e) => {
                                        errors.push(e);
                                        continue;
                                    }
                                };
                                let is_group =
                                    cp.extension_group.as_ref().is_some_and(|g| g.value());
                                if k > additions.len() {
                                    errors.push(syn::Error::new_spanned(
                                        field,
                                        "Extension addition indices must be consecutive from 0.",
                                    ));
                                    continue;
                                }
                                if k == additions.len() {
                                    additions.push(ExtAddition {
                                        is_group,
                                        members: vec![],
                                    });
                                } else if !is_group || !additions[k].is_group {
                                    errors.push(syn::Error::new_spanned(
                                        field,
                                        "Only members of one extension addition group may share an `extension_idx`.",
                                    ));
                                    continue;
                                }
                                additions[k].members.push(ExtMember {
                                    ident: id.clone(),
                                    ty: ty_ident,
                                    group_optional_idx: cp.group_optional_idx.clone(),
                                });
                                continue;
                            }
                            let fld_decode_tokens = if optional {
                                let optional_idx = cp.optional_idx.as_ref();

                                match optional_idx {
                                    None => {
                                        errors.push(syn::Error::new_spanned(
                                            field,
                                            "Optional Field without Optional Index.",
                                        ));
                                        // Just pass empty quote
                                        quote! {}
                                    }
                                    Some(optidx) => {
                                        quote! {
                                            {
                                            let present = bitmap[#optidx];
                                            if present {
                                                Some(#ty_ident::#codec_decode_fn(data)?)
                                            } else {
                                                None
                                            }
                                            }
                                        }
                                    }
                                }
                            } else {
                                let key_field = cp.key_field.as_ref();
                                let is_key_field = if let Some(kf) = key_field {
                                    kf.value()
                                } else {
                                    false
                                };

                                let is_criticality_field =
                                    cp.criticality_field.as_ref().is_some_and(|c| c.value());

                                if is_criticality_field {
                                    // #214: remembered so a skipped unknown open
                                    // type in this SEQUENCE can report it.
                                    quote! {
                                        {
                                        let value = #ty_ident::#codec_decode_fn(data)?;
                                        data.set_criticality(value.0 as i128);
                                        value
                                        }
                                    }
                                } else if !is_key_field {
                                    quote! {
                                        {
                                        #ty_ident::#codec_decode_fn(data)?
                                        }
                                    }
                                } else {
                                    quote! {
                                        {
                                        let value = #ty_ident::#codec_decode_fn(data)?;
                                        let _ = data.set_key(value.0 as i128);
                                        value
                                        }
                                    }
                                }
                            };

                            let field_encode_token = if optional {
                                quote! {
                                    if self.#id.is_some() {
                                        let #id = self.#id.as_ref().unwrap();
                                        #id.#codec_encode_fn(data)?;
                                    } else {
                                    }
                                }
                            } else {
                                quote! {
                                    self.#id.#codec_encode_fn(data)?;
                                }
                            };
                            let header_encode_token = if optional {
                                let optional_idx = cp.optional_idx.as_ref();
                                quote! {
                                    if self.#id.is_some() {
                                        bitmap.set(#optional_idx, true);
                                    } else {
                                    }
                                }
                            } else {
                                quote! {}
                            };
                            if cp.key_field.as_ref().is_some_and(|k| k.value()) {
                                has_key_field = true;
                            }
                            decode_fields.push((id.clone(), fld_decode_tokens));
                            encode_tokens.push(field_encode_token);
                            hdr_encode_tokens.push(header_encode_token);
                        }
                    }
                }
            }
        }
    }

    // #214: a skipped unknown open type may only escape a protocol-IE SEQUENCE
    // (one with a `key_field`) from its FINAL field, and only when the SEQUENCE is
    // not extensible: then the skip has consumed the whole entry and the
    // enclosing SEQUENCE OF can drop it and carry on. From any earlier field the
    // entry is only partly read, so the skip becomes a hard failure here. (A
    // SEQUENCE without a key field converts it wholesale, in the caller.)
    let n = decode_fields.len();
    let decode_tokens = decode_fields
        .into_iter()
        .enumerate()
        .map(|(i, (id, expr))| {
            // Bound to a prefixed local so a field named like a codec local
            // (`data`, `bitmap`, ...) cannot shadow it.
            let local = local_ident(&id);
            if has_key_field && (i + 1 < n || extensible) {
                quote! {
                    let #local = (|| -> Result<_, asn1_codecs::PerCodecError> { Ok(#expr) })()
                        .map_err(asn1_codecs::PerCodecError::into_unskippable)?;
                }
            } else {
                quote! { let #local = #expr; }
            }
        })
        .collect();

    if let Some((first, others)) = errors.split_first_mut() {
        for e in others {
            first.combine(e.clone())
        }
        Err(first.clone())
    } else {
        Ok(FieldTokens {
            decode_tokens,
            hdr_encode_tokens,
            encode_tokens,
            has_key_field,
            field_idents,
            additions,
            ext_container,
        })
    }
}

struct StructFieldType {
    ty: Option<syn::Ident>,
    is_optional: bool,
}

fn get_field_type(field: &syn::Field) -> StructFieldType {
    fn field_is_optional(field: &syn::Field) -> bool {
        if let syn::Type::Path(ref typepath) = field.ty {
            typepath.path.leading_colon.is_none()
                && typepath.path.segments.len() == 1
                && typepath.path.segments.iter().next().unwrap().ident == "Option"
        } else {
            false
        }
    }

    let is_optional = field_is_optional(field);

    let ty = if is_optional {
        if let syn::Type::Path(ref tp) = field.ty {
            let type_params = &tp.path.segments.iter().next().unwrap().arguments;
            match type_params {
                syn::PathArguments::AngleBracketed(params) => {
                    let generic_args = params.args.iter().next().unwrap();
                    if let syn::GenericArgument::Type(syn::Type::Path(tpinner)) = generic_args {
                        Some(tpinner.path.segments.iter().next().unwrap().ident.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        }
    } else if let syn::Type::Path(ref tp) = field.ty {
        Some(tp.path.segments.iter().next().unwrap().ident.clone())
    } else {
        None
    };

    StructFieldType { ty, is_optional }
}
