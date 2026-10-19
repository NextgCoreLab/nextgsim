//! APER Codec Errors
//!
use std::fmt::Display;

#[derive(Debug)]
pub struct Error {
    /// `ErrorCause` can be used to take actions based on the cause.
    pub cause: ErrorCause,
    msg: String,
    context: Vec<String>,
}

#[derive(Debug)]
pub enum ErrorCause {
    /// Remaining Buffer is too short to decode
    BufferTooShort,

    /// Alignment Error during decode
    InvalidAlignment,

    /// Encoding of value is not currently supported
    EncodeNotSupported,

    /// Decoding of value is not currently supported
    DecodeNotSupported,

    /// Generic Error during Encode or Decode
    Generic,

    /// An open type with an unknown key was skipped (#214). Only produced when
    /// the caller opted in with `PerCodecData::set_skip_unknown_open_types`; the
    /// enclosing protocol-IE container consumes it and drops that entry. If it
    /// reaches the caller, skipping was not possible at that position.
    UnknownOpenTypeSkipped,
}

impl Error {
    pub fn new<T: AsRef<str> + Display>(cause: ErrorCause, msg: T) -> Self {
        Error {
            cause,
            msg: msg.to_string(),
            context: Vec::new(),
        }
    }
    pub fn push_context(&mut self, context_elem: &str) {
        self.context.push(context_elem.to_string());
    }

    /// Whether this is a skipped unknown open type (#214).
    pub fn is_unknown_open_type_skipped(&self) -> bool {
        matches!(self.cause, ErrorCause::UnknownOpenTypeSkipped)
    }

    /// Turns a skipped unknown open type into a hard failure (#214).
    ///
    /// Used at every boundary where the skip cannot be resumed from -- a nested
    /// value, a non-final field, a CHOICE -- so that only the immediately
    /// enclosing protocol-IE container ever drops an entry. Any other error is
    /// returned unchanged.
    pub fn into_unskippable(self) -> Self {
        if self.is_unknown_open_type_skipped() {
            Error::new(
                ErrorCause::Generic,
                format!(
                    "{} -- but it is not in a position that can be skipped",
                    self.msg
                ),
            )
        } else {
            self
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.context.is_empty() {
            write!(f, "cause: {}, msg: {}", self.cause, self.msg)
        } else {
            write!(
                f,
                "cause: {}, msg: [{}]:{}",
                self.cause,
                self.context.join("."),
                self.msg
            )
        }
    }
}

impl std::fmt::Display for ErrorCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for Error {}
