use std::fs::File;
use std::io::Write;

/// Get the next byte from the advice tape, if any.
pub type WriteCallback = Box<dyn FnMut(u8) + Send + Sync>;

pub struct WriteCallbackWithDefault(pub WriteCallback);

impl Default for WriteCallbackWithDefault {
    fn default() -> Self {
        Self(Box::new(|_byte| {}))
    }
}

#[cfg(feature = "std")]
pub fn get_file_write_callback(file: File) -> WriteCallback {
    let mut file2 = file;

    Box::new(move |byte: u8| file2.write_all(&[byte]).unwrap())
}
