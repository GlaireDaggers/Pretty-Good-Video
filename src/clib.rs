use std::{ffi::c_void, io::{Read, Seek}, ptr::null_mut};

use crate::dec::Decoder;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PGV_Stream {
    context: *const c_void,
    read_fn: fn(context: *const c_void, buffer: *mut u8, len: usize) -> usize,
    seek_fn: fn(context: *const c_void, offset: i64, whence: i32) -> u64,
}

impl Read for PGV_Stream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        Ok((self.read_fn)(self.context, buf.as_mut_ptr(), buf.len()))
    }
}

impl Seek for PGV_Stream {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        match pos {
            std::io::SeekFrom::Start(v) => {
                Ok((self.seek_fn)(self.context, v as i64, 0))
            }
            std::io::SeekFrom::Current(v) => {
                Ok((self.seek_fn)(self.context, v, 1))
            }
            std::io::SeekFrom::End(v) => {
                Ok((self.seek_fn)(self.context, v, 2))
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_new(stream: *mut PGV_Stream) -> *mut c_void {
    unsafe {
        match Decoder::new(*stream) {
            Ok(decoder) => {
                let box_dec = Box::new(decoder);
                return Box::into_raw(box_dec) as *mut c_void;
            }
            Err(_) => {
                return null_mut();
            }
        };
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_destroy(decoder: *mut c_void) {
    unsafe {
        drop(Box::from_raw(decoder as *mut Decoder<PGV_Stream>));
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_width(decoder: *mut c_void) -> u32 {
    unsafe {
        let dec = Box::from_raw(decoder as *mut Decoder<PGV_Stream>);
        let width = dec.width;
        Box::into_raw(dec);
        return width;
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_height(decoder: *mut c_void) -> u32 {
    unsafe {
        let dec = Box::from_raw(decoder as *mut Decoder<PGV_Stream>);
        let height = dec.height;
        Box::into_raw(dec);
        return height;
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_num_frames(decoder: *mut c_void) -> u32 {
    unsafe {
        let dec = Box::from_raw(decoder as *mut Decoder<PGV_Stream>);
        let num_frames = dec.num_frames;
        Box::into_raw(dec);
        return num_frames;
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_framerate(decoder: *mut c_void) -> u32 {
    unsafe {
        let dec = Box::from_raw(decoder as *mut Decoder<PGV_Stream>);
        let framerate = dec.framerate;
        Box::into_raw(dec);
        return framerate;
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_num_audio_frames(decoder: *mut c_void) -> u32 {
    unsafe {
        let dec = Box::from_raw(decoder as *mut Decoder<PGV_Stream>);
        let num_audio_frames = dec.num_audio_frames;
        Box::into_raw(dec);
        return num_audio_frames;
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_audio_samplerate(decoder: *mut c_void) -> u32 {
    unsafe {
        let dec = Box::from_raw(decoder as *mut Decoder<PGV_Stream>);
        let samplerate = dec.samplerate;
        Box::into_raw(dec);
        return samplerate;
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_audio_channels(decoder: *mut c_void) -> u32 {
    unsafe {
        let dec = Box::from_raw(decoder as *mut Decoder<PGV_Stream>);
        let channels = dec.channels;
        Box::into_raw(dec);
        return channels;
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_audio_sync_hint(decoder: *mut c_void) -> u32 {
    unsafe {
        let dec = Box::from_raw(decoder as *mut Decoder<PGV_Stream>);
        let audio_sync = dec.audio_sync;
        Box::into_raw(dec);
        return audio_sync;
    }
}

#[no_mangle]
pub extern "C" fn pgv_decoder_decode_frame(decoder: *mut c_void, buf_y: *mut u8, buf_u: *mut u8, buf_v: *mut u8) -> i32 {
    unsafe {
        let mut dec = Box::from_raw(decoder as *mut Decoder<PGV_Stream>);
        let result = match dec.decode_frame() {
            Ok(v) => match v {
                Some(f) => {
                    buf_y.copy_from(f.plane_y.pixels.as_ptr(), f.plane_y.pixels.len());
                    buf_u.copy_from(f.plane_u.pixels.as_ptr(), f.plane_u.pixels.len());
                    buf_v.copy_from(f.plane_v.pixels.as_ptr(), f.plane_v.pixels.len());

                    1
                }
                None => {
                    0
                }
            }
            Err(_) => {
                -1
            }
        };
        Box::into_raw(dec);

        return result;
    }
}

pub extern "C" fn pgv_decoder_decode_audio(decoder: *mut c_void, buf: *mut *mut i16, buf_len: usize) -> i32 {
    unsafe {
        let mut dec = Box::from_raw(decoder as *mut Decoder<PGV_Stream>);

         let mut tmp: Vec<Vec<i16>> = (0..dec.channels).map(|_| {
            vec![0;buf_len]
        }).collect();

        let result = match dec.decode_audio(&mut tmp) {
            Ok(_) => {
                1
            }
            Err(_) => {
                0
            }
        };

        for ch in 0..dec.channels {
            let dst = *buf.offset(ch as isize);
            dst.copy_from(tmp[ch as usize].as_ptr(), buf_len);
        }

        Box::into_raw(dec);
        return result;
    }
}