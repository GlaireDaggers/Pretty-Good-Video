use std::io::{Write, Cursor};
use std::iter::zip;

use crate::common::{PGV_MAGIC, PGV_VERSION, PGV_HEADERSIZE};
use crate::qoa::{EncodedAudioFrame, QOA_LMS_LEN, QOA_SLICE_LEN, QOA_FRAME_LEN, LMS, QOA_DEQUANT_TABLE, qoa_lms_predict, qoa_div, QOA_QUANT_TABLE};
use crate::{common::{EncodedFrame, EncodedIFrame, EncodedPFrame, EncodedPPlane}, def::{VideoFrame, ImageSlice}};
use byteorder::{self, WriteBytesExt, LittleEndian};
use crate::huffman::*;

pub struct Encoder {
    pub width: usize,
    pub height: usize,
    pub framerate: u32,
    pub audio_samplerate: u32,
    pub audio_channels: u32,
    prev_frame: VideoFrame,
    frames: Vec<EncodedFrame>,
    audio: Vec<EncodedAudioFrame>,
}

impl Encoder {
    pub fn new(width: usize, height: usize, framerate: u32, audio_samplerate: u32, audio_channels: u32) -> Encoder {
        assert!(width % 2 == 0 && height % 2 == 0);

        Encoder { width: width, height: height, framerate: framerate, audio_samplerate: audio_samplerate, audio_channels: audio_channels,
            frames: Vec::new(), audio: Vec::new(), prev_frame: VideoFrame::new(width, height) }
    }

    pub fn write<TWriter: Write>(self: &Encoder, writer: &mut TWriter) -> Result<(), std::io::Error> {
        // write PGV header
        writer.write_all(PGV_MAGIC)?;
        writer.write_u32::<LittleEndian>(PGV_VERSION)?;
        
        writer.write_u32::<LittleEndian>(self.frames.len() as u32)?;
        writer.write_u32::<LittleEndian>(self.width as u32)?;
        writer.write_u32::<LittleEndian>(self.height as u32)?;
        writer.write_u32::<LittleEndian>(self.framerate as u32)?;

        // if audio sample rate is an exact multiple of framerate, provide a sync hint
        // otherwise, leave at 0 (indicates no hint)
        let audio_sync_hint = if (self.audio_samplerate % self.framerate) == 0 { self.audio_samplerate / self.framerate } else { 0 };

        writer.write_u32::<LittleEndian>(self.audio.len() as u32)?;
        writer.write_u32::<LittleEndian>(self.audio_samplerate as u32)?;
        writer.write_u32::<LittleEndian>(self.audio_channels as u32)?;
        writer.write_u32::<LittleEndian>(audio_sync_hint as u32)?;

        // encode video frames to temporary buffer
        let mut vid_buf = Cursor::new(Vec::<u8>::new());
        self.write_video(&mut vid_buf)?;

        let vid_buf = vid_buf.get_ref();

        // encode audio to temporary buffer
        let mut audio_buf = Cursor::new(Vec::<u8>::new());
        self.write_audio(&mut audio_buf)?;

        let audio_buf = audio_buf.get_ref();

        // write offsets to video/audio buffers
        writer.write_u32::<LittleEndian>(PGV_HEADERSIZE)?;
        writer.write_u32::<LittleEndian>(PGV_HEADERSIZE + vid_buf.len() as u32)?;

        // write video buffer & audio buffer
        writer.write_all(&vid_buf)?;
        writer.write_all(&audio_buf)?;

        // finished writing
        Ok(())
    }

    pub fn get_preview(self: &Encoder) -> &VideoFrame {
        return &self.prev_frame;
    }

    pub fn encode_audio(self: &mut Encoder, audio: Vec<Vec<i16>>) {
        assert!(audio.len() == self.audio_channels as usize);

        let samples = audio[0].len();

        for a in &audio {
            assert!(a.len() == samples);
        }

        // init LMS
        let mut lmses: Vec<LMS> = audio.iter().map(|_| {
            LMS {
                weight: [0, 0, -(1 << 13), 1 << 14],
                history: [0, 0, 0, 0]
            }
        }).collect();

        let mut sample_index = 0;
        while sample_index < samples {
            let frame_len = QOA_FRAME_LEN.clamp(0, samples - sample_index);
            self.audio.push(Encoder::encode_audio_frame(&audio, &mut lmses, sample_index, frame_len));
            sample_index += QOA_FRAME_LEN;
        }
    }

    pub fn encode_iframe(self: &mut Encoder, frame: &VideoFrame) {
        assert!(frame.width == self.width && frame.height == self.height);

        let enc_y = frame.plane_y.encode_plane();
        let dec_y = ImageSlice::decode_plane(&enc_y);

        let enc_u = frame.plane_u.encode_plane();
        let dec_u = ImageSlice::decode_plane(&enc_u);

        let enc_v = frame.plane_v.encode_plane();
        let dec_v = ImageSlice::decode_plane(&enc_v);

        self.frames.push(EncodedFrame::IFrame(EncodedIFrame { y: enc_y, u: enc_u, v: enc_v } ));

        self.prev_frame.plane_y.blit(&dec_y, 0, 0, 0, 0, dec_y.width, dec_y.height);
        self.prev_frame.plane_u.blit(&dec_u, 0, 0, 0, 0, dec_u.width, dec_u.height);
        self.prev_frame.plane_v.blit(&dec_v, 0, 0, 0, 0, dec_v.width, dec_v.height);
    }

    pub fn encode_pframe(self: &mut Encoder, frame: &VideoFrame) {
        assert!(frame.width == self.width && frame.height == self.height);

        let enc_y = ImageSlice::encode_delta_plane(&self.prev_frame.plane_y, &frame.plane_y);
        let dec_y = ImageSlice::decode_delta_plane(&enc_y, &self.prev_frame.plane_y);

        let enc_u = ImageSlice::encode_delta_plane(&self.prev_frame.plane_u, &frame.plane_u);
        let dec_u = ImageSlice::decode_delta_plane(&enc_u, &self.prev_frame.plane_u);

        let enc_v = ImageSlice::encode_delta_plane(&self.prev_frame.plane_v, &frame.plane_v);
        let dec_v = ImageSlice::decode_delta_plane(&enc_v, &self.prev_frame.plane_v);

        self.frames.push(EncodedFrame::PFrame(EncodedPFrame { y: enc_y, u: enc_u, v: enc_v } ));

        self.prev_frame.plane_y.blit(&dec_y, 0, 0, 0, 0, dec_y.width, dec_y.height);
        self.prev_frame.plane_u.blit(&dec_u, 0, 0, 0, 0, dec_u.width, dec_u.height);
        self.prev_frame.plane_v.blit(&dec_v, 0, 0, 0, 0, dec_v.width, dec_v.height);
    }

    fn write_audio<TWriter: Write>(self: &Encoder, writer: &mut TWriter) -> Result<(), std::io::Error> {
        // for each encoded frame:
        //  write number of samples per channel in frame
        //  write total slice count in frame
        //  write LMS history & weights for each channel
        //  write each slice in frame

        for frame in &self.audio {
            writer.write_u16::<LittleEndian>(frame.samples as u16)?;
            writer.write_u16::<LittleEndian>(frame.slices.len() as u16)?;

            for lms in &frame.lmses {
                for history in lms.history {
                    writer.write_i16::<LittleEndian>(history as i16)?;
                }

                for weight in lms.weight {
                    writer.write_i16::<LittleEndian>(weight as i16)?;
                }
            }

            for slice in &frame.slices {
                writer.write_u64::<LittleEndian>(*slice)?;
            }
        }

        Ok(())
    }

    fn write_video<TWriter: Write>(self: &Encoder, writer: &mut TWriter) -> Result<(), std::io::Error> {
        // while there are frames left to write:
        //  get the next group of frames (1 iframe + n pframes)
        //  compute huffman tree from DCT coefficients of each subblock of every frame in group
        //  write huffman tree
        //  write frame group

        let frame_groups = Encoder::gather_frame_groups(&self.frames);

        for group in &frame_groups {
            // gather runlength-encoded DCT coefficients for each frame
            let mut rle_results: Vec<Vec<u8>> = Vec::new();

            for frame in group {
                let mut dct_coeffs = Vec::new();
                
                let blocks = match frame {
                    EncodedFrame::IFrame(i) => {
                        [i.y.blocks.as_slice(), i.u.blocks.as_slice(), i.v.blocks.as_slice()].concat()
                    }
                    EncodedFrame::PFrame(p) => {
                        [p.y.blocks.as_slice(), p.u.blocks.as_slice(), p.v.blocks.as_slice()].concat()
                    }
                };

                for block in blocks {
                    for subblock in block.subblocks {
                        dct_coeffs.extend_from_slice(&subblock.zigzag_scan());
                    }
                }

                let mut tmp_rle = Cursor::new(Vec::new());
                Encoder::runlength_encode(&dct_coeffs, &mut tmp_rle)?;
                let tmp_rle = tmp_rle.into_inner();

                rle_results.push(tmp_rle);
            }

            // compact bytes together
            let mut enc_data: Vec<u8> = Vec::new();

            for frame_enc in &rle_results {
                enc_data.extend_from_slice(&frame_enc);
            }

            // build huffman tree from coefficients
            let tree = HuffmanTree::from_data(&enc_data);
            let table = tree.get_table();

            // write frame group header (size, huffman tree)
            writer.write_u32::<LittleEndian>(group.len() as u32)?;
            writer.write_all(table)?;

            // write each frame's header + data
            for (frame, frame_enc) in zip(group, rle_results) {
                match frame {
                    EncodedFrame::IFrame(_) => {
                        // Encoder::write_iplane_header(writer, &i.y)?;
                        // Encoder::write_iplane_header(writer, &i.u)?;
                        // Encoder::write_iplane_header(writer, &i.v)?;
                    }
                    EncodedFrame::PFrame(p) => {
                        Encoder::write_pplane_header(writer, &p.y)?;
                        Encoder::write_pplane_header(writer, &p.u)?;
                        Encoder::write_pplane_header(writer, &p.v)?;
                    }
                }

                let mut data_cursor = Cursor::new(Vec::new());
                Encoder::write_plane_data(&mut data_cursor, &tree, &frame_enc)?;
                let data = data_cursor.into_inner();

                // write huffman-compressed blob to stream
                writer.write_u32::<LittleEndian>(data.len() as u32)?;
                writer.write_u32::<LittleEndian>(frame_enc.len() as u32)?;
                writer.write_all(&data)?;
            }
        }

        Ok(())
    }

    fn write_plane_data<TWriter: Write>(writer: &mut TWriter, huffman_tree: &HuffmanTree, data: &[u8]) -> Result<(), std::io::Error> {
        match huffman_tree.write(writer, data) {
            Ok(_) => {}
            Err(e) => {
                match e {
                    HuffmanError::EncodeError => { panic!("Data not found in codebook - something went very wrong") }
                    HuffmanError::DecodeError => { unreachable!() }
                    HuffmanError::IOError(e2) => {
                        return Err(e2);
                    }
                }
            }
        };

        Ok(())
    }

    //fn write_iplane_header<TWriter: Write>(_writer: &mut TWriter, _plane: &EncodedIPlane) -> Result<(), std::io::Error> {
    //    Ok(())
    //}

    fn write_pplane_header<TWriter: Write>(writer: &mut TWriter, plane: &EncodedPPlane) -> Result<(), std::io::Error> {
        // write motion vectors for each block
        for vec in &plane.offset {
            writer.write_i8(vec.x)?;
            writer.write_i8(vec.y)?;
        }

        Ok(())
    }

    fn runlength_encode<TWriter: Write>(src: &[u8], dst: &mut TWriter) -> Result<(), std::io::Error> {
        let mut num_zeroes = 0;

        for val in src {
            if *val == 0 && num_zeroes < u8::MAX {
                num_zeroes += 1;
            } else {
                dst.write_u8(num_zeroes as u8)?;
                dst.write_u8(*val)?;
                num_zeroes = 0;
            }
        }

        if num_zeroes > 0 {
            dst.write_u8(num_zeroes as u8)?;
            dst.write_u8(0)?;
        }

        Ok(())
    }

    fn gather_frame_groups(frames: &Vec<EncodedFrame>) -> Vec<Vec<&EncodedFrame>> {
        let mut groups = Vec::new();
        let mut cur_group = Vec::new();

        for frame in frames {
            match frame {
                EncodedFrame::IFrame(_) => {
                    if cur_group.len() != 0 {
                        groups.push(cur_group);
                        cur_group = Vec::new();
                    }
                    
                    cur_group.push(frame);
                }
                EncodedFrame::PFrame(_) => {
                    cur_group.push(frame);
                }
            }
        }

        if cur_group.len() > 0 {
            groups.push(cur_group);
        }

        groups
    }

    // adapted from https://github.com/mattdesl/qoa-format/blob/main/encode.js

    fn encode_audio_frame(audio: &Vec<Vec<i16>>, lmses: &mut Vec<LMS>, sample_offset: usize, frame_len: usize) -> EncodedAudioFrame {
        let mut result = EncodedAudioFrame {
            samples: frame_len as usize,
            slices: Vec::new(),
            lmses: lmses.clone(),
        };

        let mut sample_index = 0;
        while sample_index < frame_len {
            for c in 0..audio.len() {
                let slice_start = sample_index;
                let slice_len = QOA_SLICE_LEN.clamp(0, frame_len - sample_index);

                // brute force search for best scale factor (just loop through all possible scale factors and compare error)

                let mut best_err = i16::MAX as i32;
                let mut best_slice = Vec::new();
                let mut best_slice_scalefactor = 0;
                let mut best_lms = LMS { history: [0;QOA_LMS_LEN], weight: [0;QOA_LMS_LEN] };
                let sampledata = &audio[c];

                for scalefactor in 0..16 {
                    let mut lms = lmses[c];
                    let table = QOA_DEQUANT_TABLE[scalefactor];

                    let mut slice = Vec::new();
                    let mut current_error = 0;
                    let mut idx = slice_start + sample_offset;

                    for _ in 0..slice_len {
                        let sample = sampledata[idx] as i32;
                        idx += 1;

                        let predicted = qoa_lms_predict(lms);
                        let residual = sample - predicted;
                        let scaled = qoa_div(residual, scalefactor);
                        let clamped = scaled.clamp(-8, 8);
                        let quantized = QOA_QUANT_TABLE[(clamped + 8) as usize];
                        let dequantized = table[quantized as usize];
                        let reconstructed = (predicted + dequantized).clamp(i16::MIN as i32, i16::MAX as i32);
                        let error = sample - reconstructed;
                        current_error += error * error;
                        if current_error > best_err {
                            break;
                        }

                        lms.update(reconstructed, dequantized);
                        slice.push(quantized);
                    }

                    if current_error < best_err {
                        best_err = current_error;
                        best_slice = slice;
                        best_slice_scalefactor = scalefactor;
                        best_lms = lms;
                    }
                }

                lmses[c] = best_lms;

                // pack bits into slice - low 4 bits are scale factor, remaining 60 bits are quantized residuals
                let mut slice = (best_slice_scalefactor & 0xF) as u64;

                for i in 0..best_slice.len() {
                    let v = best_slice[i] as u64;
                    slice |= ((v & 0x7) << ((i * 3) + 4)) as u64;
                }

                result.slices.push(slice);
            }

            sample_index += QOA_SLICE_LEN;
        }

        result
    }
}