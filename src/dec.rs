use std::io::{Read, Seek, Cursor};

use byteorder::{ReadBytesExt, LittleEndian};
use rayon::prelude::{ParallelIterator, IntoParallelRefMutIterator};

use crate::{common::{PGV_MAGIC, PGV_VERSION, EncodedMacroBlock, MotionVector, MacroBlock}, dct::{DctQuantizedMatrix8x8, DctMatrix8x8, Q_TABLE_INTER, Q_TABLE_INTRA}, def::{VideoFrame, ImageSlice}, qoa::{QOA_LMS_LEN, LMS, QOA_SLICE_LEN, qoa_lms_predict, QOA_DEQUANT_TABLE}};
use crate::huffman::*;

pub struct Decoder<TReader: Read + Seek> {
    pub version: u32,
    pub num_frames: u32,
    pub width: u32,
    pub height: u32,
    pub framerate: u32,
    pub num_audio_frames: u32,
    pub samplerate: u32,
    pub channels: u32,
    pub audio_sync: u32,
    _video_offset: u32,
    _audio_offset: u32,
    reader: TReader,
    video_pos: usize,
    audio_pos: usize,
    pic_in_gop: usize,
    huffman_tree: HuffmanTree,
    cur_frame: VideoFrame,
    read_frames: u32,
    audio_dec_buffer: Vec<Vec<i16>>,
    audio_dec_pos: usize,
    audio_dec_len: usize,
    read_audio_frames: u32,
    rle_buffer: Vec<u8>,
    block_buf_y: Vec<EncodedMacroBlock>,
    block_buf_u: Vec<EncodedMacroBlock>,
    block_buf_v: Vec<EncodedMacroBlock>,
    mvec_buf_y: Vec<MotionVector>,
    mvec_buf_u: Vec<MotionVector>,
    mvec_buf_v: Vec<MotionVector>,
    dec_buf_y: Vec<MacroBlock>,
    dec_buf_u: Vec<MacroBlock>,
    dec_buf_v: Vec<MacroBlock>,
    enc_buf: Vec<u8>,
    dec_buf: Vec<u8>,
    qtable_inter: [f32;64],
    qtable_intra: [f32;64],
}

#[derive(Debug)]
pub enum DecodeError {
    FormatError,
    VersionError,
    IOError(std::io::Error)
}

impl<TReader: Read + Seek> Decoder<TReader> {
    pub fn new(mut reader: TReader) -> Result<Decoder<TReader>, DecodeError> {
        // read header
        let mut magic = [0;8];
        match reader.read_exact(&mut magic) {
            Ok(_) => {}
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let magic_match = magic.iter().zip(PGV_MAGIC.iter()).all(|(a, b)| *a == *b);

        if !magic_match {
            return Err(DecodeError::FormatError);
        }

        // read version
        let version = match reader.read_u32::<LittleEndian>() {
            Ok(ver) => {
                if ver != PGV_VERSION {
                    return Err(DecodeError::VersionError);
                }

                ver
            }
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let num_frames = match reader.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let width = match reader.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let height = match reader.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let framerate = match reader.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let audio_frames = match reader.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let audio_samplerate = match reader.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let audio_channels = match reader.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let audio_sync_hint = match reader.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let video_offset = match reader.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let audio_offset = match reader.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let pad_width = width + (16 - (width % 16)) % 16;
        let pad_height = height + (16 - (height % 16)) % 16;

        let chroma_width = width / 2;
        let chroma_height = height / 2;

        let chroma_pad_width = chroma_width + (16 - (chroma_width % 16)) % 16;
        let chroma_pad_height = chroma_height + (16 - (chroma_height % 16)) % 16;

        let blocks_wide = pad_width / 16;
        let blocks_high = pad_height / 16;

        let chroma_blocks_wide = chroma_pad_width / 16;
        let chroma_blocks_high = chroma_pad_height / 16;

        let blocks_luma = blocks_wide * blocks_high;
        let blocks_chroma = chroma_blocks_wide * chroma_blocks_high;

        let total_blocks = blocks_luma + (blocks_chroma * 2);
        let rle_buffer_len = total_blocks * 256;

        Ok(Decoder {
            version: version,
            num_frames: num_frames,
            width: width,
            height: height,
            framerate: framerate,
            num_audio_frames: audio_frames,
            samplerate: audio_samplerate,
            channels: audio_channels,
            audio_sync: audio_sync_hint,
            _video_offset: video_offset,
            _audio_offset: audio_offset,
            reader: reader,
            audio_pos: audio_offset as usize,
            video_pos: video_offset as usize,
            pic_in_gop: 0,
            huffman_tree: HuffmanTree::empty(),
            cur_frame: VideoFrame::new_padded(width as usize, height as usize),
            read_frames: 0,
            read_audio_frames: 0,
            audio_dec_buffer: Vec::new(),
            audio_dec_pos: 0,
            audio_dec_len: 0,
            rle_buffer: vec![0;rle_buffer_len as usize],
            block_buf_y: Vec::with_capacity((blocks_wide * blocks_high) as usize),
            block_buf_u: Vec::with_capacity((chroma_blocks_wide * chroma_blocks_high) as usize),
            block_buf_v: Vec::with_capacity((chroma_blocks_wide * chroma_blocks_high) as usize),
            mvec_buf_y: Vec::with_capacity((blocks_wide * blocks_high) as usize),
            mvec_buf_u: Vec::with_capacity((chroma_blocks_wide * chroma_blocks_high) as usize),
            mvec_buf_v: Vec::with_capacity((chroma_blocks_wide * chroma_blocks_high) as usize),
            dec_buf_y: Vec::with_capacity((blocks_wide * blocks_high) as usize),
            dec_buf_u: Vec::with_capacity((chroma_blocks_wide * chroma_blocks_high) as usize),
            dec_buf_v: Vec::with_capacity((chroma_blocks_wide * chroma_blocks_high) as usize),
            enc_buf: Vec::new(),
            dec_buf: Vec::new(),
            qtable_inter: [0.0;64],
            qtable_intra: [0.0;64],
        })
    }

    /// Decode the next frame from the stream, returning either a new frame for display, or None if there are no more frames to decode
    pub fn decode_frame(self: &mut Decoder<TReader>) -> Result<Option<VideoFrame>, std::io::Error> {
        if self.read_frames >= self.num_frames {
            return Ok(None);
        }

        let mut frame = VideoFrame::new(self.width as usize, self.height as usize);

        self.reader.seek(std::io::SeekFrom::Start(self.video_pos as u64))?;

        if self.pic_in_gop == 0 {
            // read in new frame group header
            let qscale = self.reader.read_i32::<LittleEndian>()?;
            self.pic_in_gop = self.reader.read_u32::<LittleEndian>()? as usize;

            assert!(qscale >= 0 && qscale <= 8);

            self.qtable_inter = DctMatrix8x8::transform_qtable(&Q_TABLE_INTER, 8, qscale);
            self.qtable_intra = DctMatrix8x8::transform_qtable(&Q_TABLE_INTRA, 8, qscale);

            let mut huffman_table = [0;256];
            self.reader.read_exact(&mut huffman_table)?;

            self.huffman_tree = HuffmanTree::from_table(&huffman_table);

            // decode i-frame

            // read in huffman-compressed blob
            let enc_blob_len = self.reader.read_u32::<LittleEndian>()? as usize;
            let dec_blob_len = self.reader.read_u32::<LittleEndian>()? as usize;
            Decoder::read_plane_data(&mut self.reader, enc_blob_len, dec_blob_len, &mut self.huffman_tree, &mut self.enc_buf, &mut self.dec_buf)?;
            
            // run-length decode
            Decoder::<TReader>::runlength_decode(&self.dec_buf, &mut self.rle_buffer);
            let mut enc_blob_reader = Cursor::new(&self.rle_buffer);

            // decode planes
            Decoder::<TReader>::read_iplane_data(&mut self.block_buf_y, self.width as usize, self.height as usize, &mut enc_blob_reader)?;
            Decoder::<TReader>::read_iplane_data(&mut self.block_buf_u, self.width as usize / 2, self.height as usize / 2, &mut enc_blob_reader)?;
            Decoder::<TReader>::read_iplane_data(&mut self.block_buf_v, self.width as usize / 2, self.height as usize / 2, &mut enc_blob_reader)?;

            let qtable = self.qtable_intra;

            [(&self.block_buf_y, &mut self.cur_frame.plane_y, &mut self.dec_buf_y),
                (&self.block_buf_u, &mut self.cur_frame.plane_u, &mut self.dec_buf_u),
                (&self.block_buf_v, &mut self.cur_frame.plane_v, &mut self.dec_buf_v)]
            .par_iter_mut().for_each(|x| {
                let blocks_wide = x.1.width / 16;
                let blocks_high = x.1.height / 16;

                ImageSlice::decode_plane_2(blocks_wide, blocks_high, &x.0, x.2, x.1, &qtable);
            });
        } else {
            // decode p-frame
            // decode headers for each plane, then decode data for each plane

            Decoder::read_pplane_headers(&mut self.mvec_buf_y, self.width as usize, self.height as usize, &mut self.reader)?;
            Decoder::read_pplane_headers(&mut self.mvec_buf_u, self.width as usize / 2, self.height as usize / 2, &mut self.reader)?;
            Decoder::read_pplane_headers(&mut self.mvec_buf_v, self.width as usize / 2, self.height as usize / 2, &mut self.reader)?;

            // read in huffman-compressed blob
            let enc_blob_len = self.reader.read_u32::<LittleEndian>()? as usize;
            let dec_blob_len = self.reader.read_u32::<LittleEndian>()? as usize;
            Decoder::read_plane_data(&mut self.reader, enc_blob_len, dec_blob_len, &mut self.huffman_tree, &mut self.enc_buf, &mut self.dec_buf)?;

            // run-length decode
            Decoder::<TReader>::runlength_decode(&self.dec_buf, &mut self.rle_buffer);

            let mut enc_blob_reader = Cursor::new(&self.rle_buffer);

            // decode planes
            Decoder::<TReader>::read_pplane_data(&mut self.block_buf_y, self.width as usize, self.height as usize, &self.mvec_buf_y, &mut enc_blob_reader)?;
            Decoder::<TReader>::read_pplane_data(&mut self.block_buf_u, self.width as usize / 2, self.height as usize / 2, &self.mvec_buf_u, &mut enc_blob_reader)?;
            Decoder::<TReader>::read_pplane_data(&mut self.block_buf_v, self.width as usize / 2, self.height as usize / 2, &self.mvec_buf_v, &mut enc_blob_reader)?;
            
            let qtable = self.qtable_inter;

            [(&self.mvec_buf_y, &self.block_buf_y, &mut self.cur_frame.plane_y, &mut self.dec_buf_y),
                (&self.mvec_buf_u, &self.block_buf_u, &mut self.cur_frame.plane_u, &mut self.dec_buf_u),
                (&self.mvec_buf_v, &self.block_buf_v, &mut self.cur_frame.plane_v, &mut self.dec_buf_v)]
            .par_iter_mut().for_each(|x| {
                let blocks_wide = x.2.width / 16;
                let blocks_high = x.2.height / 16;

                ImageSlice::decode_delta_plane_2(blocks_wide, blocks_high, &x.0, x.1, x.3, x.2, &qtable);
            });
        }

        self.video_pos = self.reader.stream_position()? as usize;
        self.read_frames += 1;
        self.pic_in_gop -= 1;
        
        frame.plane_y.blit(&self.cur_frame.plane_y, 0, 0, 0, 0, frame.width, frame.height);
        frame.plane_u.blit(&self.cur_frame.plane_u, 0, 0, 0, 0, frame.width / 2, frame.height / 2);
        frame.plane_v.blit(&self.cur_frame.plane_v, 0, 0, 0, 0, frame.width / 2, frame.height / 2);

        Ok(Some(frame))
    }

    // decode the next chunk of audio from the stream into the output
    pub fn decode_audio(self: &mut Decoder<TReader>, output: &mut Vec<Vec<i16>>) -> Result<(), std::io::Error> {
        self.reader.seek(std::io::SeekFrom::Start(self.audio_pos as u64))?;

        assert!(output.len() == self.channels as usize);
        let output_len = output[0].len();

        for o in output.iter() {
            assert!(o.len() == output_len);
        }

        if self.read_audio_frames >= self.num_audio_frames {
            return Ok(());
        }

        let mut samples_written = 0;
        while samples_written < output_len {
            let samples_in_buffer = self.audio_dec_len - self.audio_dec_pos;
            let samples_to_write = samples_in_buffer.clamp(0, output_len - samples_written);

            if samples_to_write > 0 {
                let slice_start = self.audio_dec_pos;
                let slice_end = slice_start + samples_to_write;

                let dest_slice_start = samples_written;
                let dest_slice_end = dest_slice_start + samples_to_write;

                for (idx, o) in output.iter_mut().enumerate() {                
                    let dest_slice = &mut o[dest_slice_start..dest_slice_end];
                    dest_slice.copy_from_slice(&self.audio_dec_buffer[idx][slice_start..slice_end]);
                }

                samples_written += samples_to_write;
                self.audio_dec_pos += samples_to_write;
            }

            if self.audio_dec_pos >= self.audio_dec_len {
                if self.read_audio_frames >= self.num_audio_frames {
                    break;
                } else {
                    // read in next buffer
                    self.audio_dec_buffer = self.read_audio_frame()?;
                    self.audio_dec_len = self.audio_dec_buffer[0].len();
                    self.audio_dec_pos = 0;
                    self.read_audio_frames += 1;
                }
            }
        }

        self.audio_pos = self.reader.stream_position()? as usize;
        Ok(())
    }

    fn read_audio_frame(self: &mut Decoder<TReader>) -> Result<Vec<Vec<i16>>, std::io::Error> {
        // read number of samples per channel in this frame & number of slices
        let samples = self.reader.read_u16::<LittleEndian>()? as i32;
        let slice_count = self.reader.read_u16::<LittleEndian>()? as i32;

        let mut audio: Vec<Vec<i16>> = Vec::with_capacity(self.channels as usize);

        // read LMS state
        let mut lmses = Vec::with_capacity(self.channels as usize);

        for _ in 0..self.channels {
            let mut history = [0;QOA_LMS_LEN];
            let mut weight = [0;QOA_LMS_LEN];

            for i in 0..QOA_LMS_LEN {
                history[i] = self.reader.read_i16::<LittleEndian>()? as i32;
            }

            for i in 0..QOA_LMS_LEN {
                weight[i] = self.reader.read_i16::<LittleEndian>()? as i32;
            }

            lmses.push(LMS { history: history, weight: weight });
            audio.push(vec![0;samples as usize]);
        }

        for slice_idx in 0..slice_count {
            let slice_ch = slice_idx % self.channels as i32;
            let mut slice = self.reader.read_u64::<LittleEndian>()?;

            // decode slice
            let scalefactor = slice & 0xF;
            slice = slice >> 4;

            let slice_start = (slice_idx / self.channels as i32) as usize * QOA_SLICE_LEN;
            let slice_end = (slice_start + QOA_SLICE_LEN).clamp(0, samples as usize);

            for i in slice_start..slice_end {
                let quantized = slice & 0x7;
                slice >>= 3;
                let predicted = qoa_lms_predict(lmses[slice_ch as usize]);
                let dequantized = QOA_DEQUANT_TABLE[scalefactor as usize][quantized as usize];
                let reconstructed = (predicted + dequantized).clamp(i16::MIN as i32, i16::MAX as i32);

                audio[slice_ch as usize][i] = reconstructed as i16;
                lmses[slice_ch as usize].update(reconstructed, dequantized);
            }
        }

        Ok(audio)
    }

    fn read_plane_data(reader: &mut TReader, enc_len: usize, dec_len: usize, tree: &HuffmanTree, read_buf: &mut Vec<u8>, enc_data_buf: &mut Vec<u8>) -> Result<(), std::io::Error> {
        read_buf.resize(enc_len, 0);
        enc_data_buf.resize(dec_len, 0);

        reader.read_exact(read_buf)?;
        let mut compressed_enc_blob_cursor = Cursor::new(read_buf);

        match tree.read(&mut compressed_enc_blob_cursor, enc_data_buf) {
            Ok(_) => {}
            Err(e) => match e {
                HuffmanError::DecodeError => { return Err(std::io::Error::new(std::io::ErrorKind::Other, "Failed decoding compressed data")) }
                HuffmanError::EncodeError => { unreachable!() }
                HuffmanError::IOError(e2) => { return Err(e2) }
            }
        };

        Ok(())
    }

    fn read_pplane_headers(mblock: &mut Vec<MotionVector>, width: usize, height: usize, reader: &mut TReader) -> Result<(), std::io::Error> {
        let pad_width: usize = width + (16 - (width % 16)) % 16;
        let pad_height = height + (16 - (height % 16)) % 16;

        let blocks_wide = pad_width / 16;
        let blocks_high = pad_height / 16;

        let total_blocks = blocks_wide * blocks_high;

        unsafe {
            mblock.set_len(total_blocks);
        }

        for idx in 0..total_blocks {
            let packed = reader.read_u8()?;
            
            let mx = (packed << 4) as i8 >> 4;
            let my = (packed & 0xF0) as i8 >> 4;

            mblock[idx] = MotionVector{ x: mx, y: my };
        }

        Ok(())
    }

    fn read_pplane_data<R: Read + Seek>(mblock: &mut Vec<EncodedMacroBlock>, width: usize, height: usize, _header: &[MotionVector], reader: &mut R) -> Result<(), std::io::Error> {
        let pad_width: usize = width + (16 - (width % 16)) % 16;
        let pad_height = height + (16 - (height % 16)) % 16;

        let blocks_wide = pad_width / 16;
        let blocks_high = pad_height / 16;

        let total_blocks = blocks_wide * blocks_high;

        unsafe {
            mblock.set_len(total_blocks);
        }

        for idx in 0..total_blocks {
            // decode each subblock
            let subblock_0 = Decoder::<TReader>::decode_subblock(reader)?;
            let subblock_1 = Decoder::<TReader>::decode_subblock(reader)?;
            let subblock_2 = Decoder::<TReader>::decode_subblock(reader)?;
            let subblock_3 = Decoder::<TReader>::decode_subblock(reader)?;

            mblock[idx] = EncodedMacroBlock { subblocks: [subblock_0, subblock_1, subblock_2, subblock_3] };
        }

        Ok(())
    }

    fn read_iplane_data<R: Read + Seek>(mblock: &mut Vec<EncodedMacroBlock>, width: usize, height: usize, reader: &mut R) -> Result<(), std::io::Error> {
        let pad_width: usize = width + (16 - (width % 16)) % 16;
        let pad_height = height + (16 - (height % 16)) % 16;

        let blocks_wide = pad_width / 16;
        let blocks_high = pad_height / 16;

        let total_blocks = blocks_wide * blocks_high;

        unsafe {
            mblock.set_len(total_blocks);
        }

        for idx in 0..total_blocks {
            // decode each subblock
            let subblock_0 = Decoder::<TReader>::decode_subblock(reader)?;
            let subblock_1 = Decoder::<TReader>::decode_subblock(reader)?;
            let subblock_2 = Decoder::<TReader>::decode_subblock(reader)?;
            let subblock_3 = Decoder::<TReader>::decode_subblock(reader)?;

            mblock[idx] = EncodedMacroBlock { subblocks: [subblock_0, subblock_1, subblock_2, subblock_3] };
        }

        Ok(())
    }

    fn runlength_decode(encoded: &[u8], into: &mut [u8]) {
        let mut out_idx = 0;
        
        let mut idx = 0;
        while idx < encoded.len() {
            let run = encoded[idx] as usize;
            
            into[out_idx..(out_idx + run)].fill(0);
            out_idx += run;

            // RLE can sometimes be padded with a last dummy 0 value, exit loop in this case
            if out_idx >= into.len() {
                break;
            }

            into[out_idx] = encoded[idx + 1];
            out_idx += 1;

            idx += 2;
        }
    }

    fn decode_subblock<R: Read + Seek>(reader: &mut R) -> Result<DctQuantizedMatrix8x8, std::io::Error> {
        let mut subblock_data = [0;64];
        reader.read_exact(&mut subblock_data)?;

        // inv zigzag scan to get final subblock data
        Ok(DctQuantizedMatrix8x8 { m: subblock_data })
    }
}