#![allow(dead_code)]

pub const PGV_MAGIC: &[u8] = b"PGVIDEO\0";
pub const PGV_VERSION: u32 = 100;
pub const PGV_HEADERSIZE: u32 = 52;

use crate::{dct::{DctQuantizedMatrix8x8, DctMatrix8x8, Q_TABLE_INTER, Q_TABLE_INTRA}, def::ImageSlice};
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct MotionVector {
    pub x: i8,
    pub y: i8
}

#[derive(Clone, Copy)]
pub struct EncodedMacroBlock {
    pub subblocks: [DctQuantizedMatrix8x8;4]
}

pub enum EncodedFrame {
    IFrame (EncodedIFrame),
    PFrame (EncodedPFrame)
}

pub struct EncodedIFrame {
    pub y: EncodedIPlane,
    pub u: EncodedIPlane,
    pub v: EncodedIPlane,
}

pub struct EncodedPFrame {
    pub y: EncodedPPlane,
    pub u: EncodedPPlane,
    pub v: EncodedPPlane,
}

pub struct EncodedIPlane {
    pub width: usize,
    pub height: usize,
    pub blocks_wide: usize,
    pub blocks_high: usize,
    pub blocks: Vec<EncodedMacroBlock>,
}

pub struct EncodedPPlane {
    pub width: usize,
    pub height: usize,
    pub blocks_wide: usize,
    pub blocks_high: usize,
    pub offset: Vec<MotionVector>,
    pub blocks: Vec<EncodedMacroBlock>,
}

impl<TPixel: Copy + Default> ImageSlice<TPixel> {
    pub fn reduce(self: &ImageSlice<TPixel>) -> ImageSlice<TPixel> {
        let mut new_slice = ImageSlice::new(self.width / 2, self.height / 2);

        for iy in 0..new_slice.height {
            for ix in 0..new_slice.width {
                let sx = ix * 2;
                let sy = iy * 2;

                new_slice.pixels[ix + (iy * new_slice.width)] = self.pixels[sx + (sy * self.width)];
            }
        }

        new_slice
    }

    pub fn double(self: &ImageSlice<TPixel>) -> ImageSlice<TPixel> {
        let mut new_slice = ImageSlice::new(self.width * 2, self.height * 2);

        for iy in 0..self.height {
            for ix in 0..self.width {
                let dx = ix * 2;
                let dy = iy * 2;
                let d_idx = dx + (dy * new_slice.width);
                let px = self.pixels[ix + (iy * self.width)];

                new_slice.pixels[d_idx] = px;
                new_slice.pixels[d_idx + 1] = px;
                new_slice.pixels[d_idx + new_slice.width] = px;
                new_slice.pixels[d_idx + new_slice.width + 1] = px;
            }
        }

        new_slice
    }

    pub fn get_slice(self: &ImageSlice<TPixel>, sx: usize, sy: usize, sw: usize, sh: usize) -> ImageSlice<TPixel> {
        let mut new_slice = ImageSlice::new(sw, sh);
        new_slice.blit(self, 0, 0, sx, sy, sw, sh);

        new_slice
    }

    pub fn blit(self: &mut ImageSlice<TPixel>, src: &ImageSlice<TPixel>, dx: usize, dy: usize, sx: usize, sy: usize, sw: usize, sh: usize) {
        for row in 0..sh {
            let src_row = row + sy;
            let dest_row = row + dy;
            let src_offset = (src_row * src.width) + sx;
            let dst_offset = (dest_row * self.width) + dx;

            self.pixels[dst_offset..(dst_offset + sw)].copy_from_slice(&src.pixels[src_offset..(src_offset + sw)]);
        }
    }
}

impl ImageSlice<u8> {
    pub fn mean_squared_error(from: &ImageSlice<u8>, to: &ImageSlice<u8>) -> f32 {
        assert!(from.width == to.width && from.height == to.height);

        let mut sum = 0.0;

        for (_, (a, b)) in from.pixels.iter().zip(&to.pixels).enumerate() {
            let diff = *a as f32 - *b as f32;
            sum += diff * diff;
        }

        return sum / (from.pixels.len() as f32);
    }

    pub fn calc_residual(from: &ImageSlice<u8>, to: &ImageSlice<u8>) -> ImageSlice<u8> {
        assert!(from.width == to.width && from.height == to.height);

        let mut residual = ImageSlice::new(from.width, from.height);

        for idx in 0..from.pixels.len() {
            let a = from.pixels[idx] as i16;
            let b = to.pixels[idx] as i16;

            residual.pixels[idx] = (((b - a) >> 1) + 128) as u8;
        }

        residual
    }

    pub fn apply_residual(from: &ImageSlice<u8>, residuals: &ImageSlice<u8>, into: &mut ImageSlice<u8>) {
        assert!(from.width == residuals.width && from.height == residuals.height);

        for idx in 0..from.pixels.len() {
            let delta = (residuals.pixels[idx] as i16 - 128) << 1;
            into.pixels[idx] = (from.pixels[idx] as i16 + delta).clamp(0, 255) as u8;
        }
    }

    pub fn decode_plane(src: &EncodedIPlane) -> ImageSlice<u8> {
        let mut plane = ImageSlice::new(src.blocks_wide * 16, src.blocks_high * 16);

        let total_blocks = src.blocks_wide * src.blocks_high;
        let results: Vec<_> = (0..total_blocks).into_par_iter().map(|x| {
            let block_x = x % src.blocks_wide;
            let block_y = x / src.blocks_wide;

            ImageSlice::decode_block(&src.blocks[block_x + (block_y * src.blocks_wide)])
        }).collect();

        for block_y in 0..src.blocks_high {
            for block_x in 0..src.blocks_wide {
                let block = &results[block_x + (block_y * src.blocks_wide)];
                plane.blit(block, block_x * 16, block_y * 16, 0, 0, 16, 16);
            }
        }

        plane
    }

    pub fn decode_plane_2(blocks_wide: usize, blocks_high: usize, blockbuf: &Vec<EncodedMacroBlock>, resultbuf: &mut Vec<ImageSlice<u8>>, cur_plane: &mut ImageSlice<u8>) {
        let total_blocks = blocks_wide * blocks_high;
        (0..total_blocks).into_par_iter().map(|x| {
            let block_x = x % blocks_wide;
            let block_y = x / blocks_wide;

            ImageSlice::decode_block(&blockbuf[block_x + (block_y * blocks_wide)])
        }).collect_into_vec(resultbuf);

        for block_y in 0..blocks_high {
            for block_x in 0..blocks_wide {
                let block = &resultbuf[block_x + (block_y * blocks_wide)];
                cur_plane.blit(block, block_x * 16, block_y * 16, 0, 0, 16, 16);
            }
        }
    }

    pub fn decode_delta_plane(src: &EncodedPPlane, prev: &ImageSlice<u8>) -> ImageSlice<u8> {
        let mut plane = ImageSlice::new(src.blocks_wide * 16, src.blocks_high * 16);

        let total_blocks = src.blocks_wide * src.blocks_high;
        let results: Vec<_> = (0..total_blocks).into_par_iter().map(|x| {
            let block_x = x % src.blocks_wide;
            let block_y = x / src.blocks_wide;

            let motion = src.offset[block_x + (block_y * src.blocks_wide)];
            let px = (((block_x * 16) as i32) + (motion.x as i32)) as usize;
            let py = (((block_y * 16) as i32) + (motion.y as i32)) as usize;
            let prev_block = prev.get_slice(px, py, 16, 16);
            let mut new_block = ImageSlice::new(16, 16);
            ImageSlice::decode_delta_block(&src.blocks[block_x + (block_y * src.blocks_wide)], &prev_block, &mut new_block);

            new_block
        }).collect();

        for block_y in 0..src.blocks_high {
            for block_x in 0..src.blocks_wide {
                let block = &results[block_x + (block_y * src.blocks_wide)];
                plane.blit(block, block_x * 16, block_y * 16, 0, 0, 16, 16);
            }
        }

        plane
    }

    pub fn decode_delta_plane_2(blocks_wide: usize, blocks_high: usize, mvec: &Vec<MotionVector>, blockbuf: &Vec<EncodedMacroBlock>, resultbuf: &mut Vec<ImageSlice<u8>>, cur_plane: &mut ImageSlice<u8>) {
        let total_blocks = blocks_wide * blocks_high;
        (0..total_blocks).into_par_iter().map(|x| {
            let block_x = x % blocks_wide;
            let block_y = x / blocks_wide;

            let motion = mvec[block_x + (block_y * blocks_wide)];
            let px = (((block_x * 16) as i32) + (motion.x as i32)) as usize;
            let py = (((block_y * 16) as i32) + (motion.y as i32)) as usize;
            let prev_block = cur_plane.get_slice(px, py, 16, 16);
            let mut new_block = ImageSlice::new(16, 16);
            ImageSlice::decode_delta_block(&blockbuf[block_x + (block_y * blocks_wide)], &prev_block, &mut new_block);

            new_block
        }).collect_into_vec(resultbuf);

        for block_y in 0..blocks_high {
            for block_x in 0..blocks_wide {
                let block = &resultbuf[block_x + (block_y * blocks_wide)];
                cur_plane.blit(block, block_x * 16, block_y * 16, 0, 0, 16, 16);
            }
        }
    }

    pub fn encode_plane(self: &ImageSlice<u8>) -> EncodedIPlane {
        let pad_width: usize = self.width + (16 - (self.width % 16)) % 16;
        let pad_height = self.height + (16 - (self.height % 16)) % 16;
        let mut img_copy = ImageSlice::new(pad_width, pad_height);
        img_copy.blit(self, 0, 0, 0, 0, self.width, self.height);

        let blocks_wide = pad_width / 16;
        let blocks_high = pad_height / 16;

        let mut blocks: Vec<ImageSlice<u8>> = Vec::with_capacity(blocks_wide * blocks_high);

        // split image plane into 16x16 macroblocks
        for block_y in 0..blocks_high {
            for block_x in 0..blocks_wide {
                let mut block = ImageSlice::new(16, 16);
                block.blit(&img_copy, 0, 0, block_x * 16, block_y * 16, 16, 16);
                blocks.push(block);
            }
        }

        // encode each macroblock in parallel
        let enc_result: Vec<_> = blocks.par_iter().map(|x| {
            ImageSlice::encode_block(x, &Q_TABLE_INTRA)
        }).collect();

        EncodedIPlane { width: pad_width, height: pad_height, blocks_wide: blocks_wide, blocks_high: blocks_high, blocks: enc_result }
    }

    pub fn encode_delta_plane(from: &ImageSlice<u8>, to: &ImageSlice<u8>) -> EncodedPPlane {
        assert!(to.width == from.width && to.height == from.height);

        let pad_width: usize = to.width + (16 - (to.width % 16)) % 16;
        let pad_height = to.height + (16 - (to.height % 16)) % 16;

        let blocks_wide = pad_width / 16;
        let blocks_high = pad_height / 16;

        let mut img_copy = ImageSlice::new(pad_width, pad_height);
        img_copy.blit(to, 0, 0, 0, 0, to.width, to.height);

        let mut blocks: Vec<ImageSlice<u8>> = Vec::with_capacity(blocks_wide * blocks_high);

        // split image plane into 16x16 macroblocks
        for block_y in 0..blocks_high {
            for block_x in 0..blocks_wide {
                let mut block = ImageSlice::new(16, 16);
                block.blit(&img_copy, 0, 0, block_x * 16, block_y * 16, 16, 16);
                blocks.push(block);
            }
        }

        // determine motion vectors for each macroblock which minimize error
        let block_vectors: Vec<_> = blocks.par_iter().enumerate().map(|(idx, block)| {
            let bx = (idx % blocks_wide) * 16;
            let by = (idx / blocks_wide) * 16;
            ImageSlice::match_block(from, block, bx, by)
        }).collect();

        // calculate & encode residuals from previous frame
        let enc_result: Vec<_> = blocks.par_iter().enumerate().map(|(idx, block)| {
            let bx = (idx % blocks_wide) * 16;
            let by = (idx / blocks_wide) * 16;
            let mut prev_block = ImageSlice::new(16, 16);
            prev_block.blit(from, 0, 0,
                (bx as i32 + block_vectors[idx].x as i32) as usize,
                (by as i32 + block_vectors[idx].y as i32) as usize,
                16, 16);
            let residual = ImageSlice::calc_residual(&prev_block, block);
            ImageSlice::encode_block(&residual, &Q_TABLE_INTER)
        }).collect();

        EncodedPPlane { width: from.width, height: from.height, blocks_wide: blocks_wide, blocks_high: blocks_high, blocks: enc_result, offset: block_vectors }
    }

    fn decode_block(src: &EncodedMacroBlock) -> ImageSlice<u8> {
        let subblocks = [
            ImageSlice::decode_subblock(&src.subblocks[0], &Q_TABLE_INTRA),
            ImageSlice::decode_subblock(&src.subblocks[1], &Q_TABLE_INTRA),
            ImageSlice::decode_subblock(&src.subblocks[2], &Q_TABLE_INTRA),
            ImageSlice::decode_subblock(&src.subblocks[3], &Q_TABLE_INTRA)];

        let mut block = ImageSlice::new(16, 16);
        block.blit(&subblocks[0], 0, 0, 0, 0, 8, 8);
        block.blit(&subblocks[1], 8, 0, 0, 0, 8, 8);
        block.blit(&subblocks[2], 0, 8, 0, 0, 8, 8);
        block.blit(&subblocks[3], 8, 8, 0, 0, 8, 8);

        block
    }

    fn decode_delta_block(src: &EncodedMacroBlock, prev: &ImageSlice<u8>, into: &mut ImageSlice<u8>) {
        let subblocks = [
            ImageSlice::decode_subblock(&src.subblocks[0], &Q_TABLE_INTER),
            ImageSlice::decode_subblock(&src.subblocks[1], &Q_TABLE_INTER),
            ImageSlice::decode_subblock(&src.subblocks[2], &Q_TABLE_INTER),
            ImageSlice::decode_subblock(&src.subblocks[3], &Q_TABLE_INTER)];

        let mut block = ImageSlice::new(16, 16);
        block.blit(&subblocks[0], 0, 0, 0, 0, 8, 8);
        block.blit(&subblocks[1], 8, 0, 0, 0, 8, 8);
        block.blit(&subblocks[2], 0, 8, 0, 0, 8, 8);
        block.blit(&subblocks[3], 8, 8, 0, 0, 8, 8);

        ImageSlice::apply_residual(prev, &block, into);
    }

    fn decode_subblock(src: &DctQuantizedMatrix8x8, q_table: &[f32;64]) -> ImageSlice<u8> {
        let mut dct = DctMatrix8x8::decode(src, q_table, 8);
        dct.dct_inverse_transform_columns();
        dct.dct_inverse_transform_rows();

        let mut result = ImageSlice::new(8, 8);
        
        for (idx, px) in dct.m.iter().enumerate() {
            result.pixels[idx] = (*px + 128.0).clamp(0.0, 255.0) as u8;
        }

        result
    }

    fn encode_block(src: &ImageSlice<u8>, q_table: &[f32;64]) -> EncodedMacroBlock {
        assert!(src.width == 16 && src.height == 16);

        // split into 4 subblocks and encode each one
        let subblocks = [
            ImageSlice::encode_subblock(&src.get_slice(0, 0, 8, 8), q_table),
            ImageSlice::encode_subblock(&src.get_slice(8, 0, 8, 8), q_table),
            ImageSlice::encode_subblock(&src.get_slice(0, 8, 8, 8), q_table),
            ImageSlice::encode_subblock(&src.get_slice(8, 8, 8, 8), q_table)];

        EncodedMacroBlock { subblocks: subblocks }
    }

    fn encode_subblock(src: &ImageSlice<u8>, q_table: &[f32;64]) -> DctQuantizedMatrix8x8 {
        assert!(src.width == 8 && src.height == 8);

        let mut dct = DctMatrix8x8::new();
        let cell_px: Vec<f32> = src.pixels.iter().map(|x| (*x as f32) - 128.0).collect();
        dct.m.copy_from_slice(&cell_px);

        dct.dct_transform_rows();
        dct.dct_transform_columns();

        dct.encode(q_table, 8)
    }

    fn match_block(src: &ImageSlice<u8>, block: &ImageSlice<u8>, bx: usize, by: usize) -> MotionVector {
        let mut best_err = f32::INFINITY;
        let mut best_sx = 0;
        let mut best_sy = 0;

        for iy in -16..16 {
            for ix in -16..16 {
                let src_x = bx as i32 + ix;
                let src_y = by as i32 + iy;

                if src_x < 0 || src_x > src.width as i32 - 16 { continue; }
                if src_y < 0 || src_y > src.height as i32 - 16 { continue; }

                let src_block = src.get_slice(src_x as usize, src_y as usize, 16, 16);
                let err = ImageSlice::mean_squared_error(&src_block, block);

                if err < best_err {
                    best_err = err;
                    best_sx = ix;
                    best_sy = iy;
                }
            }
        }
        MotionVector { x: best_sx as i8, y: best_sy as i8 }
    }
}