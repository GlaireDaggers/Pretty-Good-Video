pub mod def;
pub mod enc;
pub mod dec;
mod common;
mod dct;
mod qoa;
mod huffman;

#[cfg(test)]
mod tests {
    use std::{path::Path, fs::File, time::Instant, io::{Cursor, Read}};

    use crate::{common::ImageSlice, dct::{DctMatrix8x8, Q_TABLE_INTRA, DctQuantizedMatrix8x8}, enc::Encoder, def::VideoFrame, dec::Decoder, huffman::HuffmanTree};
    use image::{io::Reader as ImageReader, GrayImage, RgbImage};
    use wav::WAV_FORMAT_PCM;

    #[test]
    fn test_huffman() {
        let data = [0, 1, 1, 5, 5, 5, 10, 10, 10, 10];

        let huffman = HuffmanTree::from_data(&data);
        let table = huffman.get_table();

        println!("Probability table: {:?}", table);

        let mut enc_cursor = Cursor::new(Vec::new());
        huffman.write(&mut enc_cursor, &data).unwrap();

        let enc = enc_cursor.into_inner();

        println!("Encoded data: {:?}", enc);

        let mut dec_cursor = Cursor::new(enc);
        let mut dec = [0;10];

        huffman.read(&mut dec_cursor, &mut dec).unwrap();

        println!("Decoded data: {:?}", dec);

        for idx in 0..10 {
            if dec[idx] != data[idx] {
                panic!("FAILED: data corruption at {} (expected {}, got {})", idx, data[idx], dec[idx]);
            }
        }
    }

    #[test]
    fn test_huffman_2() {
        let mut infile = File::open("compress_dump.bin").unwrap();
        let mut data = Vec::new();
        infile.read_to_end(&mut data).unwrap();

        let huffman = HuffmanTree::from_data(&data);
        let table = huffman.get_table();

        println!("Probability table: {:?}", table);

        let mut enc_cursor = Cursor::new(Vec::new());
        huffman.write(&mut enc_cursor, &data).unwrap();

        let enc = enc_cursor.into_inner();

        println!("Encoded data ({} bytes)", enc.len());

        let mut dec_cursor = Cursor::new(enc);
        let mut dec = vec![0;data.len()];

        huffman.read(&mut dec_cursor, &mut dec).unwrap();

        println!("Decoded data ({} bytes)", dec.len());

        for idx in 0..data.len() {
            if dec[idx] != data[idx] {
                panic!("FAILED: data corruption at {} (expected {}, got {})", idx, data[idx], dec[idx]);
            }
        }
    }

    #[test]
    fn test_zigzag_scan() {
        let mut test_matrix = DctQuantizedMatrix8x8 { m: [0;64] };

        for i in 0..64 {
            test_matrix.m[i] = i as u8;
        }

        println!("SRC MATRIX: {:?}\n", test_matrix.m);

        let test_zigzag = test_matrix.zigzag_scan();

        println!("ZIGZAG MATRIX: {:?}\n", test_zigzag);

        let test_inv_zigzag = DctQuantizedMatrix8x8::inv_zigzag_scan(&test_zigzag);
        
        println!("INV ZIGZAG MATRIX: {:?}", test_inv_zigzag.m);
    }

    #[test]
    fn test_encoder() {
        let mut inp_audio_file = File::open("test_audio.wav").unwrap();
        let (audio_header, audio_data) = wav::read(&mut inp_audio_file).unwrap();

        let mut encoder = Encoder::new(512, 384, 24, audio_header.sampling_rate, audio_header.channel_count as u32);

        for frame_id in 1..163 {
            let frame_path = format!("test_frames/{:0>3}.png", frame_id);
            let frame = load_frame(frame_path);

            if (frame_id - 1) % 16 == 0 {
                encoder.encode_iframe(&frame);
            } else {
                encoder.encode_pframe(&frame);
            }

            println!("Encoded: {} / {}", frame_id, 162);
        }

        let audio_data: Vec<i16> = match audio_data {
            wav::BitDepth::Eight(v) => {
                v.iter().map(|x| {
                    let f = (*x as f32 / 128.0) - 1.0;
                    (f * 32768.0) as i16
                }).collect()
            }
            wav::BitDepth::Sixteen(v) => {
                v
            }
            wav::BitDepth::ThirtyTwoFloat(v) => {
                v.iter().map(|x| {
                    (*x * 32768.0) as i16
                }).collect()
            }
            _ => {
                panic!("Not implemented")
            }
        };

        let mut audio_data_channels = Vec::new();

        for c in 0..audio_header.channel_count as usize {
            let ch: Vec<_> = audio_data.iter().enumerate().filter(|(idx, _)| {
                *idx > c && (*idx - c) % audio_header.channel_count as usize == 0
            }).collect();

            let ch_data: Vec<_> = ch.iter().map(|(_, v)| **v).collect();

            audio_data_channels.push(ch_data);
        }

        encoder.encode_audio(audio_data_channels);
        println!("Encoded audio");

        let mut out_video = File::create("test.pgv").unwrap();
        encoder.write(&mut out_video).unwrap();
    }

    #[test]
    fn test_decoder_speed() {
        for run in 0..10 {
            println!("RUN {}", run);

            let in_video = File::open("test.pgv").unwrap();
            let mut decoder = Decoder::new(in_video).unwrap();

            let start = Instant::now();

            let mut min = u128::MAX;
            let mut max = 0;
            let mut total = 0;

            for _ in 0..decoder.num_frames {
                let fs = Instant::now();
                decoder.decode_frame().unwrap().unwrap();
                let fdur = fs.elapsed().as_millis();
                min = min.min(fdur);
                max = max.max(fdur);
                total += fdur;
            }

            let avg = (total as f32) / (decoder.num_frames as f32);

            let duration = start.elapsed().as_millis();
            let fps = decoder.num_frames as f32 / (duration as f32 / 1000.0);
            println!("Decoded {} frames in {} ms ({} FPS, min: {}, max: {}, avg: {})", decoder.num_frames, duration, fps,
                min, max, avg);
        }
    }

    #[test]
    fn test_decoder() {
        let in_video = File::open("test.pgv").unwrap();
        let mut decoder = Decoder::new(in_video).unwrap();

        assert!(decoder.width == 512 && decoder.height == 384);
        assert!(decoder.framerate == 24);
        assert!(decoder.samplerate == 48000);
        assert!(decoder.channels == 2);

        for idx in 1..decoder.num_frames + 1 {
            let frame = decoder.decode_frame().unwrap().unwrap();

            let frame_out_path = format!("test_frames_out/{:0>3}.png", idx);
            save_frame(frame_out_path, &frame);

            println!("Decoded: {} / {}", idx, decoder.num_frames);
        }

        let mut audio = Vec::new();

        for _ in 0..decoder.channels {
            audio.push(vec![0;(decoder.num_audio_frames * 5120) as usize]);
        }

        decoder.decode_audio(&mut audio).unwrap();

        for (idx, audio_channel_data) in audio.iter().enumerate() {
            let audio_path = format!("audio_out_c{:0>3}.wav", idx);
            let mut out_audio_file = File::create(audio_path).unwrap();

            let header = wav::Header::new(WAV_FORMAT_PCM, 1, decoder.samplerate, 16);
            let data = wav::BitDepth::Sixteen(audio_channel_data.clone());

            wav::write(header, &data, &mut out_audio_file).unwrap();
        }

        println!("Decoded audio");
    }

    #[test]
    fn test_encode_iplane() {
        let frame = load_greyscale_plane("test1.png");
        let encoded_plane = frame.encode_plane();
        let decoded_plane = ImageSlice::decode_plane(&encoded_plane);
        save_greyscale_plane("test1_enc.png", &decoded_plane);
    }

    #[test]
    fn test_dct_image() {
        let frame = load_greyscale_plane("test1.png");
        let mut frame2 = ImageSlice::new(frame.width, frame.height);

        let cells_w = frame.width / 8;
        let cells_h = frame.height / 8;

        for j in 0..cells_h {
            for i in 0..cells_w {
                let mut cell = frame.get_slice(i * 8, j * 8, 8, 8);
                let mut dct = DctMatrix8x8::new();
                let cell_px: Vec<f32> = cell.pixels.iter().map(|x| (*x as f32) - 128.0).collect();
                dct.m.copy_from_slice(&cell_px);

                dct.dct_transform_rows();
                dct.dct_transform_columns();

                let enc = dct.encode(&Q_TABLE_INTRA, 8);
                dct = DctMatrix8x8::decode(&enc, &Q_TABLE_INTRA, 8);

                dct.dct_inverse_transform_columns();
                dct.dct_inverse_transform_rows();

                let inv_cell_px: Vec<u8> = dct.m.iter().map(|x| (*x + 128.0).clamp(0.0, 255.0) as u8).collect();
                cell.pixels.copy_from_slice(&inv_cell_px);

                frame2.blit(&cell, i * 8, j * 8, 0, 0, 8, 8);
            }
        }

        save_greyscale_plane("test1_dct.png", &frame2);
    }

    fn load_frame<Q: AsRef<Path>>(path: Q) -> VideoFrame {
        let src_img = ImageReader::open(path).unwrap().decode().unwrap().into_rgb8();
        
        let yuv_pixels: Vec<[u8;3]> = src_img.pixels().map(|rgb| {
            // https://en.wikipedia.org/wiki/YCbCr - "JPEG Conversion"
            let y = (0.299 * rgb.0[0] as f32) + (0.587 * rgb.0[1] as f32) + (0.114 * rgb.0[2] as f32);
            let u = 128.0 - (0.168736 * rgb.0[0] as f32) - (0.331264 * rgb.0[1] as f32) + (0.5 * rgb.0[2] as f32);
            let v = 128.0 + (0.5 * rgb.0[0] as f32) - (0.418688 * rgb.0[1] as f32) - (0.081312 * rgb.0[2] as f32);
            [y as u8, u as u8, v as u8]
        }).collect();

        // split into three planes
        let y_buffer: Vec<_> = yuv_pixels.iter().map(|x| x[0]).collect();
        let u_buffer: Vec<_> = yuv_pixels.iter().map(|x| x[1]).collect();
        let v_buffer: Vec<_> = yuv_pixels.iter().map(|x| x[2]).collect();

        let y_plane = ImageSlice::from_slice(src_img.width() as usize, src_img.height() as usize, &y_buffer);
        let u_plane = ImageSlice::from_slice(src_img.width() as usize, src_img.height() as usize, &u_buffer);
        let v_plane = ImageSlice::from_slice(src_img.width() as usize, src_img.height() as usize, &v_buffer);

        VideoFrame::from_planes(src_img.width() as usize, src_img.height() as usize, y_plane, u_plane, v_plane)
    }

    fn save_frame<Q: AsRef<Path>>(path: Q, frame: &VideoFrame) {
        let plane_u = frame.plane_u.double();
        let plane_v = frame.plane_v.double();

        let yuv_pixels: Vec<[u8;3]> = frame.plane_y.pixels.iter().enumerate().map(|(idx, y)| {
            let y = *y;
            let u = plane_u.pixels[idx];
            let v = plane_v.pixels[idx];
            
            [y, u, v]
        }).collect();

        let mut rgb_buf: Vec<u8> = Vec::new();

        for yuv in yuv_pixels.iter() {
            let y = yuv[0] as f32;
            let u = yuv[1] as f32 - 128.0;
            let v = yuv[2] as f32 - 128.0;
            
            // https://en.wikipedia.org/wiki/YCbCr - "JPEG Conversion"
            let r = y + (1.402 * v);
            let g = y - (0.344136 * u) - (0.714136 * v);
            let b = y + (1.772 * u);

            rgb_buf.push(r as u8);
            rgb_buf.push(g as u8);
            rgb_buf.push(b as u8);
        }

        let img_buf = RgbImage::from_vec(frame.width as u32, frame.height as u32, rgb_buf).unwrap();
        img_buf.save(path).unwrap();
    }

    fn load_greyscale_plane<Q: AsRef<Path>>(path: Q) -> ImageSlice<u8> {
        let frame_img = ImageReader::open(path).unwrap().decode().unwrap().into_luma8();
        let frame_pixels: Vec<u8> = frame_img.pixels().map(|x| x.0[0]).collect();
        let mut frame = ImageSlice::new(frame_img.width() as usize, frame_img.height() as usize);
        frame.pixels.copy_from_slice(&frame_pixels);

        frame
    }

    fn save_greyscale_plane<Q: AsRef<Path>>(path: Q, plane: &ImageSlice<u8>) {
        let buf = plane.pixels.clone();
        let img_buf = GrayImage::from_vec(plane.width as u32, plane.height as u32, buf).unwrap();
        img_buf.save(path).unwrap();
    }
}
