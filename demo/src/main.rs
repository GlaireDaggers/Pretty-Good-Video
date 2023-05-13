extern crate pgv_rs;

use std::io::{Read, Seek};
use std::sync::{Mutex, Arc};
use std::{fs::File, io::BufReader};

use pgv_rs::dec::Decoder;
use sdl2::audio::{AudioCallback, AudioSpecDesired};
use sdl2::{event::Event, render, pixels::PixelFormatEnum, rect::Rect};
use sdl2::pixels::Color;

struct PGVAudio<R: Read + Seek> {
    buffer: Vec<Vec<i16>>,
    decoder: Arc<Mutex<Decoder<R>>>
}

impl<R: Read + Seek + Send> AudioCallback for PGVAudio<R> {
    type Channel = i16;

    fn callback(&mut self, out: &mut [i16]) {
        let mut dec = self.decoder.lock().unwrap();
        dec.decode_audio(&mut self.buffer).unwrap();

        let channels = self.buffer.len();

        for (ch, buf) in self.buffer.iter().enumerate() {
            for i in 0..buf.len() {
                out[i * channels + ch] = buf[i];
            }
        }
    }
}

fn main() {
    let infile = File::open("test.pgv").unwrap();
    let infile = BufReader::new(infile);

    let decoder = Decoder::new(infile).unwrap();
    let decoder = Arc::new(Mutex::new(decoder));

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let audio_subsystem = sdl_context.audio().unwrap();
    let timer_subsystem = sdl_context.timer().unwrap();

    let window = video_subsystem.window("PGV Codec Test", 1280, 720)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().present_vsync().build().unwrap();

    canvas.set_draw_color(Color::RGB(0, 0, 0));
    canvas.clear();
    canvas.present();

    let desired_spec = {
        let dec = decoder.lock().unwrap();

        AudioSpecDesired {
            freq: Some(dec.samplerate as i32),
            channels: Some(dec.channels as u8),
            samples: if dec.audio_sync > 0 { Some(dec.audio_sync as u16) } else { None }
        }
    };

    let device = audio_subsystem.open_playback(None, &desired_spec, |spec| {
        PGVAudio {
            buffer: (0..spec.channels).map(|_| {
                vec![0;spec.samples as usize]
            }).collect(),
            decoder: decoder.clone()
        }
    }).unwrap();

    let tex_creator = canvas.texture_creator();
    let mut tex = {
        let dec = decoder.lock().unwrap();
        tex_creator.create_texture(PixelFormatEnum::IYUV,
            render::TextureAccess::Streaming, dec.width, dec.height).unwrap()
    };

    let mut event_pump = sdl_context.event_pump().unwrap();

    let mut frametimer = timer_subsystem.performance_counter();
    let mut vid_accum = 0.0;
    let vid_delta = {
        let dec = decoder.lock().unwrap();
        1.0 / dec.framerate as f32
    };

    let mut audio_playing = false;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} => {
                    break 'running
                },
                _ => {}
            }
        }

        let new_frametimer = timer_subsystem.performance_counter();
        let delta = (new_frametimer - frametimer) as f32 / timer_subsystem.performance_frequency() as f32;
        frametimer = new_frametimer;

        vid_accum += delta;
        if vid_accum > 0.06 {
            vid_accum = 0.06;
        }

        while vid_accum >= vid_delta {
            vid_accum -= vid_delta;

            let mut dec = decoder.lock().unwrap();

            if !audio_playing {
                device.resume();
                audio_playing = true;
            }

            match dec.decode_frame().unwrap() {
                Some(v) => {
                    tex.update_yuv(Rect::new(0, 0, dec.width, dec.height), &v.plane_y.pixels, dec.width as usize,
                        &v.plane_u.pixels, (dec.width / 2) as usize,
                        &v.plane_v.pixels, (dec.width / 2) as usize).unwrap();
                }
                None => {
                }
            };
        }

        canvas.copy(&tex, None, None).unwrap();
        canvas.present();
    }
}
