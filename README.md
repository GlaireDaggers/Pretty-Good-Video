# Pretty-Good-Video
A toy MPEG-like video codec

## Encoding videos

Create pgv_rs::enc::Encoder, feed in frames & audio, and then write to file:

```rs
use pgv_rs::enc::Encoder;

let mut enc = Encoder::new(width, height, framerate, samplerate, audio_channels);

// feed in frames as VideoFrames (1 keyframe every 15 frames)
for (idx, frame) in &my_frames.iter().enumerate() {
  if idx % 15 == 0 {
    enc.encode_iframe(frame);
  } else {
    enc.encode_pframe(frame);
  }
}

// encode audio (one Vec<i16> audio buffer per channel)
enc.encode_audio(my_audio);

// write file to disk
let mut out_video = File::create("my_video.pgv").unwrap();
enc.write(&mut out_video).unwrap();
```

## Decoding videos

Create pgv_rs::dec::Decoder, call decode_frame to get next frame of video, & call decode_audio to get next chunk of audio:

```rs
use pgv_rs::dec::Decoder;

let mut dec = Decoder::new(my_file).unwrap();

for _ in 0..dec.num_frames {
  // returns Option<VideoFrame>. Note that returned U and V planes will be half the size of the Y plane in both dimensions!
  dec.decode_frame().unwrap();
}

// outputs audio into vector of Vec<i16> audio buffers (one buffer per channel). All buffers must have same length.
dec.decode_audio(my_output_buffers).unwrap();
```
