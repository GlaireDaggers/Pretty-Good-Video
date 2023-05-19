# Pretty-Good-Video
A toy MPEG-like video codec primarily designed for offline video playback in games & other applications

## Encoding videos

Create pgv_rs::enc::Encoder, feed in frames & audio, and then write to file:

```rs
use pgv_rs::enc::Encoder;

let mut enc = Encoder::new(width, height, framerate, 0, samplerate, audio_channels);

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

## Codec Comparisons

While mostly a toy codec, I have still done some benchmarking & comparisons of other codecs - mostly against libtheora.

For a particular 1280x720 30FPS video (which I cannot include due to copyright), I compared visual quality, file size, and speed of decoding the entire
sequence from beginning to end (3774 frames total).

The CPU used to perform these tests was an i5-9300H at 2400 MHz. Both tests were compiled with -O3 for Skylake architecture.

PGVs results at qscale=0 are visually slightly worse than Theora set to 5 mbits/sec, and the file sizes are slightly larger. However, video decoding is a bit faster, and additionally audio decoding is very lightweight as a QOA-based scheme is used (though audio performance was not measured here - you can read the QOA author's [own benchmarks](https://phoboslab.org/log/2023/04/qoa-specification))

| codec | library | file size | time to decode |
| --- | --- | --- | --- |
| Theora | libtheora (via TheoraPlay) | 53.4 MB | 6700 ms |
| PGV | pgv_rs | 60 MB | 5400 ms |