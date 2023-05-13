pub struct ImageSlice<TPixel> {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<TPixel>,
}

pub struct VideoFrame {
    pub width: usize,
    pub height: usize,
    pub plane_y: ImageSlice<u8>,
    pub plane_u: ImageSlice<u8>,
    pub plane_v: ImageSlice<u8>,
}

impl<TPixel: Copy + Default> ImageSlice<TPixel> {
    pub fn new(width: usize, height: usize) -> ImageSlice<TPixel> {
        ImageSlice { width: width, height: height, pixels: vec![TPixel::default();width * height] }
    }

    pub fn from_slice(width: usize, height: usize, buffer: &[TPixel]) -> ImageSlice<TPixel> {
        assert!(buffer.len() == (width * height));
        let mut slice = ImageSlice::new(width, height);
        slice.pixels.copy_from_slice(buffer);

        slice
    }
}

impl VideoFrame {
    pub fn new(width: usize, height: usize) -> VideoFrame {
        VideoFrame { width: width, height: height,
            plane_y: ImageSlice::new(width, height),
            plane_u: ImageSlice::new(width / 2, height / 2),
            plane_v: ImageSlice::new(width / 2, height / 2) }
    }

    pub fn new_padded(width: usize, height: usize) -> VideoFrame {
        let pad_width: usize = width + (16 - (width % 16)) % 16;
        let pad_height = height + (16 - (height % 16)) % 16;

        let chroma_width = width / 2;
        let chroma_height = height / 2;

        let chroma_pad_width: usize = chroma_width + (16 - (chroma_width % 16)) % 16;
        let chroma_pad_height = chroma_height + (16 - (chroma_height % 16)) % 16;

        VideoFrame { width: width, height: height,
            plane_y: ImageSlice::new(pad_width, pad_height),
            plane_u: ImageSlice::new(chroma_pad_width, chroma_pad_height),
            plane_v: ImageSlice::new(chroma_pad_width, chroma_pad_height) }
    }

    pub fn from_planes(width: usize, height: usize, plane_y: ImageSlice<u8>, plane_u: ImageSlice<u8>, plane_v: ImageSlice<u8>) -> VideoFrame {
        assert!(plane_y.width == width && plane_y.height == height);
        assert!(plane_u.width == width && plane_u.height == height);
        assert!(plane_v.width == width && plane_v.height == height);

        VideoFrame { width: width, height: height,
            plane_y: plane_y,
            plane_u: plane_u.reduce(),
            plane_v: plane_v.reduce() }
    }
}