use crate::common::ImageSlice;

pub struct VideoFrame {
    pub width: usize,
    pub height: usize,
    pub plane_y: ImageSlice<u8>,
    pub plane_u: ImageSlice<u8>,
    pub plane_v: ImageSlice<u8>,
}

impl VideoFrame {
    pub fn new(width: usize, height: usize) -> VideoFrame {
        VideoFrame { width: width, height: height,
            plane_y: ImageSlice::new(width, height),
            plane_u: ImageSlice::new(width / 2, height / 2),
            plane_v: ImageSlice::new(width / 2, height / 2) }
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