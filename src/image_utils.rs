use image::{GenericImageView, imageops::FilterType};
use ndarray::{Array4, s};
use std::fs;

pub fn load_image_as_tensor(path: &str, resize: Option<(u32, u32)>) -> Array4<f64> {
    let img = image::open(path).expect("Failed to open image");
    let img = if let Some((width, height)) = resize {
        img.resize_exact(width, height, FilterType::Nearest)
    } else {
        img
    };
    let (width, height) = img.dimensions();

    // 将图像转化为灰度图像（单通道）
    let img = img.to_luma8();

    // 创建一个4维张量 [1, 1, height, width]
    let mut tensor = Array4::<f64>::zeros((1, 1, height as usize, width as usize));

    for (x, y, pixel) in img.enumerate_pixels() {
        let intensity = pixel[0] as f64 / 255.0; // 归一化像素值
        tensor[[0, 0, y as usize, x as usize]] = intensity;
    }

    tensor
}

pub fn load_image_dataset(directory: &str, image_size: Option<(u32, u32)>) -> Array4<f64> {
    let mut paths: Vec<_> = fs::read_dir(directory).unwrap()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    // 排序路径
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let mut images = vec![];

    for path in paths {
        let img = image::open(&path).expect("Failed to open image");
        let img = if let Some((width, height)) = image_size {
            img.resize_exact(width, height, FilterType::Nearest)
        } else {
            img
        };
        let (width, height) = img.dimensions();
        let img = img.to_luma8();
        let mut tensor = Array4::<f64>::zeros((1, 1, height as usize, width as usize));
        for (x, y, pixel) in img.enumerate_pixels() {
            let intensity = pixel[0] as f64 / 255.0;
            tensor[[0, 0, y as usize, x as usize]] = intensity;
        }
        images.push(tensor);
    }

    let num_images = images.len();
    let (_, _, height, width) = images[0].dim();
    let mut dataset = Array4::<f64>::zeros((num_images, 1, height, width));
    
    for (i, image) in images.into_iter().enumerate() {
        dataset.slice_mut(s![i, .., .., ..]).assign(&image.slice(s![0, .., .., ..]));
    }

    dataset
}
