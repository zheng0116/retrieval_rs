use candle::{DType, Device, Tensor};
use std::path::PathBuf;

#[derive(Debug)]
pub struct RetrievalResult {
    pub path: PathBuf,
    pub similarity: f32,
}

impl RetrievalResult {
    pub fn new(path: PathBuf, similarity: f32) -> Self {
        Self { path, similarity }
    }
}

pub fn get_image_files(dir_path: &str) -> anyhow::Result<Vec<PathBuf>> {
    let mut image_files = Vec::new();
    let supported_extensions = ["jpg", "jpeg", "png", "gif", "bmp"];

    for entry in std::fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(extension) = path.extension() {
                if let Some(ext_str) = extension.to_str() {
                    if supported_extensions.contains(&ext_str.to_lowercase().as_str()) {
                        image_files.push(path);
                    }
                }
            }
        }
    }

    Ok(image_files)
}

pub fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    Ok(img)
}

pub fn compute_similarity(
    image_features: &Tensor,
    text_features: &Tensor,
) -> anyhow::Result<Tensor> {
    let image_features = if image_features.dims().len() == 1 {
        image_features.unsqueeze(0)?
    } else {
        image_features.clone()
    };

    let text_features = if text_features.dims().len() == 1 {
        text_features.unsqueeze(0)?
    } else {
        text_features.clone()
    };

    let i_norm = image_features
        .sqr()?
        .sum(1)?
        .sqrt()?
        .reshape((image_features.dim(0)?, 1))?;
    let t_norm = text_features
        .sqr()?
        .sum(1)?
        .sqrt()?
        .reshape((text_features.dim(0)?, 1))?;

    let normalized_image_features = image_features.broadcast_div(&i_norm)?;
    let normalized_text_features = text_features.broadcast_div(&t_norm)?;

    let similarities =
        normalized_image_features.matmul(&normalized_text_features.transpose(0, 1)?)?;

    Ok(similarities)
}
