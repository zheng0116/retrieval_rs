use anyhow::Error as E;
use candle::{Device, Tensor};
use candle_transformers::models::clip;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    path: PathBuf,
    features: Vec<f32>,
    last_modified: u64,
}

pub struct ImageFeatureCache {
    cache: HashMap<PathBuf, CacheEntry>,
}

impl ImageFeatureCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn load_cache(path: &str) -> anyhow::Result<Self> {
        let cache_content = fs::read_to_string(path)?;
        let cache: HashMap<PathBuf, CacheEntry> = serde_json::from_str(&cache_content)?;
        Ok(Self { cache })
    }

    pub fn save_cache(&self, path: &str) -> anyhow::Result<()> {
        let cache_content = serde_json::to_string_pretty(&self.cache)?;
        fs::write(path, cache_content)?;
        Ok(())
    }

    pub fn get_or_compute(
        &mut self,
        path: &PathBuf,
        model: &clip::ClipModel,
        device: &Device,
        config: &clip::ClipConfig,
    ) -> anyhow::Result<Tensor> {
        let metadata = fs::metadata(path)?;
        let last_modified = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        if let Some(entry) = self.cache.get(path) {
            if entry.last_modified == last_modified {
                return Ok(Tensor::new(entry.features.clone(), device)?);
            }
        }

        let image = crate::processor::utils::load_image(path, config.image_size)?;
        let image = image.to_device(device)?;
        let features = model.get_image_features(&image.unsqueeze(0)?)?;

        let features_vec = features.flatten_all()?.to_vec1::<f32>()?;
        self.cache.insert(
            path.clone(),
            CacheEntry {
                path: path.clone(),
                features: features_vec,
                last_modified,
            },
        );

        Ok(features)
    }
}

pub fn get_tokenizer(tokenizer: Option<String>) -> anyhow::Result<Tokenizer> {
    let tokenizer = match tokenizer {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.repo(hf_hub::Repo::with_revision(
                "openai/clip-vit-base-patch32".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/15".to_string(),
            ));
            api.get("tokenizer.json")?
        }
        Some(file) => file.into(),
    };
    Tokenizer::from_file(tokenizer).map_err(E::msg)
}

pub fn tokenize_text(text: &str, tokenizer: &Tokenizer, device: &Device) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode(text, true).map_err(E::msg)?;
    let tokens = tokens.get_ids().to_vec();
    let input_ids = Tensor::new(vec![tokens], device)?;
    Ok(input_ids)
}
