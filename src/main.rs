use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;
use clap::Parser;
use retrieval_rs::{
    model::{clip::get_tokenizer, clip::tokenize_text, clip::ImageFeatureCache},
    processor::utils::{compute_similarity, get_image_files, RetrievalResult},
    Args,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let start_time = Instant::now();
    let args = Args::parse();

    println!("Initializing system...");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.repo(hf_hub::Repo::with_revision(
                "openai/clip-vit-base-patch32".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/15".to_string(),
            ));
            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };

    let tokenizer = get_tokenizer(args.tokenizer)?;
    let config = clip::ClipConfig::vit_base_patch32();
    let device = candle_examples::device(args.cpu)?;

    let image_paths = match args.image_dir {
        Some(dir) => get_image_files(&dir)?,
        None => {
            eprintln!("Please specify an image directory using --image-dir!");
            return Ok(());
        }
    };

    if image_paths.is_empty() {
        println!("No images found in the specified directory!");
        return Ok(());
    }

    println!("Found {} images", image_paths.len());

    println!("Loading CLIP model...");
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &device)? };
    let model = clip::ClipModel::new(vb, &config)?;

    let mut feature_cache = if let Some(cache_path) = &args.cache_file {
        if std::path::Path::new(cache_path).exists() {
            println!("Loading feature cache...");
            ImageFeatureCache::load_cache(cache_path)?
        } else {
            println!("Creating new feature cache...");
            ImageFeatureCache::new()
        }
    } else {
        ImageFeatureCache::new()
    };

    let query = args.query.unwrap_or_else(|| "a photo".to_string());
    println!("Processing query: '{}'", query);

    let query_tokens = tokenize_text(&query, &tokenizer, &device)?;
    let text_features = model.get_text_features(&query_tokens)?;

    println!("Processing images...");
    let batch_size = 32;
    let mut all_results = Vec::new();

    for (i, chunk) in image_paths.chunks(batch_size).enumerate() {
        println!(
            "Processing batch {}/{}",
            i + 1,
            (image_paths.len() + batch_size - 1) / batch_size
        );

        let mut batch_features = Vec::new();
        for path in chunk {
            let features = feature_cache.get_or_compute(path, &model, &device, &config)?;
            batch_features.push(features);
        }

        let batch_features = Tensor::cat(&batch_features, 0)?;
        let similarities = compute_similarity(&batch_features, &text_features)?;
        let similarities = similarities.flatten_all()?.to_vec1::<f32>()?;

        for (j, similarity) in similarities.iter().enumerate() {
            if *similarity >= args.similarity_threshold {
                all_results.push(RetrievalResult::new(chunk[j].clone(), *similarity));
            }
        }
    }

    if args.save_cache {
        if let Some(cache_path) = args.cache_file {
            println!("Saving feature cache...");
            feature_cache.save_cache(&cache_path)?;
        }
    }

    all_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

    println!("\nSearch Results for: '{}'", query);
    println!("Similarity threshold: {}", args.similarity_threshold);
    println!("Time taken: {:.2?}", start_time.elapsed());
    println!("\nTop {} results:", args.top_k);

    for (i, result) in all_results.iter().take(args.top_k).enumerate() {
        println!(
            "{}. {} (similarity: {:.4})",
            i + 1,
            result.path.display(),
            result.similarity
        );
    }

    if all_results.is_empty() {
        println!("\nNo images found matching the query with similarity above threshold.");
    }

    Ok(())
}
