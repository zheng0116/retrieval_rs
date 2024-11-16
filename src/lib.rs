pub mod model;
pub mod processor;

use clap::Parser;

#[derive(Parser)]
pub struct Args {
    #[arg(long)]
    pub model: Option<String>,

    #[arg(long)]
    pub tokenizer: Option<String>,

    #[arg(long)]
    pub image_dir: Option<String>,

    #[arg(long)]
    pub cpu: bool,

    #[arg(long)]
    pub query: Option<String>,

    #[arg(long, default_value = "0.2")]
    pub similarity_threshold: f32,

    #[arg(long, default_value = "5")]
    pub top_k: usize,

    #[arg(long)]
    pub cache_file: Option<String>,

    #[arg(long)]
    pub save_cache: bool,
}
