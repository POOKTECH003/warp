use std::collections::HashMap;
use serde::Deserialize;
use crate::ai::llms::{LLMInfo, LLMUsageMetadata, LLMProvider, LLMContextWindow};

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Deserialize)]
struct OllamaModel {
    name: String,
}

/// Fetches local Ollama models and converts them into Warp's `LLMInfo` format.
/// These models will be treated as Custom Endpoints automatically injected into the model picker.
pub async fn fetch_ollama_models() -> Vec<LLMInfo> {
    // Attempt to quickly read from Ollama's local endpoint
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(1500))
        .build()
        .unwrap_or_default();

    let response = match client.get("http://localhost:11434/api/tags").send().await {
        Ok(res) => res,
        Err(_) => return vec![],
    };

    let tags_response = match response.json::<OllamaTagsResponse>().await {
        Ok(json) => json,
        Err(_) => return vec![],
    };

    tags_response.models.into_iter().map(|model| {
        let label = format!("Ollama: {}", model.name);
        // Prefix with a UUID-like structure or just a unique config_key
        let config_key = format!("ollama-auto-{}", model.name);
        
        LLMInfo {
            display_name: label.clone(),
            base_model_name: label,
            id: config_key.into(),
            reasoning_level: None,
            usage_metadata: LLMUsageMetadata {
                request_multiplier: 1,
                credit_multiplier: None,
            },
            description: Some(format!("Custom 路 Local Ollama")),
            disable_reason: None,
            vision_supported: true,
            spec: None,
            provider: LLMProvider::Unknown,
            host_configs: HashMap::new(),
            discount_percentage: None,
            context_window: LLMContextWindow::default(),
        }
    }).collect()
}
