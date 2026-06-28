//! BYO-key LLM decision extraction (feature `byo-llm`, PLAN.md §5.3 T3).
//!
//! The deterministic cue-phrase extractor ([`crate::decision`]) is the air-gap default. This
//! is the **opt-in** richer extractor: it asks the user's *own* LLM endpoint —
//! **OpenAI-compatible** (`/v1/chat/completions`) or **Anthropic Messages** (`/v1/messages`) —
//! to distill a turn into the structured `{decision, rationale, rejected}` shape the
//! `memory.recall_decision` MCP tool already returns (§5.2). It is OFF by default and gated
//! behind `--features byo-llm` because it needs **network + an API key**, which the air-gap
//! default forbids; nodes it produces are stamped `extractor = "byo-llm-v1"`.
//!
//! ## Testability
//! The two pure functions — [`build_request`] (prompt → request body) and [`parse_response`]
//! (chat-completion JSON → extractions) — carry all the logic and are unit-tested **with no
//! key and no network**. Only [`extract`] performs the HTTP call; it is a thin wrapper over
//! those two and is exercised against a live endpoint, not in CI.
//!
//! ## Wiring (live)
//! Wired into ingest (`store::extract_decisions` / `llm_augment`): with `--features byo-llm`
//! AND `LORE_LLM_*` set, each turn the deterministic pass flagged is re-distilled by the LLM;
//! any error (no key, 4xx, network) falls back to the deterministic candidate for that turn,
//! so decisions are never dropped. The default build never compiles or calls this.

use serde::Deserialize;

/// A distilled decision — the `memory.recall_decision` schema (PLAN.md §5.2). `rationale` and
/// `rejected` are optional so a terse model reply still parses.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct DecisionExtraction {
    pub decision: String,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub rejected: Vec<String>,
}

/// Which wire dialect the endpoint speaks. OpenAI chat-completions is the default; Anthropic
/// Messages is supported because loregraph's audience (Claude Code users) hold Anthropic keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAi,
    Anthropic,
}

/// Endpoint + credentials, read from the environment so nothing is hard-coded or logged.
pub struct LlmConfig {
    /// Full URL: an OpenAI-compatible `/v1/chat/completions` (OpenAI, OpenRouter, vLLM, …) or
    /// Anthropic's `/v1/messages`.
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub provider: Provider,
}

impl LlmConfig {
    /// Build from `LORE_LLM_ENDPOINT`, `LORE_LLM_API_KEY`, `LORE_LLM_MODEL`. Returns `None`
    /// when not fully configured, so a caller can cleanly fall back to the deterministic
    /// extractor instead of erroring. The provider comes from `LORE_LLM_PROVIDER`
    /// (`openai` | `anthropic`), else it is inferred from the endpoint host.
    pub fn from_env() -> Option<Self> {
        let endpoint = std::env::var("LORE_LLM_ENDPOINT").ok()?;
        let api_key = std::env::var("LORE_LLM_API_KEY").ok()?;
        let model = std::env::var("LORE_LLM_MODEL").ok()?;
        if endpoint.is_empty() || api_key.is_empty() || model.is_empty() {
            return None;
        }
        let provider = match std::env::var("LORE_LLM_PROVIDER").ok().as_deref() {
            Some("anthropic") => Provider::Anthropic,
            Some("openai") => Provider::OpenAi,
            _ if endpoint.contains("anthropic") => Provider::Anthropic,
            _ => Provider::OpenAi,
        };
        Some(LlmConfig { endpoint, api_key, model, provider })
    }
}

const SYSTEM_PROMPT: &str = "You extract durable engineering DECISIONS from a chat transcript \
turn. Return ONLY a JSON object {\"decisions\":[...]} (no prose, no fences) where each entry \
has keys: \"decision\" (one sentence, what was chosen), \"rationale\" (why, may be empty), \
\"rejected\" (array of alternatives turned down, may be empty). If the turn contains no \
decision, return {\"decisions\":[]}.";

/// Build the request body for a turn's `text`, in the `provider`'s dialect. Pure — no network.
pub fn build_request(provider: Provider, model: &str, text: &str) -> serde_json::Value {
    match provider {
        Provider::OpenAi => serde_json::json!({
            "model": model,
            "temperature": 0,
            "response_format": { "type": "json_object" }, // ignored by endpoints that don't grok it
            "messages": [
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": text },
            ],
        }),
        // Anthropic Messages: `system` is a top-level field, `max_tokens` is required, and
        // there is no response_format — the system prompt carries the JSON contract.
        Provider::Anthropic => serde_json::json!({
            "model": model,
            "max_tokens": 1024,
            "temperature": 0,
            "system": SYSTEM_PROMPT,
            "messages": [ { "role": "user", "content": text } ],
        }),
    }
}

/// Parse a response body into decisions, in the `provider`'s dialect. Pure — no network.
///
/// Extracts the model's text (OpenAI `choices[0].message.content`; Anthropic `content[0].text`),
/// strips an optional ```json fence, then accepts a JSON **array**, a single **object**, or a
/// `{"decisions":[...]}` wrapper. A missing/garbled body is an `Err`; an empty array is `Ok`.
pub fn parse_response(
    provider: Provider,
    body: &serde_json::Value,
) -> anyhow::Result<Vec<DecisionExtraction>> {
    let content = match provider {
        Provider::OpenAi => body
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow::anyhow!("no choices[0].message.content in OpenAI response"))?,
        Provider::Anthropic => body
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|blocks| blocks.iter().find_map(|b| b.get("text").and_then(|t| t.as_str())))
            .ok_or_else(|| anyhow::anyhow!("no content[].text in Anthropic response"))?,
    };

    let cleaned = strip_code_fence(content.trim());
    // Do NOT put `cleaned` in the error — `llm_augment` logs this, and the model's text can
    // echo transcript content; report only a safe length + the parse error.
    let value: serde_json::Value = serde_json::from_str(cleaned)
        .map_err(|e| anyhow::anyhow!("LLM content was not JSON ({e}); {} bytes returned", cleaned.len()))?;

    // Accept: a bare array, a `{decisions: [...]}` wrapper, or a single decision object.
    let items: Vec<serde_json::Value> = match value {
        serde_json::Value::Array(a) => a,
        serde_json::Value::Object(ref o) if o.get("decisions").map(|d| d.is_array()) == Some(true) => {
            o["decisions"].as_array().cloned().unwrap_or_default()
        }
        obj @ serde_json::Value::Object(_) => vec![obj],
        other => anyhow::bail!("LLM JSON was neither array nor object: {other}"),
    };

    let mut out = Vec::new();
    for it in items {
        // Skip malformed entries (e.g. missing "decision") rather than failing the whole batch.
        if let Ok(d) = serde_json::from_value::<DecisionExtraction>(it) {
            if !d.decision.trim().is_empty() {
                out.push(d);
            }
        }
    }
    Ok(out)
}

/// Strip a leading ```json / ``` fence and trailing ``` that some models wrap JSON in.
fn strip_code_fence(s: &str) -> &str {
    let s = s.trim();
    let Some(rest) = s.strip_prefix("```") else {
        return s;
    };
    // Drop an optional language tag on the first line (e.g. "json\n").
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.trim_start_matches('\n');
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

/// Call the configured LLM to extract decisions from `text`.
///
/// **Network path** — only this function makes an HTTP request (blocking; ingest is sync and
/// off the hot path). It is a thin wrapper over the unit-tested [`build_request`] /
/// [`parse_response`]; verify it against a live endpoint, not in CI. Errors propagate so the
/// caller can fall back to the deterministic extractor.
#[cfg(feature = "byo-llm")]
pub fn extract(cfg: &LlmConfig, text: &str) -> anyhow::Result<Vec<DecisionExtraction>> {
    let body = build_request(cfg.provider, &cfg.model, text);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build HTTP client: {e}"))?;
    let req = client.post(&cfg.endpoint).json(&body);
    // Per-provider auth: OpenAI uses a bearer token; Anthropic uses x-api-key + a version header.
    let req = match cfg.provider {
        Provider::OpenAi => req.bearer_auth(&cfg.api_key),
        Provider::Anthropic => req
            .header("x-api-key", &cfg.api_key)
            .header("anthropic-version", "2023-06-01"),
    };
    let resp = req
        .send()
        .map_err(|e| anyhow::anyhow!("LLM request to {} failed: {e}", cfg.endpoint))?
        .error_for_status()
        .map_err(|e| anyhow::anyhow!("LLM endpoint returned an error status: {e}"))?;
    let json: serde_json::Value = resp
        .json()
        .map_err(|e| anyhow::anyhow!("LLM response was not JSON: {e}"))?;
    parse_response(cfg.provider, &json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_request_openai_and_anthropic_shapes() {
        let oa = build_request(Provider::OpenAi, "gpt-4o-mini", "Let's use redb.");
        assert_eq!(oa["model"], "gpt-4o-mini");
        assert_eq!(oa["response_format"]["type"], "json_object");
        assert_eq!(oa["messages"][0]["role"], "system");
        assert_eq!(oa["messages"][1]["content"], "Let's use redb.");

        let an = build_request(Provider::Anthropic, "claude-haiku-4-5", "Let's use redb.");
        assert_eq!(an["model"], "claude-haiku-4-5");
        assert_eq!(an["max_tokens"], 1024, "Anthropic requires max_tokens");
        assert!(an["system"].is_string(), "system is top-level for Anthropic");
        assert_eq!(an["messages"][0]["role"], "user");
        assert!(an.get("response_format").is_none(), "Anthropic has no response_format");
    }

    /// OpenAI chat-completions envelope around assistant `content`.
    fn completion(content: &str) -> serde_json::Value {
        serde_json::json!({ "choices": [ { "message": { "role": "assistant", "content": content } } ] })
    }
    /// Anthropic Messages envelope around assistant `content`.
    fn anthropic(content: &str) -> serde_json::Value {
        serde_json::json!({ "content": [ { "type": "text", "text": content } ] })
    }

    #[test]
    fn parses_a_json_array_of_decisions() {
        let body = completion(
            r#"[{"decision":"Use redb for the durable store","rationale":"crash-safe, pure-Rust","rejected":["sled","sqlite"]}]"#,
        );
        let got = parse_response(Provider::OpenAi, &body).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].decision, "Use redb for the durable store");
        assert_eq!(got[0].rejected, vec!["sled", "sqlite"]);
    }

    #[test]
    fn parses_anthropic_content_block() {
        let body = anthropic(r#"{"decisions":[{"decision":"Use redb","rationale":"pure-Rust"}]}"#);
        let got = parse_response(Provider::Anthropic, &body).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].decision, "Use redb");
        // wrong dialect → error (Anthropic body has no choices[].message.content)
        assert!(parse_response(Provider::OpenAi, &body).is_err());
    }

    #[test]
    fn parses_a_single_object_and_a_fenced_block() {
        let body = completion(r#"{"decision":"Adopt BM25 for the lexical fallback"}"#);
        let got = parse_response(Provider::OpenAi, &body).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].rationale, "", "rationale defaults when omitted");

        let body = completion("```json\n[{\"decision\":\"Pin gix 0.83\"}]\n```");
        let got = parse_response(Provider::OpenAi, &body).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].decision, "Pin gix 0.83");
    }

    #[test]
    fn empty_array_is_ok_and_garbage_errors() {
        assert_eq!(parse_response(Provider::OpenAi, &completion("[]")).unwrap().len(), 0);
        assert_eq!(parse_response(Provider::OpenAi, &completion(r#"[{"foo":1}]"#)).unwrap().len(), 0);
        assert!(parse_response(Provider::OpenAi, &completion("I could not find a decision.")).is_err());
        assert!(parse_response(Provider::OpenAi, &serde_json::json!({ "choices": [] })).is_err());
    }

    #[test]
    fn config_from_env_needs_all_three() {
        // Exercise from_env() directly to validate the actual env-var names used in the impl.
        // In CI none of the three vars are set, so from_env() must return None.
        // Skip the assertion when all three happen to be configured in the caller's environment.
        let all_set = std::env::var("LORE_LLM_ENDPOINT").map(|v| !v.is_empty()).unwrap_or(false)
            && std::env::var("LORE_LLM_API_KEY").map(|v| !v.is_empty()).unwrap_or(false)
            && std::env::var("LORE_LLM_MODEL").map(|v| !v.is_empty()).unwrap_or(false);
        if !all_set {
            assert!(
                LlmConfig::from_env().is_none(),
                "from_env() must return None when any of \
                 LORE_LLM_ENDPOINT/LORE_LLM_API_KEY/LORE_LLM_MODEL is absent or empty"
            );
        }
    }
}
