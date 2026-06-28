//! Codex CLI connector.
//!
//! Reads the rollout transcripts Codex writes under
//! `~/.codex/sessions/YYYY/MM/DD/rollout-<ISO>-<UUID>.jsonl`. Each line is a
//! `{timestamp, type, payload}` envelope (PLAN.md §3.1). The format is **undocumented and
//! has three incompatible eras** (2025/08, mid, ≥0.44), so every field read is *lenient* —
//! probed from both the envelope and the `payload`, in several spellings — and an
//! unrecognized record lowers to `MessageKind::Unknown`, never a hard error.
//!
//! The key shape that trips naive parsers (corrected per §3.1): there is **no top-level
//! `exec_command`**. Tool use is a `response_item` whose payload `type` is `function_call`
//! (`name`, `arguments` — a JSON **string** — and `call_id`), paired to a
//! `function_call_output` (`call_id`, `output`). We attach the files a `function_call`
//! touched (parsed out of its `arguments`) to the turn, and treat the `*_output` as an
//! `Unknown` result record.

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::SessionSource;
use crate::model::{AgentKind, CanonicalSession, MessageKind, Turn};
use crate::redact;

#[derive(Default)]
pub struct Codex;

impl Codex {
    fn sessions_dir() -> Option<PathBuf> {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
        let dir = PathBuf::from(home).join(".codex").join("sessions");
        dir.is_dir().then_some(dir)
    }
}

impl SessionSource for Codex {
    fn id(&self) -> &'static str {
        "codex"
    }

    fn discover(&self) -> anyhow::Result<Vec<PathBuf>> {
        let Some(root) = Self::sessions_dir() else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        // sessions/YYYY/MM/DD/rollout-*.jsonl — bounded depth so we don't walk forever.
        for entry in walkdir::WalkDir::new(&root)
            .max_depth(5)
            .into_iter()
            .filter_map(Result::ok)
        {
            let p = entry.path();
            if p.is_file()
                && p.extension().is_some_and(|e| e == "jsonl")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("rollout-"))
            {
                out.push(p.to_path_buf());
            }
        }
        out.sort();
        Ok(out)
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading transcript {}", path.display()))?;

        // `rollout-<ISO>-<UUID>` → keep the trailing canonical UUID (its own hyphens mean a
        // naive `rsplit_once('-')` would keep only the last group). Validate the 36-char
        // `8-4-4-4-12` shape; else fall back to the last hyphen-group, else the whole stem.
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
        let native_session_id = stem
            .get(stem.len().saturating_sub(36)..)
            .filter(|u| {
                u.len() == 36
                    && u.bytes().enumerate().all(|(i, b)| match i {
                        8 | 13 | 18 | 23 => b == b'-',
                        _ => b.is_ascii_hexdigit(),
                    })
            })
            .or_else(|| stem.rsplit_once('-').map(|(_, suffix)| suffix).filter(|u| u.len() >= 8))
            .unwrap_or(stem)
            .to_string();

        let mut turns: Vec<Turn> = Vec::new();
        let mut repo_root: Option<String> = None;
        let mut commit: Option<String> = None;
        let mut connector_version: Option<String> = None;
        let mut started_at_ms: Option<i64> = None;

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue; // lenient: skip a torn/corrupt line, never fatal
            };
            // Era drift: some records nest the meat under `payload`, some are flat. Probe both.
            let null = serde_json::Value::Null;
            let payload = v.get("payload").unwrap_or(&v);

            if repo_root.is_none() {
                repo_root = probe_str(&v, payload, &["cwd"]).map(str::to_string);
            }
            if commit.is_none() {
                // Accept a real SHA only (≥7 hex), never a branch name.
                commit = payload
                    .get("git")
                    .and_then(|g| g.get("commit"))
                    .and_then(|c| c.as_str())
                    .or_else(|| probe_str(&v, payload, &["commit", "git_commit"]))
                    .filter(|s| s.len() >= 7 && s.bytes().all(|b| b.is_ascii_hexdigit()))
                    .map(str::to_string);
            }
            if connector_version.is_none() {
                connector_version =
                    probe_str(&v, payload, &["cli_version", "version"]).map(str::to_string);
            }

            let ts_ms = probe_str(&v, payload, &["timestamp"]).and_then(parse_iso_ms);
            if started_at_ms.is_none() {
                started_at_ms = ts_ms;
            }

            // The record's logical type can sit on the envelope or the payload.
            let typ = payload
                .get("type")
                .or_else(|| v.get("type"))
                .and_then(|t| t.as_str())
                .unwrap_or("");

            let mut text_buf = String::new();
            let mut touched = Vec::new();
            let kind = match typ {
                "message" => {
                    let role = payload.get("role").and_then(|r| r.as_str()).unwrap_or("");
                    extract_content(payload.get("content").unwrap_or(&null), &mut text_buf);
                    match role {
                        "user" => MessageKind::User,
                        "assistant" => MessageKind::Assistant,
                        "system" => MessageKind::System,
                        _ => MessageKind::Unknown,
                    }
                }
                "function_call" => {
                    // The agent took an action; record it as an assistant turn so the files it
                    // touched draw `Touches` edges. `arguments` is a JSON *string*.
                    if let Some(name) = payload.get("name").and_then(|n| n.as_str()) {
                        text_buf.push_str("[tool] ");
                        text_buf.push_str(name);
                    }
                    if let Some(args) = payload.get("arguments").and_then(|a| a.as_str()) {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(args) {
                            collect_paths(&parsed, &mut touched);
                        }
                    }
                    MessageKind::Assistant
                }
                // The tool result carries no memory value; keep it as an Unknown atom.
                "function_call_output" => MessageKind::Unknown,
                _ => MessageKind::Unknown,
            };

            let text_buf = redact::redact(&text_buf);
            turns.push(Turn {
                index: turns.len(),
                kind,
                text: text_buf,
                ts_ms,
                touched_files: touched,
                raw: v,
            });
        }

        Ok(CanonicalSession {
            native_session_id,
            agent: AgentKind::Codex,
            repo_root,
            commit,
            started_at_ms,
            source_path: Some(path.display().to_string()),
            connector_version,
            turns,
        })
    }
}

/// Probe a string `key` from the envelope first, then the payload — covers the era drift.
fn probe_str<'a>(
    v: &'a serde_json::Value,
    payload: &'a serde_json::Value,
    keys: &[&str],
) -> Option<&'a str> {
    for k in keys {
        if let Some(s) = v.get(k).and_then(|x| x.as_str()) {
            return Some(s);
        }
        if let Some(s) = payload.get(k).and_then(|x| x.as_str()) {
            return Some(s);
        }
    }
    None
}

/// Pull human text out of a Responses-API `content`: a string, or an array of typed blocks
/// (`input_text` / `output_text` / `text`). Unknown block shapes are ignored.
fn extract_content(content: &serde_json::Value, text: &mut String) {
    if let Some(s) = content.as_str() {
        text.push_str(s);
        return;
    }
    let Some(blocks) = content.as_array() else {
        return;
    };
    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("input_text" | "output_text" | "text") => {
                if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
            }
            _ => {}
        }
    }
}

/// Find file paths in a parsed tool-call `arguments` (Codex shell/apply_patch shapes).
fn collect_paths(input: &serde_json::Value, touched: &mut Vec<String>) {
    for key in ["file_path", "path", "filePath"] {
        if let Some(p) = input.get(key).and_then(|x| x.as_str()) {
            if !touched.iter().any(|t| t == p) {
                touched.push(p.to_string());
            }
        }
    }
    // some shapes nest under `input` / `args` / a list of edits
    for nest in ["input", "args"] {
        if let Some(o) = input.get(nest) {
            collect_paths(o, touched);
        }
    }
    if let Some(edits) = input.get("edits").and_then(|x| x.as_array()) {
        for e in edits {
            collect_paths(e, touched);
        }
    }
}

/// RFC3339 timestamp → epoch millis.
fn parse_iso_ms(s: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_codex_rollout_lenient() {
        let mut f = tempfile::Builder::new()
            .prefix("rollout-2026-")
            .suffix(".jsonl")
            .tempfile()
            .unwrap();
        // Assemble the fake key at runtime so no token-shaped literal lives in source (keeps
        // the OSS secret scanner happy); it still triggers the redactor's `sk-ant-…` rule.
        let secret = format!("sk-ant-{}", "a".repeat(30));
        // session_meta (cwd/git/cli_version), a user + assistant message, a function_call with
        // a secret + a touched file, its output, an unknown type, and a torn line.
        writeln!(f, r#"{{"timestamp":"2026-06-01T10:00:00Z","type":"session_meta","payload":{{"cwd":"/work/demo","cli_version":"0.44.0","git":{{"commit":"a1b2c3d4e5f6"}}}}}}"#).unwrap();
        writeln!(f, r#"{{"timestamp":"2026-06-01T10:00:05Z","type":"response_item","payload":{{"type":"message","role":"user","content":[{{"type":"input_text","text":"use redb for the store"}}]}}}}"#).unwrap();
        writeln!(f, r#"{{"timestamp":"2026-06-01T10:00:10Z","type":"response_item","payload":{{"type":"message","role":"assistant","content":[{{"type":"output_text","text":"Let's use redb. key {secret}"}}]}}}}"#).unwrap();
        writeln!(f, r#"{{"timestamp":"2026-06-01T10:00:12Z","type":"response_item","payload":{{"type":"function_call","name":"shell","arguments":"{{\"file_path\":\"src/store.rs\",\"token\":\"{secret}\"}}","call_id":"c1"}}}}"#).unwrap();
        writeln!(f, r#"{{"timestamp":"2026-06-01T10:00:13Z","type":"response_item","payload":{{"type":"function_call_output","call_id":"c1","output":"ok"}}}}"#).unwrap();
        writeln!(f, r#"{{"timestamp":"2026-06-01T10:00:14Z","type":"turn_context","payload":{{"foo":1}}}}"#).unwrap();
        writeln!(f, r#"{{ not json"#).unwrap();
        f.flush().unwrap();

        let s = Codex.read_session(f.path()).unwrap();
        assert_eq!(s.agent, AgentKind::Codex);
        assert_eq!(s.repo_root.as_deref(), Some("/work/demo"));
        assert_eq!(s.commit.as_deref(), Some("a1b2c3d4e5f6"), "real SHA accepted");
        assert_eq!(s.connector_version.as_deref(), Some("0.44.0"));
        assert_eq!(s.turns.len(), 6, "torn line dropped, the rest kept");
        assert_eq!(s.turns[1].kind, MessageKind::User);
        assert_eq!(s.turns[1].text, "use redb for the store");
        assert_eq!(s.turns[2].kind, MessageKind::Assistant);
        // the function_call turn carries the touched file and is redacted
        assert_eq!(s.turns[3].kind, MessageKind::Assistant);
        assert_eq!(s.turns[3].touched_files, vec!["src/store.rs"]);
        assert!(!s.turns[2].text.contains(&secret), "message-text secret redacted");
        assert!(s.turns[2].text.contains("redb"), "non-secret message text survives");
        assert!(!s.turns[3].text.contains(&secret), "tool turn stays secret-free");
        assert_eq!(s.turns[4].kind, MessageKind::Unknown, "function_call_output");
        assert_eq!(s.turns[5].kind, MessageKind::Unknown, "unknown type");
    }

    #[test]
    fn keeps_the_full_uuid_session_id() {
        // A canonical `rollout-<ISO>-<UUID>.jsonl` name: the UUID (with its own hyphens) must
        // survive whole, not be truncated to its last group.
        let dir = tempfile::tempdir().unwrap();
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let path = dir.path().join(format!("rollout-2026-06-01T10-00-00-{uuid}.jsonl"));
        std::fs::write(&path, "{\"type\":\"session_meta\",\"payload\":{\"cwd\":\"/x\"}}\n").unwrap();
        let s = Codex.read_session(&path).unwrap();
        assert_eq!(s.native_session_id, uuid, "the full UUID is the native id");
    }
}
