//! Claude Code connector.
//!
//! Reads the JSONL transcripts Claude Code already writes under
//! `~/.claude/projects/<slug>/*.jsonl` (slug = the project cwd with `/` → `-`). Verified
//! against a live install (PLAN.md §5): the layout is **flat** (older docs claiming a
//! `sessions/` subdir are stale), record `type` is an **open** set (besides
//! `user`/`assistant`/`system`/`summary` you also see `queue-operation`/`attachment`/
//! `last-prompt`/… — all map to `MessageKind::Unknown`), and `usage` carries more than the
//! four stable token fields. So: glob **both** layouts, never hard-code the type list, and
//! retain the raw record losslessly.

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::SessionSource;
use crate::model::{AgentKind, CanonicalSession, MessageKind, Turn};
use crate::redact;

#[derive(Default)]
pub struct ClaudeCode;

impl ClaudeCode {
    fn projects_dir() -> Option<PathBuf> {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
        let dir = PathBuf::from(home).join(".claude").join("projects");
        dir.is_dir().then_some(dir)
    }
}

impl SessionSource for ClaudeCode {
    fn id(&self) -> &'static str {
        "claude_code"
    }

    fn discover(&self) -> anyhow::Result<Vec<PathBuf>> {
        let Some(root) = Self::projects_dir() else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        // Walk projects/<slug>/**.jsonl — covers both the flat layout and a `sessions/`
        // subdir, so we survive either Claude Code version.
        for entry in walkdir::WalkDir::new(&root)
            .max_depth(3)
            .into_iter()
            .filter_map(Result::ok)
        {
            let p = entry.path();
            if p.is_file() && p.extension().is_some_and(|e| e == "jsonl") {
                out.push(p.to_path_buf());
            }
        }
        out.sort();
        Ok(out)
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading transcript {}", path.display()))?;

        let native_session_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
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
            // Lenient: a corrupt/partial line is skipped, never fatal.
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };

            // Session-level context, taken from the first record that carries it.
            if repo_root.is_none() {
                repo_root = v.get("cwd").and_then(|x| x.as_str()).map(str::to_string);
            }
            if commit.is_none() {
                // Only accept an actual commit SHA — NOT `gitBranch` (a branch name like
                // `main` is not a commit identity and would create unjoinable provenance).
                commit = v
                    .get("gitCommit")
                    .or_else(|| v.get("commit"))
                    .and_then(|x| x.as_str())
                    .filter(|s| s.len() >= 7 && s.bytes().all(|b| b.is_ascii_hexdigit()))
                    .map(str::to_string);
            }
            if connector_version.is_none() {
                connector_version = v.get("version").and_then(|x| x.as_str()).map(str::to_string);
            }

            let typ = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
            let kind = match typ {
                "user" => MessageKind::User,
                "assistant" => MessageKind::Assistant,
                "system" => MessageKind::System,
                "summary" => MessageKind::Summary,
                _ => MessageKind::Unknown, // queue-operation / attachment / last-prompt / …
            };

            let ts_ms = v
                .get("timestamp")
                .and_then(|x| x.as_str())
                .and_then(parse_iso_ms);
            if started_at_ms.is_none() {
                started_at_ms = ts_ms;
            }

            let mut text_buf = String::new();
            let mut touched = Vec::new();

            if typ == "summary" {
                if let Some(s) = v.get("summary").and_then(|x| x.as_str()) {
                    text_buf.push_str(s);
                }
            } else if let Some(msg) = v.get("message") {
                extract_message(msg, &mut text_buf, &mut touched);
            }

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
            agent: AgentKind::ClaudeCode,
            repo_root,
            commit,
            started_at_ms,
            source_path: Some(path.display().to_string()),
            connector_version,
            turns,
        })
    }
}

/// Pull human-readable text + edited file paths out of an Anthropic `message` object.
/// `content` is either a string or an array of typed blocks (`text` / `tool_use` /
/// `tool_result`). Unknown block shapes are ignored, never fatal.
fn extract_message(msg: &serde_json::Value, text: &mut String, touched: &mut Vec<String>) {
    let Some(content) = msg.get("content") else {
        return;
    };
    if let Some(s) = content.as_str() {
        text.push_str(s);
        return;
    }
    let Some(blocks) = content.as_array() else {
        return;
    };
    for block in blocks {
        match block.get("type").and_then(|x| x.as_str()) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
            }
            Some("tool_use") => {
                if let Some(input) = block.get("input") {
                    collect_paths(input, touched);
                }
            }
            _ => {}
        }
    }
}

/// Find file paths in a tool_use `input` (Edit/Write/Read/NotebookEdit and multi-edit
/// shapes) so we can draw `Touches` edges from the turn to the file nodes.
fn collect_paths(input: &serde_json::Value, touched: &mut Vec<String>) {
    for key in ["file_path", "path", "notebook_path", "filePath"] {
        if let Some(p) = input.get(key).and_then(|x| x.as_str()) {
            if !touched.iter().any(|t| t == p) {
                touched.push(p.to_string());
            }
        }
    }
    // multi-edit: { edits: [ { file_path }, ... ] }
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
    fn parses_lenient_jsonl_with_unknown_and_redaction() {
        let mut f = tempfile::Builder::new().suffix(".jsonl").tempfile().unwrap();
        // Assemble the fake key at runtime so no token-shaped literal lives in source (keeps
        // the OSS secret scanner happy); it still triggers the redactor's `sk-ant-…` rule.
        let secret = format!("sk-ant-{}", "a".repeat(30));
        // a normal user turn, an assistant turn with a tool_use edit + a secret,
        // an unknown record type, and a torn line — only the torn line is dropped.
        writeln!(f, r#"{{"type":"user","timestamp":"2026-05-02T14:11:00Z","cwd":"/repo","gitBranch":"main","version":"2.1.195","message":{{"role":"user","content":"let's use redb"}}}}"#).unwrap();
        writeln!(f, r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"ok, key is {secret}"}},{{"type":"tool_use","name":"Edit","input":{{"file_path":"src/store.rs"}}}}]}}}}"#).unwrap();
        writeln!(f, r#"{{"type":"queue-operation","foo":1}}"#).unwrap();
        writeln!(f, r#"{{ this is not json"#).unwrap();
        f.flush().unwrap();

        let s = ClaudeCode.read_session(f.path()).unwrap();
        assert_eq!(s.agent, AgentKind::ClaudeCode);
        assert_eq!(s.repo_root.as_deref(), Some("/repo"));
        assert_eq!(s.connector_version.as_deref(), Some("2.1.195"));
        assert_eq!(s.turns.len(), 3, "torn line dropped, the rest kept");
        assert_eq!(s.turns[0].kind, MessageKind::User);
        assert_eq!(s.turns[1].touched_files, vec!["src/store.rs"]);
        assert!(!s.turns[1].text.contains(&secret), "secret redacted");
        assert_eq!(s.turns[2].kind, MessageKind::Unknown);
    }
}
