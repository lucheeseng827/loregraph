//! Secret redaction on ingest.
//!
//! Chat transcripts are dense with API keys, `.env` dumps, and bearer tokens. Redaction
//! runs in the connector's `lower()` step — **before** anything is persisted, embedded, or
//! hashed — replacing each match with a stable `<redacted:kind:blake3-8>` token so dedup
//! still works and the durable store never contains plaintext secrets (PLAN.md §3, §5).
//! The pattern set extends the the scanner chassis scanner heuristics; this PoC ships the
//! high-precision prefix patterns. Entropy-based catch-all and `lore redact --audit` are
//! tracked for MVP.

use regex::Regex;
use std::sync::OnceLock;

struct Rule {
    kind: &'static str,
    re: Regex,
}

fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES
        .get_or_init(|| {
            let mk = |kind, pat: &str| Rule {
                kind,
                re: Regex::new(pat).expect("static redaction regex"),
            };
            vec![
                mk("anthropic_key", r"sk-ant-[A-Za-z0-9_\-]{20,}"),
                mk("openai_key", r"sk-[A-Za-z0-9]{20,}"),
                mk("aws_access_key", r"AKIA[0-9A-Z]{16}"),
                mk("slack_token", r"xox[abpr]-[0-9A-Za-z-]{10,}"),
                mk("github_token", r"gh[pousr]_[A-Za-z0-9]{20,}"),
                mk("google_key", r"AIza[0-9A-Za-z_\-]{30,}"),
                // Match the WHOLE PEM block (header + base64 body + footer), not just the
                // header, so the key material is redacted too. `(?s)` lets `.` cross newlines.
                mk(
                    "private_key",
                    r"(?s)-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
                ),
                mk("bearer", r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}"),
                mk(
                    "env_assignment",
                    r"(?im)^\s*[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|API)[A-Z0-9_]*\s*=\s*\S+",
                ),
            ]
        })
        .as_slice()
}

/// Replace every secret-looking span with `<redacted:kind:hash8>`. Idempotent: re-running
/// on already-redacted text is a no-op (the replacement tokens match no rule).
pub fn redact(input: &str) -> String {
    let mut out = input.to_string();
    for rule in rules() {
        out = rule
            .re
            .replace_all(&out, |caps: &regex::Captures| {
                let m = &caps[0];
                let h = blake3::hash(m.as_bytes());
                let hex: String = h.as_bytes()[..4].iter().map(|b| format!("{b:02x}")).collect();
                format!("<redacted:{}:{}>", rule.kind, hex)
            })
            .into_owned();
    }
    out
}

/// True if redaction changed the text (used by `lore doctor` to flag secret-dense sources).
pub fn contains_secret(input: &str) -> bool {
    rules().iter().any(|r| r.re.is_match(input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_keys_but_keeps_shape() {
        // Assemble the key at runtime so no secret-shaped literal lives in source (the public
        // mirror's secret scanner would otherwise flag it); redaction is still fully exercised.
        let key = format!("sk-{}", "a".repeat(30));
        let raw = format!("use OPENAI_API_KEY={key} then call it");
        let red = redact(&raw);
        assert!(!red.contains(&key));
        assert!(red.contains("<redacted:"));
        // idempotent
        assert_eq!(redact(&red), red);
    }

    #[test]
    fn redacts_whole_pem_block_not_just_header() {
        // Build the PEM markers at runtime (no literal `BEGIN … PRIVATE KEY` in source).
        let kind = "OPENSSH";
        let begin = format!("-----BEGIN {kind} PRIVATE KEY-----");
        let end = format!("-----END {kind} PRIVATE KEY-----");
        let body = "b3BlbnNzaC1rZXktdjEAAAAA";
        let raw = format!("key:\n{begin}\n{body}\nQAAAAdz\n{end}\ndone");
        let red = redact(&raw);
        assert!(!red.contains(body), "key body redacted");
        assert!(!red.contains(&end), "footer redacted");
        assert!(red.contains("<redacted:private_key:"));
    }

    #[test]
    fn leaves_clean_text_untouched() {
        let clean = "we decided to use redb for the durable store";
        assert_eq!(redact(clean), clean);
        assert!(!contains_secret(clean));
    }
}
