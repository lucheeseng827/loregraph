//! Rule-based decision extraction (the PoC default extractor).
//!
//! Decisions are the value nodes (PLAN.md §2). The PoC extracts them with a deterministic
//! cue-phrase heuristic — zero-COGS, air-gap, and honest about being approximate
//! (`confidence ≈ 0.5`, extractor `deterministic-v1`). The BYO-key LLM extractor
//! (`byo-llm-v1`) is a Beta swap behind the same seam; the wedge does not depend on it.

use crate::model::{MessageKind, Turn};

/// Cue phrases that mark a decision in agent chatter. Lowercased match.
const CUES: &[&str] = &[
    "we'll use",
    "let's use",
    "i'll use",
    "decided to",
    "decision:",
    "go with",
    "going with",
    "switch to",
    "switching to",
    "instead of",
    "chose ",
    "we should use",
    "rejected",
    "trade-off",
    "tradeoff",
    "the plan is",
];

/// A candidate decision lifted from a turn.
#[derive(Clone)]
pub struct Candidate {
    pub turn_index: usize,
    pub text: String,
    pub confidence: f32,
    /// Why it was chosen (empty for the deterministic extractor; filled by `byo-llm`).
    pub rationale: String,
    /// Alternatives turned down (empty for deterministic; filled by `byo-llm`).
    pub rejected: Vec<String>,
    /// Which extractor produced this, stamped into provenance: `deterministic-v1` | `byo-llm-v1`.
    pub extractor: &'static str,
}

/// Markers Claude Code (and friends) wrap around non-memory chatter: slash-command
/// invocations, local-command output, and interrupt notices. A turn carrying one of these
/// is UI/control noise, not a memory — it must not become a `Turn` node or a decision
/// candidate (else recall surfaces `<command-name>/clear` as a "memory"). T1, PLAN.md §5.3.
const NOISE_MARKERS: &[&str] = &[
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "[Request interrupted",
];

/// True when a turn carries no recallable value (empty, a slash-command/control wrapper, or
/// a system caveat). Shared by the Turn-node and decision passes so both skip it identically.
pub fn is_low_value_turn(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    if NOISE_MARKERS.iter().any(|m| t.contains(m)) {
        return true;
    }
    // The standing "Caveat: …" banner Claude Code prepends to some user turns.
    t.starts_with("Caveat:")
}

/// Minimum word count for a distilled decision — drops a bare cue match ("Rejected.",
/// "go with it") that carries the cue but no substance. A real decision names what was
/// chosen, so it clears this easily.
const MIN_DECISION_WORDS: usize = 5;

/// Scan a session's turns for decision-shaped sentences. Only considers `User`/`Assistant`
/// turns (matching the store layer), skips control/command-noise, and drops bare cue matches
/// with no substance (T1, PLAN.md §5.3).
pub fn extract(turns: &[Turn]) -> Vec<Candidate> {
    let mut out = Vec::new();
    for t in turns {
        if !matches!(t.kind, MessageKind::User | MessageKind::Assistant)
            || is_low_value_turn(&t.text)
        {
            continue;
        }
        // ASCII-only case fold: preserves byte length so offsets into `lower` stay valid
        // when slicing `t.text` (full `to_lowercase()` can change byte length, e.g. 'ẞ'→'ß',
        // and panic). The cue phrases are all ASCII, so this matches them correctly.
        let lower = t.text.to_ascii_lowercase();
        if let Some(cue) = CUES.iter().find(|c| lower.contains(**c)) {
            if let Some(sentence) = sentence_around(&t.text, &lower, cue) {
                if sentence.split_whitespace().count() < MIN_DECISION_WORDS {
                    continue; // bare cue, no decision content — not worth a node
                }
                out.push(Candidate {
                    turn_index: t.index,
                    text: sentence,
                    confidence: 0.5,
                    rationale: String::new(),
                    rejected: Vec::new(),
                    extractor: "deterministic-v1",
                });
            }
        }
    }
    out
}

/// Extract the sentence containing the first occurrence of `cue` (case-insensitive),
/// trimmed to a sane length.
fn sentence_around(original: &str, lower: &str, cue: &str) -> Option<String> {
    let at = lower.find(cue)?;
    let start = original[..at].rfind(['.', '!', '?', '\n']).map(|i| i + 1).unwrap_or(0);
    let rel_end = original[at..]
        .find(['.', '!', '?', '\n'])
        .map(|i| at + i + 1)
        .unwrap_or(original.len());
    let s = original[start..rel_end].trim();
    if s.is_empty() {
        return None;
    }
    Some(s.chars().take(280).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MessageKind, Turn};

    fn turn(i: usize, k: MessageKind, text: &str) -> Turn {
        Turn {
            index: i,
            kind: k,
            text: text.to_string(),
            ts_ms: None,
            touched_files: vec![],
            raw: serde_json::Value::Null,
        }
    }

    #[test]
    fn lifts_decision_sentences() {
        let turns = vec![
            turn(0, MessageKind::User, "hey can you look at the store"),
            turn(1, MessageKind::Assistant, "Sure. Let's use redb for durability instead of sled. It has clean txns."),
            turn(2, MessageKind::Summary, "we'll use something"),
        ];
        let got = extract(&turns);
        assert_eq!(got.len(), 1, "only the assistant turn (summary skipped)");
        assert!(got[0].text.to_lowercase().contains("let's use redb"));
    }

    #[test]
    fn skips_command_noise_and_bare_cues() {
        let turns = vec![
            // slash-command wrapper — control noise, even though it could match a cue
            turn(0, MessageKind::User, "<command-name>/clear</command-name>"),
            // a cue ("rejected") but no substance — under the word floor
            turn(1, MessageKind::Assistant, "Rejected."),
            // a real decision survives both filters
            turn(2, MessageKind::Assistant, "We'll switch to redb instead of sled for the durable store."),
        ];
        let got = extract(&turns);
        assert_eq!(got.len(), 1, "only the substantive decision survives");
        assert!(got[0].text.to_lowercase().contains("redb"));
    }

    #[test]
    fn low_value_turn_detection() {
        assert!(is_low_value_turn("  "));
        assert!(is_low_value_turn("<command-name>/model</command-name>"));
        assert!(is_low_value_turn("<local-command-stdout>ok</local-command-stdout>"));
        assert!(is_low_value_turn("Caveat: the user ran ..."));
        assert!(!is_low_value_turn("Let's use redb for the durable store."));
    }
}
