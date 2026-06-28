//! End-to-end: index the golden Claude Code transcript + the fixture repo, then recall the
//! storage decision and prove it survives a reload. This is the PoC's "one demo" in test
//! form (PLAN.md §9).

use std::path::PathBuf;

use lore::ingest::claude_code::ClaudeCode;
use lore::ingest::SessionSource;
use lore::store::Store;

#[test]
fn index_then_recall_the_decision() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let fixture = PathBuf::from(manifest).join("examples/sessions/demo-claude-code.jsonl");
    let repo = PathBuf::from(manifest).join("examples/repo_fixture");
    let dir = tempfile::tempdir().unwrap();

    // Scope the first store so its redb file lock is released before we reopen below.
    let count;
    {
        let mut store = Store::open(dir.path()).unwrap();
        let sess = ClaudeCode.read_session(&fixture).unwrap();
        assert!(sess.turns.len() >= 4, "user/assistant/unknown/user turns parsed");
        store.ingest_session(&sess).unwrap();
        store.ingest_repo(&repo).unwrap();
        store.save().unwrap(); // checkpoint into redb, truncate the WAL

        // Session + turns + at least one Decision + repo File/Symbol nodes.
        assert!(store.graph.node_count() > 6, "got {}", store.graph.node_count());

        // The agent's storage decision is recallable — nobody needs to re-ask.
        let res = lore::ask::recall(&store, "what storage engine did we choose", 10);
        assert!(!res.results.is_empty());
        let blob = res
            .results
            .iter()
            .map(|e| format!("{} {}", e.label, e.snippet))
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        assert!(blob.contains("redb"), "recall should surface the redb decision; got: {blob}");
        count = store.graph.node_count();
    }

    // Reopen: the graph rehydrates from redb with the same shape (durable round-trip).
    let reloaded = Store::open(dir.path()).unwrap();
    assert_eq!(reloaded.graph.node_count(), count);
    assert!(!reloaded.index.is_empty());
}
