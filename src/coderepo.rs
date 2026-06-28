//! Repo crunching (the PoC default tier).
//!
//! Walks a repo and lifts `Repo`/`File`/`Symbol`/`Framework` nodes + `Contains`/`Defines`/
//! `DependsOn` edges. The PoC symbol tier is a **high-recall, approximate** pure-Rust
//! regex scanner (the the scanner chassis/55 chassis) — confidence-tagged, defs-only. Precise
//! defs+refs via tree-sitter is the `treesitter` feature (the only C dep, off by default);
//! dependency parsing from manifests is precise. See PLAN.md §4 for the precision split.

use std::path::Path;

use regex::Regex;
use serde::Serialize;
use std::sync::OnceLock;

/// One extracted symbol definition.
#[derive(Debug, Clone, Serialize)]
pub struct Symbol {
    pub name: String,
    pub kind: &'static str,
}

/// The result of crunching one repo.
#[derive(Debug, Default)]
pub struct RepoScan {
    pub repo_label: String,
    /// repo-relative path → its symbols
    pub files: Vec<(String, Vec<Symbol>)>,
    /// **Every** repo-relative file path the walk visited (skip-dirs already excluded), not
    /// just symbol-bearing ones — so callers needing the full on-disk file set (e.g. to attach
    /// git `ChangedBy` edges to `.md`/`.toml` files) reuse this one filtered walk instead of
    /// re-walking the tree (which would re-descend `target/`/`.git/`/`node_modules/`).
    pub all_files: Vec<String>,
    /// dependency/framework names from manifests (precise)
    pub frameworks: Vec<String>,
}

struct LangRules {
    exts: &'static [&'static str],
    rules: Vec<(&'static str, Regex)>,
}

fn lang_rules() -> &'static [LangRules] {
    static R: OnceLock<Vec<LangRules>> = OnceLock::new();
    R.get_or_init(|| {
        let re = |p: &str| Regex::new(p).expect("static symbol regex");
        vec![
            LangRules {
                exts: &["rs"],
                rules: vec![
                    ("fn", re(r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")),
                    ("struct", re(r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)")),
                    ("enum", re(r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)")),
                    ("trait", re(r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)")),
                ],
            },
            LangRules {
                exts: &["py"],
                rules: vec![
                    ("def", re(r"(?m)^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)")),
                    ("class", re(r"(?m)^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")),
                ],
            },
            LangRules {
                exts: &["js", "ts", "jsx", "tsx", "mjs"],
                rules: vec![
                    ("function", re(r"(?m)\bfunction\s+([A-Za-z_$][A-Za-z0-9_$]*)")),
                    ("class", re(r"(?m)\bclass\s+([A-Za-z_$][A-Za-z0-9_$]*)")),
                    ("const", re(r"(?m)^\s*(?:export\s+)?const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=")),
                ],
            },
            LangRules {
                exts: &["go"],
                rules: vec![
                    ("func", re(r"(?m)^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)")),
                    ("type", re(r"(?m)^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)")),
                ],
            },
        ]
    })
    .as_slice()
}

fn is_skippable(name: &str) -> bool {
    matches!(name, ".git" | "target" | "node_modules" | "dist" | "build" | ".venv" | "vendor")
}

/// Crunch a repo directory into a [`RepoScan`]. Caps per-file symbols so a generated file
/// can't dominate the graph.
pub fn scan(root: &Path) -> anyhow::Result<RepoScan> {
    let repo_label = root
        .canonicalize()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "repo".to_string());

    let mut out = RepoScan {
        repo_label,
        ..Default::default()
    };

    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !e.file_name().to_str().is_some_and(is_skippable))
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        // Every visited file (skip-dirs already excluded by filter_entry above) — the full set
        // callers use for git ChangedBy linking, captured in this single walk.
        out.all_files.push(rel.clone());

        // Manifests → precise dependency/framework names.
        if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
            if matches!(fname, "Cargo.toml" | "package.json" | "go.mod" | "pyproject.toml") {
                if let Ok(text) = std::fs::read_to_string(path) {
                    out.frameworks.extend(parse_manifest(fname, &text));
                }
            }
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let Some(lr) = lang_rules().iter().find(|l| l.exts.contains(&ext)) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        if text.len() > 1_000_000 {
            continue; // skip very large/generated files
        }

        let mut syms = Vec::new();
        'rules: for (kind, re) in &lr.rules {
            for caps in re.captures_iter(&text) {
                if let Some(m) = caps.get(1) {
                    syms.push(Symbol { name: m.as_str().to_string(), kind });
                    if syms.len() >= 200 {
                        break 'rules; // cap across ALL rules, not just the current one
                    }
                }
            }
        }
        if !syms.is_empty() {
            out.files.push((rel, syms));
        }
    }

    out.frameworks.sort();
    out.frameworks.dedup();
    Ok(out)
}

/// Pull dependency names from a manifest (best-effort, line-oriented — no full TOML/JSON
/// parse). Section-aware so manifest metadata (`name`, `version`, `scripts`, …) is NOT
/// misread as a dependency, which would otherwise persist false `DependsOn` edges.
fn parse_manifest(fname: &str, text: &str) -> Vec<String> {
    let mut deps = Vec::new();
    match fname {
        "Cargo.toml" => {
            let mut in_deps = false;
            for line in text.lines() {
                let l = line.trim();
                if l.is_empty() || l.starts_with('#') {
                    continue;
                }
                if let Some(h) = l.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                    let h = h.trim();
                    // Table form: [dependencies.serde], [target.'cfg(..)'.dependencies.foo]
                    if let Some(idx) = h.rfind("dependencies.") {
                        let name = h[idx + "dependencies.".len()..].trim().trim_matches('"');
                        if !name.is_empty() {
                            deps.push(name.to_string());
                        }
                        in_deps = false;
                    } else {
                        in_deps = h == "dependencies"
                            || h.ends_with("-dependencies")      // dev-/build-dependencies
                            || h.ends_with(".dependencies"); // target.'cfg(..)'.dependencies
                    }
                    continue;
                }
                if in_deps {
                    if let Some(name) = l.split(['=', ' ']).next() {
                        let name = name.trim();
                        if !name.is_empty() {
                            deps.push(name.to_string());
                        }
                    }
                }
            }
        }
        "go.mod" => {
            let mut in_block = false;
            for line in text.lines() {
                let l = line.trim();
                if l.is_empty() || l.starts_with("//") {
                    continue;
                }
                if in_block {
                    if l.starts_with(')') {
                        in_block = false;
                    } else if let Some(name) = l.split_whitespace().next() {
                        deps.push(name.to_string());
                    }
                    continue;
                }
                if l.starts_with("require (") {
                    in_block = true;
                } else if let Some(rest) = l.strip_prefix("require ") {
                    if let Some(name) = rest.split_whitespace().next() {
                        deps.push(name.to_string());
                    }
                }
            }
        }
        "package.json" => {
            // For each dependency object, find its `{ ... }` (inline or multiline) by
            // matching braces, then take the KEY (first quoted string) of each entry. The
            // leading quote in the search key anchors it, so `"dependencies"` never matches
            // inside `"devDependencies"`.
            for key in [
                "\"dependencies\"",
                "\"devDependencies\"",
                "\"peerDependencies\"",
                "\"optionalDependencies\"",
            ] {
                let bytes = text.as_bytes();
                let mut from = 0;
                while let Some(kpos) = text[from..].find(key) {
                    let abs = from + kpos;
                    let Some(brace_rel) = text[abs..].find('{') else { break };
                    let obj_start = abs + brace_rel + 1;
                    let mut depth = 1i32;
                    let mut i = obj_start;
                    let mut end = text.len();
                    while i < bytes.len() {
                        match bytes[i] {
                            b'{' => depth += 1,
                            b'}' => {
                                depth -= 1;
                                if depth == 0 {
                                    end = i;
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    for entry in text[obj_start..end].split(',') {
                        if let Some(q1) = entry.find('"') {
                            if let Some(q2) = entry[q1 + 1..].find('"') {
                                let name = &entry[q1 + 1..q1 + 1 + q2];
                                if !name.is_empty() {
                                    deps.push(name.to_string());
                                }
                            }
                        }
                    }
                    from = end.max(obj_start);
                }
            }
        }
        "pyproject.toml" => {
            let mut in_poetry_deps = false; // [tool.poetry(.*)dependencies] table → key = dep
            let mut in_pep621_array = false; // [project] dependencies = [ "pkg>=1", ... ]
            for line in text.lines() {
                let l = line.trim();
                if l.is_empty() || l.starts_with('#') {
                    continue;
                }
                if let Some(h) = l.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                    let h = h.trim();
                    in_poetry_deps = h.starts_with("tool.poetry") && h.ends_with("dependencies");
                    in_pep621_array = false;
                    continue;
                }
                if in_poetry_deps {
                    if let Some(name) = l.split(['=', ' ']).next() {
                        let name = name.trim().trim_matches('"');
                        if !name.is_empty() && name != "python" {
                            deps.push(name.to_string());
                        }
                    }
                    continue;
                }
                if !in_pep621_array && l.starts_with("dependencies") && l.contains('=') && l.contains('[')
                {
                    in_pep621_array = !l.contains(']');
                    extract_pep621(l, &mut deps);
                    continue;
                }
                if in_pep621_array {
                    if l.contains(']') {
                        in_pep621_array = false;
                    }
                    extract_pep621(l, &mut deps);
                }
            }
        }
        _ => {}
    }
    deps.retain(|d| !d.is_empty() && d.len() < 64);
    deps.into_iter().take(100).collect()
}

#[cfg(test)]
mod tests {
    use super::parse_manifest;

    #[test]
    fn cargo_reads_deps_not_metadata() {
        let toml = r#"
[package]
name = "myapp"
version = "0.1.0"

[dependencies]
serde = "1"
tokio = { version = "1", features = ["full"] }

[dependencies.redb]
version = "4.1"

[dev-dependencies]
tempfile = "3"
"#;
        let deps = parse_manifest("Cargo.toml", toml);
        assert!(deps.contains(&"serde".to_string()));
        assert!(deps.contains(&"tokio".to_string()));
        assert!(deps.contains(&"redb".to_string()));
        assert!(deps.contains(&"tempfile".to_string()));
        assert!(!deps.contains(&"name".to_string()), "package metadata is not a dep");
        assert!(!deps.contains(&"version".to_string()), "the table field is not a dep");
        assert!(!deps.contains(&"myapp".to_string()));
    }

    #[test]
    fn go_mod_require_block_and_single() {
        let gomod = "module x\n\nrequire (\n\tgithub.com/foo/bar v1.2.3\n)\n\nrequire golang.org/x/sync v0.1.0\n";
        let deps = parse_manifest("go.mod", gomod);
        assert!(deps.contains(&"github.com/foo/bar".to_string()));
        assert!(deps.contains(&"golang.org/x/sync".to_string()));
        assert!(!deps.contains(&"module".to_string()));
    }

    #[test]
    fn package_json_only_dependency_objects() {
        let pkg = r#"{
  "name": "app",
  "scripts": { "build": "tsc" },
  "dependencies": { "react": "^18.0.0" },
  "devDependencies": { "vitest": "^1.0.0" }
}"#;
        let deps = parse_manifest("package.json", pkg);
        assert!(deps.contains(&"react".to_string()));
        assert!(deps.contains(&"vitest".to_string()));
        assert!(!deps.contains(&"build".to_string()), "scripts entries are not deps");
        assert!(!deps.contains(&"name".to_string()));
    }
}

/// Extract PEP 621 dependency names from a line of `dependencies = [ "pkg>=1", ... ]`,
/// stripping version specifiers / markers (the name is the leading ident run).
fn extract_pep621(line: &str, deps: &mut Vec<String>) {
    let mut rest = line;
    while let Some(q1) = rest.find('"') {
        let after = &rest[q1 + 1..];
        let Some(q2) = after.find('"') else { break };
        let spec = &after[..q2];
        let name: String = spec
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '_' || *c == '.')
            .collect();
        if !name.is_empty() {
            deps.push(name);
        }
        rest = &after[q2 + 1..];
    }
}
