//! Tiny fixture repo for loregraph's end-to-end index test (not a real crate — no manifest,
//! so the monorepo workspace never tries to build it; the scanner just lifts symbols).

pub struct DurableStore {
    pub path: String,
}

pub trait GraphStore {
    fn upsert(&mut self);
}

pub enum NodeKind {
    Decision,
    File,
}

pub fn open_store(path: &str) -> DurableStore {
    DurableStore { path: path.to_string() }
}
