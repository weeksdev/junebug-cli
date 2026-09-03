//! Local semantic (embedding-based) code search. `search` (ripgrep) finds
//! literal text; this finds text with related *meaning* by embedding
//! workspace chunks with a local Ollama embedding model and ranking by
//! cosine similarity to the query embedding.
//!
//! Everything runs against the local Ollama runtime already used for chat —
//! nothing leaves the machine unless `OLLAMA_HOST` points elsewhere, which
//! the user already opted into by setting it. The index is a per-workspace
//! JSONL cache at `.junebug/semindex.jsonl`: one header line naming the
//! embedding model, then one line per chunk (path, line range, content hash,
//! text, and its embedding vector). Rebuilding reuses embeddings for any
//! file whose content hash hasn't changed, so repeated `/index` runs after
//! small edits only re-embed what actually changed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::provider::{OpenAiCompatibleProvider, ProviderKind, ollama_base_url};

const INDEX_RELATIVE_PATH: &str = ".junebug/semindex.jsonl";
/// Matches the `@` file-search menu's caps (`editor.rs`) so behavior stays
/// predictable across features that walk the workspace tree.
const MAX_FILES: usize = 2000;
const DEPTH_LIMIT: usize = 8;
const CHUNK_LINES: usize = 60;
const CHUNK_OVERLAP: usize = 10;
const MAX_FILE_BYTES: u64 = 512 * 1024;
/// Upper bound on total indexed chunks, so a huge monorepo cannot turn a
/// rebuild into an unbounded embedding job.
const MAX_CHUNKS: usize = 20_000;
pub const DEFAULT_RESULTS: usize = 8;
const MAX_RESULTS: usize = 20;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_mins(2);

/// Installed-model name fragments that indicate an embedding model, ranked
/// by preference. Matched by substring, case-insensitively.
const EMBED_MODEL_HINTS: &[&str] = &[
    "nomic-embed-text",
    "mxbai-embed-large",
    "bge-m3",
    "snowflake-arctic-embed",
    "granite-embedding",
    "all-minilm",
    "embed",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum IndexLine {
    Header {
        model: String,
    },
    Chunk {
        path: String,
        start: usize,
        end: usize,
        file_hash: String,
        text: String,
        embedding: Vec<f32>,
    },
}

pub struct BuildStats {
    pub files_scanned: usize,
    pub files_embedded: usize,
    pub files_reused: usize,
    pub chunks_total: usize,
}

fn index_path(root: &Path) -> PathBuf {
    root.join(INDEX_RELATIVE_PATH)
}

fn embedding_client() -> Result<Client, String> {
    Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|error| error.to_string())
}

/// Pick an installed embedding-capable Ollama model.
///
/// # Errors
///
/// Returns an actionable error when Ollama is unreachable or no installed
/// model looks like an embedding model.
fn pick_embedding_model() -> Result<String, String> {
    if !ProviderKind::Ollama.has_credential() {
        return Err(
            "semantic search needs a reachable Ollama runtime; start Ollama or set OLLAMA_HOST"
                .to_owned(),
        );
    }
    let provider = OpenAiCompatibleProvider::from_environment(ProviderKind::Ollama, None)?;
    let models = provider.list_models()?;
    EMBED_MODEL_HINTS
        .iter()
        .find_map(|hint| {
            models
                .iter()
                .find(|model| model.to_lowercase().contains(hint))
                .cloned()
        })
        .ok_or_else(|| {
            "no embedding model installed — run `ollama pull nomic-embed-text` (or another \
             embedding model) to enable semantic_search"
                .to_owned()
        })
}

#[allow(clippy::cast_possible_truncation)]
fn embed_one(client: &Client, model: &str, text: &str) -> Result<Vec<f32>, String> {
    let response = client
        .post(format!("{}/api/embeddings", ollama_base_url()))
        .json(&json!({"model": model, "prompt": text}))
        .send()
        .map_err(|error| error.to_string())?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("Ollama embeddings request returned {status}"));
    }
    let body: Value = response.json().map_err(|error| error.to_string())?;
    body.get("embedding")
        .and_then(Value::as_array)
        .ok_or("embeddings response lacks an 'embedding' array")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| "non-numeric embedding value".to_owned())
        })
        .collect()
}

fn embed_texts(client: &Client, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
    texts
        .iter()
        .map(|text| embed_one(client, model, text))
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::MIN;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn fnv1a_hex(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Collect workspace file paths, skipping hidden entries and common
/// build/dependency directories — the same rule the `@` file-search menu
/// uses (`editor.rs::workspace_files`).
fn workspace_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((directory, depth)) = stack.pop() {
        if depth > DEPTH_LIMIT || files.len() >= MAX_FILES {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                stack.push((path, depth + 1));
            } else {
                files.push(path);
                if files.len() >= MAX_FILES {
                    break;
                }
            }
        }
    }
    files.sort_unstable();
    files
}

fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8000).any(|byte| *byte == 0)
}

/// Split `text` into overlapping line-range chunks, returned as
/// `(start_line, end_line, text)` with 1-based inclusive line numbers.
fn chunk_file(text: &str) -> Vec<(usize, usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let step = CHUNK_LINES - CHUNK_OVERLAP;
    loop {
        let end = (start + CHUNK_LINES).min(lines.len());
        chunks.push((start + 1, end, lines[start..end].join("\n")));
        if end == lines.len() {
            break;
        }
        start += step;
    }
    chunks
}

fn load_index(root: &Path) -> Result<(String, Vec<IndexLine>), String> {
    let contents = std::fs::read_to_string(index_path(root)).map_err(|error| error.to_string())?;
    let mut lines = contents.lines();
    let header: IndexLine = serde_json::from_str(lines.next().ok_or("empty semantic index")?)
        .map_err(|error| error.to_string())?;
    let IndexLine::Header { model } = header else {
        return Err("semantic index is corrupt (missing header)".to_owned());
    };
    let chunks = lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<IndexLine>(line).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((model, chunks))
}

fn write_index(root: &Path, lines: &[IndexLine]) -> Result<(), String> {
    let path = index_path(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut buffer = String::new();
    for line in lines {
        buffer.push_str(&serde_json::to_string(line).map_err(|error| error.to_string())?);
        buffer.push('\n');
    }
    std::fs::write(path, buffer).map_err(|error| error.to_string())
}

/// (Re)build the semantic index for `root`, reusing embeddings for any file
/// whose content hash matches the existing index so repeated builds after
/// small edits only re-embed what changed.
///
/// # Errors
///
/// Returns an error when no embedding model is available or an embedding
/// request fails; a partially-written index is never left behind (the file
/// is only replaced once the whole build succeeds).
pub fn build_index(root: &Path) -> Result<BuildStats, String> {
    let model = pick_embedding_model()?;
    let client = embedding_client()?;

    let mut existing_by_path: HashMap<String, Vec<IndexLine>> = HashMap::new();
    if let Ok((existing_model, chunks)) = load_index(root)
        && existing_model == model
    {
        for line in chunks {
            if let IndexLine::Chunk { ref path, .. } = line {
                existing_by_path.entry(path.clone()).or_default().push(line);
            }
        }
    }

    let files = workspace_files(root);
    let mut out_lines = vec![IndexLine::Header {
        model: model.clone(),
    }];
    let mut files_embedded = 0usize;
    let mut files_reused = 0usize;
    let mut chunks_total = 0usize;

    'files: for path in &files {
        if chunks_total >= MAX_CHUNKS {
            break;
        }
        let Ok(metadata) = std::fs::metadata(path) else {
            continue;
        };
        if !metadata.is_file() || metadata.len() > MAX_FILE_BYTES {
            continue;
        }
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        if looks_binary(&bytes) {
            continue;
        }
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        let relative = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let file_hash = fnv1a_hex(&text);

        if let Some(previous) = existing_by_path.get(&relative)
            && !previous.is_empty()
            && previous.iter().all(|line| {
                matches!(line, IndexLine::Chunk { file_hash: existing, .. } if *existing == file_hash)
            })
        {
            chunks_total += previous.len();
            out_lines.extend(previous.iter().cloned());
            files_reused += 1;
            continue;
        }

        let chunks = chunk_file(&text);
        if chunks.is_empty() {
            continue;
        }
        let texts: Vec<String> = chunks.iter().map(|(_, _, text)| text.clone()).collect();
        let embeddings = embed_texts(&client, &model, &texts)?;
        files_embedded += 1;
        for ((start, end, text), embedding) in chunks.into_iter().zip(embeddings) {
            out_lines.push(IndexLine::Chunk {
                path: relative.clone(),
                start,
                end,
                file_hash: file_hash.clone(),
                text,
                embedding,
            });
            chunks_total += 1;
            if chunks_total >= MAX_CHUNKS {
                continue 'files;
            }
        }
    }

    write_index(root, &out_lines)?;
    Ok(BuildStats {
        files_scanned: files.len(),
        files_embedded,
        files_reused,
        chunks_total,
    })
}

/// Semantic search over the workspace: embeds `query`, ranks indexed chunks
/// by cosine similarity, and returns the top matches with path, line range,
/// score, and the chunk text. Builds the index automatically on first use
/// (or when the configured embedding model has changed since the last
/// build); callers doing this from an interactive turn should expect that
/// first call to take longer on a large workspace.
///
/// # Errors
///
/// Returns an error when no embedding model is available, the workspace has
/// no indexable files, or an embedding request fails.
pub fn search(root: &Path, query: &str, limit: usize) -> Result<String, String> {
    use std::fmt::Write as _;
    if query.trim().is_empty() {
        return Err("query must not be empty".to_owned());
    }
    let limit = limit.clamp(1, MAX_RESULTS);
    let model = pick_embedding_model()?;
    let chunks = match load_index(root) {
        Ok((indexed_model, chunks)) if indexed_model == model && !chunks.is_empty() => chunks,
        _ => {
            build_index(root)?;
            load_index(root)?.1
        }
    };
    if chunks.is_empty() {
        return Ok("no indexable files found in the workspace".to_owned());
    }

    let client = embedding_client()?;
    let query_embedding = embed_one(&client, &model, query)?;

    let mut scored: Vec<(f32, &IndexLine)> = chunks
        .iter()
        .filter_map(|line| match line {
            IndexLine::Chunk { embedding, .. } => {
                Some((cosine_similarity(&query_embedding, embedding), line))
            }
            IndexLine::Header { .. } => None,
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);
    if scored.is_empty() {
        return Ok("no matches".to_owned());
    }

    let mut out = String::new();
    for (index, (score, line)) in scored.iter().enumerate() {
        let IndexLine::Chunk {
            path,
            start,
            end,
            text,
            ..
        } = line
        else {
            continue;
        };
        let _ = write!(
            out,
            "{}. {path}:{start}-{end}  (score {score:.2})\n{text}\n\n",
            index + 1
        );
    }
    Ok(out.trim_end().to_owned())
}

#[cfg(test)]
mod tests {
    use super::{chunk_file, cosine_similarity, fnv1a_hex, looks_binary};

    #[test]
    fn chunks_cover_every_line_with_overlap() {
        let text = (1..=130)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_file(&text);
        assert_eq!(chunks.first().unwrap().0, 1);
        assert_eq!(chunks.last().unwrap().1, 130);
        for window in chunks.windows(2) {
            assert!(window[1].0 <= window[0].1, "chunks must overlap or touch");
        }
    }

    #[test]
    fn short_file_is_a_single_chunk() {
        let chunks = chunk_file("a\nb\nc");
        assert_eq!(chunks, vec![(1, 3, "a\nb\nc".to_owned())]);
    }

    #[test]
    fn cosine_similarity_is_one_for_identical_vectors() {
        let vector = vec![0.5_f32, 0.5, 0.7];
        assert!((cosine_similarity(&vector, &vector) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_handles_zero_vectors_without_dividing_by_zero() {
        assert!(cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]).abs() < f32::EPSILON);
    }

    #[test]
    fn fnv_hash_changes_with_content() {
        assert_ne!(fnv1a_hex("a"), fnv1a_hex("b"));
        assert_eq!(fnv1a_hex("a"), fnv1a_hex("a"));
    }

    #[test]
    fn binary_content_is_detected_by_a_null_byte() {
        assert!(looks_binary(&[0x50, 0x4b, 0x00, 0x03]));
        assert!(!looks_binary(b"fn main() {}"));
    }
}
