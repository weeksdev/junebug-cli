//! Read-only full-screen terminal browsers for workspace files and Git changes.

use std::collections::HashSet;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::as_24_bit_terminal_escaped;

use crate::checkpoint::Checkpointer;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const INVERSE: &str = "\x1b[7m";
const MAX_FILES: usize = 10_000;
const MAX_DEPTH: usize = 16;
const MAX_VIEW_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone)]
struct Node {
    path: PathBuf,
    display: String,
    depth: usize,
    directory: bool,
}

#[derive(Debug, Clone)]
struct Change {
    path: PathBuf,
    status: String,
}

/// Open a read-only tree/file browser for `root`.
///
/// # Errors
///
/// Returns an error when the workspace cannot be scanned or terminal input
/// cannot be initialized.
pub fn explorer(root: &Path) -> Result<(), String> {
    let nodes = scan_tree(root)?;
    run_explorer(root, &nodes)
}

/// Open a read-only changed-file tree with a per-file Git diff pane.
///
/// # Errors
///
/// Returns an error when `root` is not a Git work tree, Git fails, or terminal
/// input cannot be initialized.
pub fn changes(root: &Path, checkpointer: Option<&Checkpointer>) -> Result<(), String> {
    match git_changes(root) {
        Ok(changes) => run_changes(root, &changes, ChangeBaseline::WorkspaceGit),
        Err(error) if error.contains("not a Git repository") => {
            let checkpointer = checkpointer.ok_or_else(|| {
                "this workspace is not a Git repository and checkpoints are disabled; /changes has no baseline".to_owned()
            })?;
            let changes = parse_status(&checkpointer.changes_status()?);
            run_changes(root, &changes, ChangeBaseline::Checkpoint(checkpointer))
        }
        Err(error) => Err(error),
    }
}

#[derive(Clone, Copy)]
enum ChangeBaseline<'a> {
    WorkspaceGit,
    Checkpoint(&'a Checkpointer),
}

fn scan_tree(root: &Path) -> Result<Vec<Node>, String> {
    let mut nodes = Vec::new();
    scan_directory(root, root, 0, &mut nodes)?;
    Ok(nodes)
}

fn scan_directory(
    root: &Path,
    directory: &Path,
    depth: usize,
    nodes: &mut Vec<Node>,
) -> Result<(), String> {
    if depth >= MAX_DEPTH || nodes.len() >= MAX_FILES {
        return Ok(());
    }
    let mut entries = fs::read_dir(directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| {
        let directory = entry.file_type().is_ok_and(|kind| kind.is_dir());
        (
            !directory,
            entry.file_name().to_string_lossy().to_ascii_lowercase(),
        )
    });
    for entry in entries {
        if nodes.len() >= MAX_FILES {
            break;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if should_hide(&name, kind.is_dir()) || kind.is_symlink() {
            continue;
        }
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
        nodes.push(Node {
            display: name,
            path: relative,
            depth,
            directory: kind.is_dir(),
        });
        if kind.is_dir() {
            scan_directory(root, &path, depth + 1, nodes)?;
        }
    }
    Ok(())
}

fn should_hide(name: &str, directory: bool) -> bool {
    matches!(
        name,
        ".git" | ".junebug" | ".febo" | ".junie" | ".claude" | ".env"
    ) || name.starts_with(".env.")
        || (directory && matches!(name, "target" | "node_modules" | "__pycache__" | ".venv"))
}

fn git_changes(root: &Path) -> Result<Vec<Change>, String> {
    let output = git(
        root,
        &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )?;
    Ok(parse_status(&output))
}

fn parse_status(output: &str) -> Vec<Change> {
    let mut fields = output.split('\0').filter(|field| !field.is_empty());
    let mut changes = Vec::new();
    while let Some(field) = fields.next() {
        if field.len() < 4 {
            continue;
        }
        let status = field[..2].to_owned();
        let path = PathBuf::from(&field[3..]);
        if status.contains('R') || status.contains('C') {
            let _ = fields.next();
        }
        changes.push(Change { path, status });
    }
    changes.sort_by(|left, right| left.path.cmp(&right.path));
    changes
}

fn git(root: &Path, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("--no-pager")
        .args(arguments)
        .current_dir(root)
        .env_clear()
        .env(
            "PATH",
            "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        )
        .env("LC_ALL", "C")
        .output()
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        let error = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        if error.contains("not a git repository") {
            Err("this workspace is not a Git repository; /changes needs a Git baseline".to_owned())
        } else {
            Err(error)
        }
    }
}

fn diff_for(root: &Path, change: &Change, baseline: ChangeBaseline<'_>) -> Vec<String> {
    if change.status == "??" {
        return added_file_lines(&root.join(&change.path), &change.path);
    }
    let path = change.path.to_string_lossy();
    let diff = match baseline {
        ChangeBaseline::WorkspaceGit => {
            let with_head = git(root, &["diff", "--no-ext-diff", "HEAD", "--", &path]);
            with_head.or_else(|_| git(root, &["diff", "--no-ext-diff", "--", &path]))
        }
        ChangeBaseline::Checkpoint(checkpointer) => checkpointer.diff_from_head(&change.path),
    };
    match diff {
        Ok(diff) if diff.is_empty() => vec!["(no textual diff)".to_owned()],
        Ok(diff) => diff.lines().map(sanitize).collect(),
        Err(error) => vec![format!("Git diff unavailable: {error}")],
    }
}

fn added_file_lines(path: &Path, relative: &Path) -> Vec<String> {
    let Ok(metadata) = fs::metadata(path) else {
        return vec!["(file is unavailable)".to_owned()];
    };
    if metadata.len() > MAX_VIEW_BYTES {
        return vec![format!("(untracked file exceeds {MAX_VIEW_BYTES} bytes)")];
    }
    match fs::read_to_string(path) {
        Ok(contents) => std::iter::once("--- /dev/null".to_owned())
            .chain(std::iter::once(format!("+++ b/{}", relative.display())))
            .chain(contents.lines().map(|line| format!("+{}", sanitize(line))))
            .collect(),
        Err(_) => vec!["(binary or non-UTF-8 untracked file)".to_owned()],
    }
}

fn file_lines(root: &Path, path: &Path) -> Vec<String> {
    let full = root.join(path);
    let Ok(metadata) = fs::metadata(&full) else {
        return vec!["(file is unavailable)".to_owned()];
    };
    if metadata.len() > MAX_VIEW_BYTES {
        return vec![format!(
            "(file exceeds the {MAX_VIEW_BYTES} byte viewer limit)"
        )];
    }
    match fs::read_to_string(full) {
        Ok(contents) => highlighted_file_lines(path, &contents),
        Err(_) => vec!["(binary or non-UTF-8 file)".to_owned()],
    }
}

fn highlighted_file_lines(path: &Path, contents: &str) -> Vec<String> {
    static SYNTAXES: OnceLock<SyntaxSet> = OnceLock::new();
    static THEME: OnceLock<Theme> = OnceLock::new();
    let syntaxes = SYNTAXES.get_or_init(SyntaxSet::load_defaults_nonewlines);
    let theme = THEME.get_or_init(|| {
        ThemeSet::load_defaults()
            .themes
            .remove("base16-ocean.dark")
            .unwrap_or_default()
    });
    let syntax = syntaxes
        .find_syntax_for_file(path)
        .ok()
        .flatten()
        .unwrap_or_else(|| syntaxes.find_syntax_plain_text());
    let mut highlighter = HighlightLines::new(syntax, theme);
    contents
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let line = sanitize(line);
            let rendered = highlighter.highlight_line(&line, syntaxes).map_or_else(
                |_| line.clone(),
                |regions| as_24_bit_terminal_escaped(&regions, false),
            );
            format!("{DIM}{:>5}{RESET}  {rendered}{RESET}", index + 1)
        })
        .collect()
}

#[allow(clippy::too_many_lines)]
fn run_explorer(root: &Path, nodes: &[Node]) -> Result<(), String> {
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return Err("/explorer requires an interactive terminal".to_owned());
    }
    terminal::enable_raw_mode().map_err(|error| error.to_string())?;
    enter_screen();
    let mut collapsed = HashSet::<PathBuf>::new();
    let mut selected = 0usize;
    let mut scroll = 0usize;
    let mut query = String::new();
    let mut searching = false;
    let mut detail_focused = false;
    let mut cached_path = None::<PathBuf>;
    let mut cached_lines = Vec::<String>::new();
    let result = loop {
        let visible = visible_nodes(nodes, &collapsed, &query);
        selected = selected.min(visible.len().saturating_sub(1));
        let selected_node = visible.get(selected).copied();
        if let Some(node) = selected_node.filter(|node| !node.directory) {
            if cached_path.as_ref() != Some(&node.path) {
                cached_lines = file_lines(root, &node.path);
                cached_path = Some(node.path.clone());
            }
        } else {
            cached_path = None;
            cached_lines.clear();
        }
        let lines = &cached_lines;
        let detail_title = selected_node
            .map(|node| node.path.to_string_lossy().into_owned())
            .unwrap_or_default();
        draw_screen(
            "Explorer",
            root,
            &visible
                .iter()
                .map(|node| explorer_label(node, collapsed.contains(&node.path)))
                .collect::<Vec<_>>(),
            selected,
            &detail_title,
            lines,
            scroll,
            &query,
            searching,
            false,
            detail_focused,
        );
        let event = match event::read() {
            Ok(event) => event,
            Err(error) => break Err(error.to_string()),
        };
        let Event::Key(key) = event else { continue };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        if searching {
            match key.code {
                KeyCode::Enter => searching = false,
                KeyCode::Esc => {
                    searching = false;
                    query.clear();
                }
                KeyCode::Backspace => {
                    query.pop();
                    selected = 0;
                    scroll = 0;
                }
                KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    query.push(character);
                    selected = 0;
                    scroll = 0;
                }
                _ => {}
            }
            continue;
        }
        if detail_focused {
            match key.code {
                KeyCode::Char('q') => break Ok(()),
                KeyCode::Left | KeyCode::Esc => detail_focused = false,
                KeyCode::Up | KeyCode::Char('k') => scroll = scroll.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => {
                    scroll = scroll.saturating_add(1);
                }
                KeyCode::PageDown => scroll = scroll.saturating_add(view_height()),
                KeyCode::PageUp => scroll = scroll.saturating_sub(view_height()),
                KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    scroll = scroll.saturating_add(view_height());
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    scroll = scroll.saturating_sub(view_height());
                }
                KeyCode::Home => scroll = 0,
                KeyCode::End => scroll = lines.len().saturating_sub(view_height()),
                _ => {}
            }
            continue;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
            KeyCode::Char('/') => searching = true,
            KeyCode::Up | KeyCode::Char('k') => {
                selected = selected.saturating_sub(1);
                scroll = 0;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(visible.len().saturating_sub(1));
                scroll = 0;
            }
            KeyCode::Enter => {
                if let Some(node) = selected_node {
                    if node.directory {
                        if !collapsed.remove(&node.path) {
                            collapsed.insert(node.path.clone());
                        }
                    } else {
                        detail_focused = true;
                    }
                }
            }
            KeyCode::Right => {
                if let Some(node) = selected_node {
                    if node.directory {
                        collapsed.remove(&node.path);
                    } else {
                        detail_focused = true;
                    }
                }
            }
            KeyCode::Left => {
                if let Some(node) = selected_node.filter(|node| node.directory) {
                    collapsed.insert(node.path.clone());
                }
            }
            KeyCode::PageDown => {
                scroll = scroll.saturating_add(view_height());
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                scroll = scroll.saturating_add(view_height());
            }
            KeyCode::PageUp => {
                scroll = scroll.saturating_sub(view_height());
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                scroll = scroll.saturating_sub(view_height());
            }
            KeyCode::Home => scroll = 0,
            KeyCode::End => scroll = lines.len().saturating_sub(view_height()),
            _ => {}
        }
    };
    leave_screen();
    let _ = terminal::disable_raw_mode();
    result
}

#[allow(clippy::too_many_lines)]
fn run_changes(
    root: &Path,
    changes: &[Change],
    baseline: ChangeBaseline<'_>,
) -> Result<(), String> {
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return Err("/changes requires an interactive terminal".to_owned());
    }
    terminal::enable_raw_mode().map_err(|error| error.to_string())?;
    enter_screen();
    let mut selected = 0usize;
    let mut scroll = 0usize;
    let mut query = String::new();
    let mut searching = false;
    let mut detail_focused = false;
    let result = loop {
        let query_lower = query.to_ascii_lowercase();
        let visible = changes
            .iter()
            .filter(|change| {
                query.is_empty()
                    || change
                        .path
                        .to_string_lossy()
                        .to_ascii_lowercase()
                        .contains(&query_lower)
            })
            .collect::<Vec<_>>();
        selected = selected.min(visible.len().saturating_sub(1));
        let selected_change = visible.get(selected).copied();
        let lines =
            selected_change.map_or_else(Vec::new, |change| diff_for(root, change, baseline));
        let labels = visible
            .iter()
            .map(|change| change_label(change))
            .collect::<Vec<_>>();
        let detail_title = selected_change
            .map(|change| change.path.to_string_lossy().into_owned())
            .unwrap_or_default();
        draw_screen(
            "Changes",
            root,
            &labels,
            selected,
            &detail_title,
            &lines,
            scroll,
            &query,
            searching,
            true,
            detail_focused,
        );
        let event = match event::read() {
            Ok(event) => event,
            Err(error) => break Err(error.to_string()),
        };
        let Event::Key(key) = event else { continue };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        if searching {
            match key.code {
                KeyCode::Enter => searching = false,
                KeyCode::Esc => {
                    searching = false;
                    query.clear();
                }
                KeyCode::Backspace => {
                    query.pop();
                    selected = 0;
                    scroll = 0;
                }
                KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    query.push(character);
                    selected = 0;
                    scroll = 0;
                }
                _ => {}
            }
            continue;
        }
        if detail_focused {
            match key.code {
                KeyCode::Char('q') => break Ok(()),
                KeyCode::Left | KeyCode::Esc => detail_focused = false,
                KeyCode::Up | KeyCode::Char('k') => scroll = scroll.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => {
                    scroll = scroll.saturating_add(1);
                }
                KeyCode::PageDown => scroll = scroll.saturating_add(view_height()),
                KeyCode::PageUp => scroll = scroll.saturating_sub(view_height()),
                KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    scroll = scroll.saturating_add(view_height());
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    scroll = scroll.saturating_sub(view_height());
                }
                KeyCode::Home => scroll = 0,
                KeyCode::End => scroll = lines.len().saturating_sub(view_height()),
                _ => {}
            }
            continue;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
            KeyCode::Char('/') => searching = true,
            KeyCode::Enter | KeyCode::Right if selected_change.is_some() => {
                detail_focused = true;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                selected = selected.saturating_sub(1);
                scroll = 0;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(visible.len().saturating_sub(1));
                scroll = 0;
            }
            KeyCode::PageDown => {
                scroll = scroll.saturating_add(view_height());
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                scroll = scroll.saturating_add(view_height());
            }
            KeyCode::PageUp => {
                scroll = scroll.saturating_sub(view_height());
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                scroll = scroll.saturating_sub(view_height());
            }
            KeyCode::Home => scroll = 0,
            KeyCode::End => scroll = lines.len().saturating_sub(view_height()),
            _ => {}
        }
    };
    leave_screen();
    let _ = terminal::disable_raw_mode();
    result
}

fn visible_nodes<'a>(
    nodes: &'a [Node],
    collapsed: &HashSet<PathBuf>,
    query: &str,
) -> Vec<&'a Node> {
    let query = query.to_ascii_lowercase();
    nodes
        .iter()
        .filter(|node| {
            if !query.is_empty() {
                return node
                    .path
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .contains(&query);
            }
            !collapsed
                .iter()
                .any(|parent| node.path.starts_with(parent) && node.path != *parent)
        })
        .collect()
}

fn explorer_label(node: &Node, collapsed: bool) -> String {
    let indent = "  ".repeat(node.depth);
    let marker = if node.directory {
        if collapsed { "▸" } else { "▾" }
    } else {
        "─"
    };
    let suffix = if node.directory { "/" } else { "" };
    format!("{indent}{marker} {}{suffix}", node.display)
}

fn change_label(change: &Change) -> String {
    let depth = change.path.components().count().saturating_sub(1);
    format!(
        "{}{} {}",
        "  ".repeat(depth),
        change.status,
        change
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    )
}

#[allow(clippy::too_many_arguments)]
fn draw_screen(
    title: &str,
    root: &Path,
    labels: &[String],
    selected: usize,
    detail_title: &str,
    detail: &[String],
    detail_scroll: usize,
    query: &str,
    searching: bool,
    diff: bool,
    detail_focused: bool,
) {
    let (columns, rows) = terminal::size().unwrap_or((100, 30));
    let columns = usize::from(columns).max(40);
    let rows = usize::from(rows).max(8);
    let left_width = (columns / 3).clamp(22, 44);
    let right_width = columns.saturating_sub(left_width + 3);
    let body_rows = rows.saturating_sub(4);
    let list_start = selected.saturating_sub(body_rows.saturating_sub(1));
    let max_scroll = detail.len().saturating_sub(body_rows);
    let detail_scroll = detail_scroll.min(max_scroll);
    let search = if searching {
        format!("{YELLOW}/{query}▌{RESET}")
    } else if query.is_empty() {
        String::new()
    } else {
        format!("{DIM}filter: {query}{RESET}")
    };
    let mut output = io::stderr().lock();
    let _ = write!(output, "\x1b[H\x1b[2J");
    let _ = write!(
        output,
        "{BOLD}{CYAN}Junebug {title}{RESET} {DIM}· {}{RESET}  {search}\r\n",
        root.display()
    );
    let left_header = format!(
        "{} ({})",
        if diff { "changed files" } else { "workspace" },
        labels.len()
    );
    let detail_header = fit(detail_title, right_width);
    let detail_header = if detail_focused && !detail_title.is_empty() {
        format!("{INVERSE}{detail_header}{RESET}")
    } else {
        detail_header
    };
    let _ = write!(
        output,
        "{} │ {}\r\n",
        fit(&left_header, left_width),
        detail_header
    );
    for row in 0..body_rows {
        let list_index = list_start + row;
        let left_plain = labels.get(list_index).map_or("", String::as_str);
        let left_fit = fit(left_plain, left_width);
        let left = if list_index == selected && !labels.is_empty() && !detail_focused {
            format!("{INVERSE}{left_fit}{RESET}")
        } else {
            left_fit
        };
        let right_plain = detail.get(detail_scroll + row).map_or("", String::as_str);
        let right = if diff {
            let right_fit = fit(right_plain, right_width);
            color_diff(right_plain, &right_fit)
        } else {
            fit_ansi(right_plain, right_width)
        };
        let _ = write!(output, "{left} │ {right}\r\n");
    }
    let footer = if detail_focused {
        "FILE/DIFF FOCUS · ↑↓ scroll · pgup/pgdn · ← back to tree · q close"
    } else if columns >= 100 {
        if diff {
            "↑↓ files · → focus diff · / search · read-only · q/esc close"
        } else {
            "↑↓ files · → focus file · / search · ←/→ tree · read-only · q/esc close"
        }
    } else if diff {
        "↑↓ files · pgup/pgdn diff · / search · q close"
    } else {
        "↑↓ files · pgup/pgdn view · / search · ←/→ tree · q close"
    };
    let _ = write!(output, "{DIM}{}{RESET}", fit(footer, columns));
    let _ = output.flush();
}

fn color_diff(original: &str, fitted: &str) -> String {
    if original.starts_with('+') && !original.starts_with("+++") {
        format!("{GREEN}{fitted}{RESET}")
    } else if original.starts_with('-') && !original.starts_with("---") {
        format!("{RED}{fitted}{RESET}")
    } else if original.starts_with("@@") {
        format!("{CYAN}{fitted}{RESET}")
    } else if original.starts_with("diff ") || original.starts_with("index ") {
        format!("{DIM}{fitted}{RESET}")
    } else {
        fitted.to_owned()
    }
}

fn fit(value: &str, width: usize) -> String {
    let sanitized = sanitize(value);
    let mut chars = sanitized.chars();
    let mut output = chars.by_ref().take(width).collect::<String>();
    if chars.next().is_some() && width > 0 {
        output.pop();
        output.push('…');
    }
    let length = output.chars().count();
    output.push_str(&" ".repeat(width.saturating_sub(length)));
    output
}

fn fit_ansi(value: &str, width: usize) -> String {
    let visible = visible_ansi_width(value);
    let truncated = visible > width;
    let target = if truncated {
        width.saturating_sub(1)
    } else {
        width
    };
    let mut output = String::new();
    let mut used = 0usize;
    let mut characters = value.chars();
    while let Some(character) = characters.next() {
        if character == '\x1b' {
            output.push(character);
            for escape in characters.by_ref() {
                output.push(escape);
                if escape == 'm' {
                    break;
                }
            }
        } else if used < target {
            output.push(character);
            used += 1;
        } else {
            break;
        }
    }
    if truncated && width > 0 {
        output.push('…');
        used += 1;
    }
    output.push_str(RESET);
    output.push_str(&" ".repeat(width.saturating_sub(used)));
    output
}

fn visible_ansi_width(value: &str) -> usize {
    let mut visible = 0usize;
    let mut escaped = false;
    for character in value.chars() {
        if escaped {
            if character == 'm' {
                escaped = false;
            }
        } else if character == '\x1b' {
            escaped = true;
        } else {
            visible += 1;
        }
    }
    visible
}

fn sanitize(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\t' => output.push_str("    "),
            character if character.is_control() => output.push('�'),
            character => output.push(character),
        }
    }
    output
}

fn view_height() -> usize {
    terminal::size()
        .map_or(20, |(_, rows)| usize::from(rows).saturating_sub(4))
        .max(1)
}

fn enter_screen() {
    eprint!("\x1b[?1049h\x1b[?25l");
    let _ = io::stderr().flush();
}

fn leave_screen() {
    eprint!("\x1b[?25h\x1b[?1049l");
    let _ = io::stderr().flush();
}

#[cfg(test)]
mod tests {
    use super::{
        Change, Node, change_label, explorer_label, fit, fit_ansi, parse_status, sanitize,
        visible_ansi_width, visible_nodes,
    };
    use std::collections::HashSet;
    use std::path::PathBuf;

    #[test]
    fn terminal_text_is_sanitized_and_clipped() {
        assert_eq!(sanitize("safe\x1b[31m"), "safe�[31m");
        assert_eq!(fit("abcdef", 4), "abc…");
        assert_eq!(fit("a", 3), "a  ");
        let colored = fit_ansi("\x1b[31mabcdef\x1b[0m", 4);
        assert_eq!(visible_ansi_width(&colored), 4);
        assert!(colored.contains("abc…"));
    }

    #[test]
    fn collapsed_directories_hide_descendants_and_search_finds_them() {
        let nodes = vec![
            Node {
                path: PathBuf::from("src"),
                display: "src".to_owned(),
                depth: 0,
                directory: true,
            },
            Node {
                path: PathBuf::from("src/main.rs"),
                display: "main.rs".to_owned(),
                depth: 1,
                directory: false,
            },
        ];
        let collapsed = HashSet::from([PathBuf::from("src")]);
        assert_eq!(visible_nodes(&nodes, &collapsed, "").len(), 1);
        assert_eq!(visible_nodes(&nodes, &collapsed, "main").len(), 1);
        assert!(explorer_label(&nodes[0], true).contains('▸'));
    }

    #[test]
    fn changed_paths_render_as_tree_indented_files() {
        let label = change_label(&Change {
            path: PathBuf::from("src/main.rs"),
            status: " M".to_owned(),
        });
        assert!(label.starts_with("   M"));
        assert!(label.ends_with("main.rs"));
    }

    #[test]
    fn porcelain_status_becomes_changed_file_rows() {
        let changes = parse_status(" M src/main.rs\0?? new.txt\0");
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].path, PathBuf::from("new.txt"));
        assert_eq!(changes[1].path, PathBuf::from("src/main.rs"));
    }
}
