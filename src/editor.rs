//! Raw-mode line editor for the REPL with completion menus: `/` completes
//! slash commands and `@` completes workspace file paths, rendered in a
//! selectable menu below the input line (Claude Code style).

use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal;

pub const SLASH_COMMANDS: [(&str, &str); 15] = [
    ("/changes", "browse changed files and per-file diffs"),
    ("/compact", "summarize the conversation to free context"),
    ("/diff", "show the uncommitted Git diff"),
    ("/exit", "quit (Ctrl-D also works)"),
    ("/explorer", "browse, search, and edit workspace files"),
    ("/help", "show help"),
    ("/keys", "set or replace a provider API key"),
    ("/model", "pick or switch the model"),
    ("/permissions", "change what Junebug may do without asking"),
    ("/quit", "quit"),
    (
        "/rewind",
        "restore workspace files to an earlier checkpoint",
    ),
    ("/status", "provider, model, permissions, session"),
    (
        "/swarm",
        "run a boss/worker/checker model swarm on a goal (resume continues an aborted one)",
    ),
    (
        "/swarm-status",
        "progress readout of the saved swarm (add ai for a model summary)",
    ),
    ("/swarm-setup", "assign models to swarm roles"),
];

const MENU_LIMIT: usize = 8;
const FILE_LIMIT: usize = 2000;
const DEPTH_LIMIT: usize = 8;
const PROMPT_COLUMNS: u16 = 2;

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const INVERSE: &str = "\x1b[7m";

#[derive(Debug, Clone, PartialEq, Eq)]
struct MenuItem {
    /// Text inserted into the buffer when accepted.
    insert: String,
    /// Text shown in the menu row.
    label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CompletionContext {
    /// Query after the leading `/` of the first token.
    Slash(String),
    /// `start` is the char index just after `@`; `query` runs to the cursor.
    File {
        start: usize,
        query: String,
    },
    None,
}

pub struct Editor {
    root: PathBuf,
    /// Workspace file list, gathered lazily once per process.
    files: Option<Vec<String>>,
    history: Vec<String>,
    /// Multi-line paste contents, referenced from the input line by
    /// `[pasted #N +L lines]` placeholders and expanded on submit. Kept for
    /// the whole session so history recall keeps working.
    pastes: Vec<String>,
    /// User-defined slash commands as (`/name`, description), appended to
    /// the builtin list in the `/` completion menu.
    custom_commands: Vec<(String, String)>,
}

impl Editor {
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            files: None,
            history: Vec::new(),
            pastes: Vec::new(),
            custom_commands: Vec::new(),
        }
    }

    pub fn set_custom_commands(&mut self, commands: Vec<(String, String)>) {
        self.custom_commands = commands;
    }

    /// The placeholder inserted into the input line for paste `index`.
    fn paste_placeholder(&self, index: usize) -> String {
        format!(
            "[pasted #{} +{} lines]",
            index + 1,
            self.pastes[index].lines().count().max(1)
        )
    }

    /// Replace every intact paste placeholder with its stored content.
    /// Placeholders the user edited no longer match and stay literal text.
    fn expand_pastes(&self, text: &str) -> String {
        let mut expanded = text.to_owned();
        for index in 0..self.pastes.len() {
            expanded = expanded.replace(&self.paste_placeholder(index), &self.pastes[index]);
        }
        expanded
    }

    /// Read one input line, drawing the prompt and a styled `footer` hint below
    /// it (when no completion menu is open), and the completion menu. Returns
    /// `None` on end of input (Ctrl-D on an empty line).
    pub fn read_line(&mut self, footer: &str) -> Option<String> {
        self.read_line_with_shortcut(footer, None)
    }

    /// Read a line while allowing Shift+Tab to update the footer in place.
    /// The callback is responsible for cycling application state and returns
    /// the newly rendered footer. Typed input and cursor position are kept.
    pub fn read_line_with_shortcut(
        &mut self,
        footer: &str,
        on_shift_tab: Option<&mut dyn FnMut() -> String>,
    ) -> Option<String> {
        if !io::stdin().is_terminal() || terminal::enable_raw_mode().is_err() {
            return fallback_read_line();
        }
        // With bracketed paste on, a paste arrives as one atomic event
        // instead of keystrokes — so newlines in it cannot submit the line.
        let _ = crossterm::execute!(io::stderr(), event::EnableBracketedPaste);
        let result = self.edit_loop(footer.to_owned(), on_shift_tab);
        let _ = crossterm::execute!(io::stderr(), event::DisableBracketedPaste);
        let _ = terminal::disable_raw_mode();
        if let Some(line) = &result
            && !line.is_empty()
        {
            // History keeps the compact placeholder form; expansion happens
            // on the returned line (and again on any history resubmit).
            self.history.push(line.clone());
        }
        result.map(|line| self.expand_pastes(&line))
    }

    #[allow(clippy::too_many_lines)]
    fn edit_loop(
        &mut self,
        mut footer: String,
        mut on_shift_tab: Option<&mut dyn FnMut() -> String>,
    ) -> Option<String> {
        let mut buffer: Vec<char> = Vec::new();
        let mut cursor = 0usize;
        let mut selected = 0usize;
        let mut history_index: Option<usize> = None;
        let mut draft: Vec<char> = Vec::new();
        // Which wrapped row of the input block the cursor rests on; redraws
        // must climb back to the prompt row before clearing.
        let mut cursor_row = 0usize;
        loop {
            let text: String = buffer.iter().collect();
            let context = completion_context(&buffer, cursor);
            let items = self.menu_items(&context);
            if selected >= items.len() {
                selected = 0;
            }
            draw(&text, cursor, &items, selected, &footer, &mut cursor_row);
            let key = match event::read() {
                Ok(Event::Key(key)) => key,
                Ok(Event::Resize(..)) => {
                    // A resize rewraps the scrollback unpredictably; forget
                    // the row tracking rather than climbing into unrelated
                    // lines, and let the next draw start from wherever the
                    // terminal left the cursor.
                    cursor_row = 0;
                    continue;
                }
                Ok(Event::Paste(pasted)) => {
                    let pasted = pasted.replace("\r\n", "\n").replace('\r', "\n");
                    let insert = if pasted.contains('\n') {
                        // Collapse a multi-line paste to a placeholder so the
                        // single-row editor stays editable; it expands back
                        // to the full text when the line is submitted.
                        self.pastes.push(pasted);
                        self.paste_placeholder(self.pastes.len() - 1)
                    } else {
                        pasted
                    };
                    for character in insert.chars() {
                        buffer.insert(cursor, character);
                        cursor += 1;
                    }
                    selected = 0;
                    continue;
                }
                _ => continue,
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            let control = key.modifiers.contains(KeyModifiers::CONTROL);
            match key.code {
                KeyCode::Char('c') if control => {
                    clear_menu_and_break_line(&text, cursor_row);
                    return Some(String::new());
                }
                KeyCode::Char('d') if control && buffer.is_empty() => {
                    clear_menu_and_break_line(&text, cursor_row);
                    return None;
                }
                KeyCode::Char('a') if control => cursor = 0,
                KeyCode::Char('e') if control => cursor = buffer.len(),
                KeyCode::Char('u') if control => {
                    buffer.drain(..cursor);
                    cursor = 0;
                }
                KeyCode::Char('k') if control => {
                    buffer.truncate(cursor);
                }
                KeyCode::Char(character) if !control => {
                    buffer.insert(cursor, character);
                    cursor += 1;
                    selected = 0;
                }
                KeyCode::Backspace if cursor > 0 => {
                    cursor -= 1;
                    buffer.remove(cursor);
                    selected = 0;
                }
                KeyCode::Delete if cursor < buffer.len() => {
                    buffer.remove(cursor);
                }
                KeyCode::Left => cursor = cursor.saturating_sub(1),
                KeyCode::Right => cursor = (cursor + 1).min(buffer.len()),
                KeyCode::Home => cursor = 0,
                KeyCode::End => cursor = buffer.len(),
                KeyCode::Up => {
                    if items.is_empty() {
                        let history_len = self.history.len();
                        if history_len > 0 {
                            let next = match history_index {
                                None => {
                                    draft.clone_from(&buffer);
                                    history_len - 1
                                }
                                Some(index) => index.saturating_sub(1),
                            };
                            history_index = Some(next);
                            buffer = self.history[next].chars().collect();
                            cursor = buffer.len();
                        }
                    } else {
                        selected = selected.checked_sub(1).unwrap_or(items.len() - 1);
                    }
                }
                KeyCode::Down => {
                    if items.is_empty() {
                        if let Some(index) = history_index {
                            if index + 1 < self.history.len() {
                                history_index = Some(index + 1);
                                buffer = self.history[index + 1].chars().collect();
                            } else {
                                history_index = None;
                                buffer.clone_from(&draft);
                            }
                            cursor = buffer.len();
                        }
                    } else {
                        selected = (selected + 1) % items.len();
                    }
                }
                KeyCode::BackTab => {
                    if let Some(callback) = on_shift_tab.as_deref_mut() {
                        footer = callback();
                    }
                }
                KeyCode::Tab if key.modifiers.contains(KeyModifiers::SHIFT) => {
                    if let Some(callback) = on_shift_tab.as_deref_mut() {
                        footer = callback();
                    }
                }
                KeyCode::Tab => {
                    if let Some(item) = items.get(selected) {
                        accept(&mut buffer, &mut cursor, &context, item);
                        selected = 0;
                    }
                }
                KeyCode::Esc => {
                    // Dismissing the menu: move the cursor out of the
                    // completion context by appending nothing; simplest
                    // predictable behavior is to leave text as-is and let
                    // the next keystroke re-open it. Jump to end of line.
                    cursor = buffer.len();
                }
                KeyCode::Enter => {
                    if let Some(item) = items.get(selected)
                        && would_change(&buffer, &context, item)
                    {
                        accept(&mut buffer, &mut cursor, &context, item);
                        selected = 0;
                        continue;
                    }
                    let final_text: String = buffer.iter().collect();
                    clear_menu_and_break_line(&final_text, cursor_row);
                    return Some(final_text.trim().to_owned());
                }
                _ => {}
            }
        }
    }

    fn menu_items(&mut self, context: &CompletionContext) -> Vec<MenuItem> {
        match context {
            CompletionContext::Slash(query) => SLASH_COMMANDS
                .iter()
                .filter(|(name, _)| name[1..].starts_with(query.as_str()))
                .map(|(name, description)| MenuItem {
                    insert: (*name).to_owned(),
                    label: format!("{name}  {DIM}{description}{RESET}"),
                })
                .chain(
                    self.custom_commands
                        .iter()
                        .filter(|(name, _)| name[1..].starts_with(query.as_str()))
                        .map(|(name, description)| MenuItem {
                            insert: name.clone(),
                            label: format!("{name}  {DIM}{description}{RESET}"),
                        }),
                )
                .take(MENU_LIMIT)
                .collect(),
            CompletionContext::File { query, .. } => {
                let files = self
                    .files
                    .get_or_insert_with(|| workspace_files(&self.root));
                filter_files(files, query)
                    .into_iter()
                    .map(|path| MenuItem {
                        insert: path.clone(),
                        label: path.clone(),
                    })
                    .collect()
            }
            CompletionContext::None => Vec::new(),
        }
    }
}

fn fallback_read_line() -> Option<String> {
    eprint!("\x1b[1;36m❯\x1b[0m ");
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line.trim().to_owned()),
    }
}

fn completion_context(buffer: &[char], cursor: usize) -> CompletionContext {
    if buffer.first() == Some(&'/') {
        let first_token_end = buffer
            .iter()
            .position(|character| character.is_whitespace())
            .unwrap_or(buffer.len());
        if cursor <= first_token_end {
            return CompletionContext::Slash(buffer[1..cursor].iter().collect());
        }
        return CompletionContext::None;
    }
    let mut index = cursor;
    while index > 0 {
        let character = buffer[index - 1];
        if character.is_whitespace() {
            break;
        }
        if character == '@' {
            return CompletionContext::File {
                start: index,
                query: buffer[index..cursor].iter().collect(),
            };
        }
        index -= 1;
    }
    CompletionContext::None
}

fn accept(
    buffer: &mut Vec<char>,
    cursor: &mut usize,
    context: &CompletionContext,
    item: &MenuItem,
) {
    match context {
        CompletionContext::Slash(_) => {
            let tail: Vec<char> = buffer[*cursor..].to_vec();
            *buffer = item.insert.chars().collect();
            *cursor = buffer.len();
            buffer.extend(tail);
        }
        CompletionContext::File { start, .. } => {
            let mut replacement: Vec<char> = item.insert.chars().collect();
            replacement.push(' ');
            let replacement_len = replacement.len();
            buffer.splice(*start..*cursor, replacement);
            *cursor = start + replacement_len;
        }
        CompletionContext::None => {}
    }
}

fn would_change(_buffer: &[char], context: &CompletionContext, item: &MenuItem) -> bool {
    match context {
        CompletionContext::Slash(query) => item.insert != format!("/{query}"),
        CompletionContext::File { query, .. } => item.insert != query.as_str(),
        CompletionContext::None => false,
    }
}

/// Where the input cursor sits inside the wrapped `❯ text` block. Rows are
/// 0-based from the prompt row; `cursor_column` is a 1-based terminal
/// column. Terminals wrap the line every `width` display columns, so redraw
/// and cursor moves must be row-aware: a single-row `\r` + clear-below left
/// one stale duplicate of the prompt line behind per keystroke once the
/// input wrapped past the terminal width.
struct InputLayout {
    cursor_row: usize,
    cursor_column: usize,
    last_row: usize,
}

fn input_layout(text: &str, cursor: usize, width: usize) -> InputLayout {
    let width = width.max(usize::from(PROMPT_COLUMNS) + 1);
    let mut columns_before_cursor = usize::from(PROMPT_COLUMNS);
    let mut total_columns = usize::from(PROMPT_COLUMNS);
    for (index, character) in text.chars().enumerate() {
        let columns = crate::browser::char_columns(character);
        if index < cursor {
            columns_before_cursor += columns;
        }
        total_columns += columns;
    }
    let last_row = (total_columns - 1) / width;
    let row = columns_before_cursor / width;
    if row > last_row {
        // The text exactly fills its final row and the cursor is at its
        // end: the terminal defers the wrap until the next character, so
        // the cursor physically rests on the last row's final column.
        InputLayout {
            cursor_row: last_row,
            cursor_column: width,
            last_row,
        }
    } else {
        InputLayout {
            cursor_row: row,
            cursor_column: columns_before_cursor % width + 1,
            last_row,
        }
    }
}

/// Truncate `line` to at most `width` display columns, passing ANSI escape
/// sequences through uncounted. Menu and footer rows must never wrap: the
/// redraw math counts each as exactly one terminal row.
fn clip_columns(line: &str, width: usize) -> String {
    let mut output = String::new();
    let mut used = 0usize;
    let mut characters = line.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '\x1b' {
            output.push(character);
            for follow in characters.by_ref() {
                output.push(follow);
                if follow.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        let columns = crate::browser::char_columns(character);
        if used + columns > width {
            break;
        }
        used += columns;
        output.push(character);
    }
    output
}

fn terminal_width() -> usize {
    terminal::size().map_or(80, |(columns, _)| usize::from(columns).max(1))
}

fn draw(
    text: &str,
    cursor: usize,
    items: &[MenuItem],
    selected: usize,
    footer: &str,
    cursor_row: &mut usize,
) {
    use std::fmt::Write as _;
    let width = terminal_width();
    let mut output = String::new();
    // Climb from the cursor's wrapped row back to the prompt row before
    // clearing; `\x1b[J` from a lower row would leave the rows above intact.
    if *cursor_row > 0 {
        let _ = write!(output, "\x1b[{}A", *cursor_row);
    }
    output.push_str("\r\x1b[J\x1b[1;36m❯\x1b[0m ");
    output.push_str(text);
    // Rows drawn below the input that the cursor must be moved back up over.
    // Each is clipped to one terminal row so the count stays exact.
    let clip_width = width.saturating_sub(1);
    let mut rows_below = 0usize;
    if items.is_empty() {
        // No completion menu: show the persistent footer hint, if any.
        if !footer.is_empty() {
            let _ = write!(
                output,
                "\r\n{DIM}{}{RESET}",
                clip_columns(footer, clip_width)
            );
            rows_below = 1;
        }
    } else {
        for (index, item) in items.iter().enumerate() {
            let style = if index == selected { INVERSE } else { DIM };
            let row = clip_columns(&format!("{style}  {}", item.label), clip_width);
            let _ = write!(output, "\r\n{row}{RESET}");
        }
        rows_below = items.len();
    }
    let layout = input_layout(text, cursor, width);
    let up = rows_below + layout.last_row - layout.cursor_row;
    if up > 0 {
        let _ = write!(output, "\x1b[{up}A");
    }
    let _ = write!(output, "\x1b[{}G", layout.cursor_column);
    *cursor_row = layout.cursor_row;
    eprint!("{output}");
    let _ = io::stderr().flush();
}

fn clear_menu_and_break_line(text: &str, cursor_row: usize) {
    use std::fmt::Write as _;
    let mut output = String::new();
    if cursor_row > 0 {
        let _ = write!(output, "\x1b[{cursor_row}A");
    }
    let _ = write!(output, "\r\x1b[J\x1b[1;36m❯\x1b[0m {text}\r\n");
    eprint!("{output}");
    let _ = io::stderr().flush();
}

/// A choice shown in an interactive `select_menu`.
pub struct Choice {
    /// Primary label (highlighted when selected).
    pub label: String,
    /// Optional dimmed hint shown after the label.
    pub hint: String,
    /// Section labels are rendered but skipped by keyboard navigation.
    selectable: bool,
}

impl Choice {
    #[must_use]
    pub fn new(label: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            hint: hint.into(),
            selectable: true,
        }
    }

    /// A non-selectable heading used to group related choices.
    #[must_use]
    pub fn section(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            hint: String::new(),
            selectable: false,
        }
    }
}

/// Show an arrow-key selectable menu on the alternate rows below `title` and
/// return the chosen index, or `None` if the user cancels (Esc/Ctrl-C) or no
/// terminal is available. `initial` is the pre-highlighted row.
#[must_use]
pub fn select_menu(title: &str, choices: &[Choice], initial: usize) -> Option<usize> {
    if choices.is_empty() || choices.iter().all(|choice| !choice.selectable) {
        return None;
    }
    if !io::stdin().is_terminal() || terminal::enable_raw_mode().is_err() {
        return None;
    }
    let mut selected = initial.min(choices.len() - 1);
    if !choices[selected].selectable {
        selected = next_selectable(choices, selected, 1)?;
    }
    let result = loop {
        draw_select(title, choices, selected);
        let Ok(Event::Key(key)) = event::read() else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                selected = next_selectable(choices, selected, -1)?;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = next_selectable(choices, selected, 1)?;
            }
            KeyCode::Enter | KeyCode::Tab => break Some(selected),
            KeyCode::Esc => break None,
            KeyCode::Char('c') if control => break None,
            _ => {}
        }
    };
    let _ = terminal::disable_raw_mode();
    // Clear the menu region and leave the cursor on a fresh line.
    eprint!("\r\x1b[J");
    let _ = io::stderr().flush();
    result
}

fn next_selectable(choices: &[Choice], selected: usize, direction: isize) -> Option<usize> {
    let mut index = selected;
    for _ in 0..choices.len() {
        index = if direction < 0 {
            index.checked_sub(1).unwrap_or(choices.len() - 1)
        } else {
            (index + 1) % choices.len()
        };
        if choices[index].selectable {
            return Some(index);
        }
    }
    None
}

fn draw_select(title: &str, choices: &[Choice], selected: usize) {
    use std::fmt::Write as _;
    // Reserve space for the title, scroll indicators, and the prompt area
    // below the picker. The selected row stays centered where possible.
    let terminal_rows = terminal::size().map_or(24, |(_, rows)| usize::from(rows));
    let item_capacity = terminal_rows.saturating_sub(6).max(3);
    let (start, end) = select_window(choices.len(), selected, item_capacity);
    let mut output = format!("\r\x1b[J\x1b[1m{title}\x1b[0m");
    let mut rendered_rows = 0usize;
    if start > 0 {
        let _ = write!(output, "\r\n{DIM}    ↑ {start} more{RESET}");
        rendered_rows += 1;
    }
    for (index, choice) in choices.iter().enumerate().take(end).skip(start) {
        output.push_str("\r\n");
        rendered_rows += 1;
        if !choice.selectable {
            output.push_str(DIM);
            output.push_str("  ─ ");
        } else if index == selected {
            output.push_str(INVERSE);
            output.push_str("  ▸ ");
        } else {
            output.push_str("    ");
        }
        output.push_str(&choice.label);
        output.push_str(RESET);
        if !choice.hint.is_empty() {
            let _ = write!(output, "  {DIM}{}{RESET}", choice.hint);
        }
    }
    if end < choices.len() {
        let _ = write!(output, "\r\n{DIM}    ↓ {} more{RESET}", choices.len() - end);
        rendered_rows += 1;
    }
    // Return the cursor to the title row so the next redraw overwrites cleanly.
    let _ = write!(output, "\x1b[{rendered_rows}A\r");
    eprint!("{output}");
    let _ = io::stderr().flush();
}

fn select_window(length: usize, selected: usize, capacity: usize) -> (usize, usize) {
    if length <= capacity {
        return (0, length);
    }
    let half = capacity / 2;
    let start = selected.saturating_sub(half).min(length - capacity);
    (start, start + capacity)
}

/// Collect relative paths of workspace files for completion, skipping
/// hidden entries and common build/dependency directories.
fn workspace_files(root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((directory, depth)) = stack.pop() {
        if depth > DEPTH_LIMIT || files.len() >= FILE_LIMIT {
            continue;
        }
        let Ok(entries) = fs::read_dir(&directory) else {
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
            } else if let Ok(relative) = path.strip_prefix(root) {
                files.push(relative.to_string_lossy().into_owned());
                if files.len() >= FILE_LIMIT {
                    break;
                }
            }
        }
    }
    files.sort_unstable();
    files
}

/// Rank name-prefix matches before path-substring matches, case-insensitive.
fn filter_files(files: &[String], query: &str) -> Vec<String> {
    let query = query.to_lowercase();
    let mut prefix_matches = Vec::new();
    let mut substring_matches = Vec::new();
    for path in files {
        let lower = path.to_lowercase();
        let file_name = lower.rsplit('/').next().unwrap_or(&lower);
        if file_name.starts_with(&query) {
            prefix_matches.push(path.clone());
        } else if lower.contains(&query) {
            substring_matches.push(path.clone());
        }
        if prefix_matches.len() >= MENU_LIMIT {
            break;
        }
    }
    prefix_matches.extend(substring_matches);
    prefix_matches.truncate(MENU_LIMIT);
    prefix_matches
}

#[cfg(test)]
mod tests {
    use super::{
        Choice, CompletionContext, Editor, clip_columns, completion_context, filter_files,
        input_layout, next_selectable, select_window,
    };

    fn chars(text: &str) -> Vec<char> {
        text.chars().collect()
    }

    #[test]
    fn wrapped_input_layout_places_the_cursor_row_aware() {
        // Fits on one row: row 0, column = prompt(2) + cursor + 1.
        let layout = input_layout("hello", 2, 80);
        assert_eq!(
            (layout.cursor_row, layout.cursor_column, layout.last_row),
            (0, 5, 0)
        );
        // 100 chars at width 40: 102 total columns spread over rows 0..=2.
        let text = "x".repeat(100);
        let layout = input_layout(&text, 100, 40);
        assert_eq!(
            (layout.cursor_row, layout.cursor_column, layout.last_row),
            (2, 23, 2)
        );
        let layout = input_layout(&text, 0, 40);
        assert_eq!((layout.cursor_row, layout.cursor_column), (0, 3));
        // Text exactly filling two rows: the terminal defers the wrap, so
        // the end-of-line cursor rests on the last row's final column.
        let text = "y".repeat(78);
        let layout = input_layout(&text, 78, 40);
        assert_eq!(
            (layout.cursor_row, layout.cursor_column, layout.last_row),
            (1, 40, 1)
        );
        // Wide characters weigh two display columns.
        let layout = input_layout("漢漢漢", 3, 8);
        assert_eq!(
            (layout.cursor_row, layout.cursor_column, layout.last_row),
            (0, 8, 0)
        );
    }

    #[test]
    fn menu_and_footer_rows_clip_to_one_terminal_row() {
        assert_eq!(clip_columns("short", 10), "short");
        assert_eq!(clip_columns("abcdefghij", 4), "abcd");
        // ANSI escape sequences pass through without counting as columns.
        assert_eq!(clip_columns("\x1b[2mabcdef\x1b[0m", 3), "\x1b[2mabc");
        // Wide characters weigh two columns, so only two fit in four.
        assert_eq!(clip_columns("漢漢漢", 4), "漢漢");
    }

    #[test]
    fn slash_context_covers_first_token_only() {
        assert_eq!(
            completion_context(&chars("/mo"), 3),
            CompletionContext::Slash("mo".to_owned())
        );
        assert_eq!(
            completion_context(&chars("/model x"), 8),
            CompletionContext::None
        );
        assert_eq!(
            completion_context(&chars("hello"), 5),
            CompletionContext::None
        );
    }

    #[test]
    fn file_context_tracks_at_token() {
        assert_eq!(
            completion_context(&chars("read @src/ma"), 12),
            CompletionContext::File {
                start: 6,
                query: "src/ma".to_owned()
            }
        );
        assert_eq!(
            completion_context(&chars("mail me"), 7),
            CompletionContext::None
        );
    }

    #[test]
    fn file_filter_prefers_name_prefix_matches() {
        let files = vec![
            "docs/main-notes.md".to_owned(),
            "src/lib.rs".to_owned(),
            "src/main.rs".to_owned(),
        ];
        let matches = filter_files(&files, "ma");
        assert_eq!(matches[0], "docs/main-notes.md");
        assert_eq!(matches[1], "src/main.rs");
        assert!(!matches.contains(&"src/lib.rs".to_owned()));
    }

    #[test]
    fn grouped_menu_navigation_skips_section_headings() {
        let choices = vec![
            Choice::section("openai"),
            Choice::new("gpt", ""),
            Choice::section("anthropic"),
            Choice::new("claude", ""),
        ];
        assert_eq!(next_selectable(&choices, 1, 1), Some(3));
        assert_eq!(next_selectable(&choices, 3, 1), Some(1));
        assert_eq!(next_selectable(&choices, 1, -1), Some(3));
    }

    #[test]
    fn selection_window_scrolls_to_follow_highlighted_row() {
        assert_eq!(select_window(100, 0, 10), (0, 10));
        assert_eq!(select_window(100, 50, 10), (45, 55));
        assert_eq!(select_window(100, 99, 10), (90, 100));
        assert_eq!(select_window(5, 4, 10), (0, 5));
    }

    #[test]
    fn paste_placeholders_expand_back_to_full_text_on_submit() {
        let mut editor = Editor::new(std::env::temp_dir());
        editor
            .pastes
            .push("line one\nline two\nline three".to_owned());
        editor.pastes.push("alpha\nbeta".to_owned());
        let first = editor.paste_placeholder(0);
        let second = editor.paste_placeholder(1);
        assert_eq!(first, "[pasted #1 +3 lines]");
        assert_eq!(second, "[pasted #2 +2 lines]");
        let line = format!("explain this: {first} and compare with {second} thanks");
        let expanded = editor.expand_pastes(&line);
        assert!(expanded.contains("line two"), "got: {expanded}");
        assert!(expanded.contains("alpha\nbeta"), "got: {expanded}");
        assert!(!expanded.contains("[pasted"), "got: {expanded}");
        // An edited placeholder no longer matches and stays literal.
        let edited = editor.expand_pastes("[pasted #1 +99 lines]");
        assert_eq!(edited, "[pasted #1 +99 lines]");
    }
}
