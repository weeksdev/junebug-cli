//! Lightweight streaming Markdown-to-ANSI rendering for the interactive UI.
//!
//! Deltas arrive in arbitrary chunks, so rendering is line-buffered: styling
//! is applied when a full line is available and the remainder is flushed at
//! the end of a turn. This intentionally covers the common subset (headers,
//! code fences, inline code, bold, bullets) rather than full `CommonMark`.

const RESET: &str = "\x1b[0m";
const BOLD_ON: &str = "\x1b[1m";
const BOLD_OFF: &str = "\x1b[22m";
const DIM: &str = "\x1b[2m";
const CYAN_BOLD: &str = "\x1b[1;36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const FG_RESET: &str = "\x1b[39m";

pub struct Renderer {
    /// Emit `\r\n` line endings (required while the terminal is in raw mode).
    crlf: bool,
    in_code_block: bool,
    partial: String,
}

impl Renderer {
    #[must_use]
    pub const fn new(crlf: bool) -> Self {
        Self {
            crlf,
            in_code_block: false,
            partial: String::new(),
        }
    }

    /// Feed a streamed delta; returns rendered output for every line that
    /// completed with this chunk.
    pub fn push(&mut self, delta: &str) -> String {
        self.partial.push_str(delta);
        let mut output = String::new();
        while let Some(newline) = self.partial.find('\n') {
            let line: String = self.partial.drain(..=newline).collect();
            output.push_str(&self.render_line(line.trim_end_matches(['\n', '\r'])));
            output.push_str(self.line_ending());
        }
        output
    }

    /// Flush any unterminated final line.
    pub fn finish(&mut self) -> String {
        if self.partial.is_empty() {
            return String::new();
        }
        let line = std::mem::take(&mut self.partial);
        let mut output = self.render_line(&line);
        output.push_str(self.line_ending());
        output
    }

    const fn line_ending(&self) -> &'static str {
        if self.crlf { "\r\n" } else { "\n" }
    }

    fn render_line(&mut self, line: &str) -> String {
        if line.trim_start().starts_with("```") {
            self.in_code_block = !self.in_code_block;
            return format!("{DIM}{line}{RESET}");
        }
        if self.in_code_block {
            return format!("{GREEN}{line}{RESET}");
        }
        if line.starts_with('#') {
            let header = line.trim_start_matches('#').trim_start();
            return format!("{CYAN_BOLD}{header}{RESET}");
        }
        let (indent, rest) = split_indent(line);
        if let Some(item) = rest.strip_prefix("- ").or_else(|| rest.strip_prefix("* ")) {
            return format!("{indent}{CYAN_BOLD}•{RESET} {}", render_inline(item));
        }
        format!("{indent}{}", render_inline(rest))
    }
}

fn split_indent(line: &str) -> (&str, &str) {
    let end = line.len() - line.trim_start().len();
    line.split_at(end)
}

/// Apply inline styling for `` `code` `` and `**bold**` spans.
fn render_inline(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut bold = false;
    let mut code = false;
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '`' {
            code = !code;
            output.push_str(if code { YELLOW } else { FG_RESET });
        } else if character == '*' && !code && characters.peek() == Some(&'*') {
            characters.next();
            bold = !bold;
            output.push_str(if bold { BOLD_ON } else { BOLD_OFF });
        } else {
            output.push(character);
        }
    }
    if code {
        output.push_str(FG_RESET);
    }
    if bold {
        output.push_str(BOLD_OFF);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::Renderer;

    #[test]
    fn styles_bold_and_code_across_split_deltas() {
        let mut renderer = Renderer::new(false);
        let mut output = renderer.push("this is **bo");
        output.push_str(&renderer.push("ld** and `code`\n"));
        assert!(output.contains("\x1b[1mbold\x1b[22m"));
        assert!(output.contains("\x1b[33mcode\x1b[39m"));
    }

    #[test]
    fn code_fence_state_spans_lines() {
        let mut renderer = Renderer::new(false);
        let output = renderer.push("```rust\nlet x = 1;\n```\nafter\n");
        assert!(output.contains("\x1b[32mlet x = 1;\x1b[0m"));
        assert!(output.ends_with("after\n"));
    }

    #[test]
    fn finish_flushes_partial_line_and_crlf_mode_uses_crlf() {
        let mut renderer = Renderer::new(true);
        assert_eq!(renderer.push("no newline yet"), "");
        assert_eq!(renderer.finish(), "no newline yet\r\n");
        assert_eq!(renderer.finish(), "");
    }

    #[test]
    fn headers_and_bullets_are_styled() {
        let mut renderer = Renderer::new(false);
        let output = renderer.push("## Title\n- item\n");
        assert!(output.contains("\x1b[1;36mTitle\x1b[0m"));
        assert!(output.contains("\x1b[1;36m•\x1b[0m item"));
    }
}
