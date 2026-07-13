//! Dependency-free line diff for showing what a `write_file` changes.
//!
//! Produces hunks of `-`/`+` lines with two lines of surrounding context,
//! separated by `···`. This is a UI affordance only: diffs are shown to the
//! user (approval prompts, activity stream, JSONL events) and never added to
//! the model context.

/// Number of unchanged lines kept around each hunk.
const CONTEXT: usize = 2;
/// LCS cost cap (old lines × new lines). Beyond it the middle section is
/// rendered as a whole-block replacement instead of a minimal diff.
const MAX_LCS_CELLS: usize = 500_000;

/// Render the line diff between `old` and `new`. Empty when identical.
#[must_use]
pub fn unified(old: &str, new: &str) -> String {
    if old == new {
        return String::new();
    }
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    // Trim the common prefix and suffix so LCS only sees the changed middle.
    let mut start = 0;
    while start < old_lines.len() && start < new_lines.len() && old_lines[start] == new_lines[start]
    {
        start += 1;
    }
    let mut old_end = old_lines.len();
    let mut new_end = new_lines.len();
    while old_end > start && new_end > start && old_lines[old_end - 1] == new_lines[new_end - 1] {
        old_end -= 1;
        new_end -= 1;
    }

    let old_mid = &old_lines[start..old_end];
    let new_mid = &new_lines[start..new_end];
    let mut ops: Vec<(char, &str)> = if old_mid.len().saturating_mul(new_mid.len()) > MAX_LCS_CELLS
    {
        old_mid
            .iter()
            .map(|line| ('-', *line))
            .chain(new_mid.iter().map(|line| ('+', *line)))
            .collect()
    } else {
        lcs_ops(old_mid, new_mid)
    };

    // Surround the changed middle with context from the trimmed edges.
    let prefix_context = &old_lines[start.saturating_sub(CONTEXT)..start];
    let suffix_context = &old_lines[old_end..(old_end + CONTEXT).min(old_lines.len())];
    let mut all: Vec<(char, &str)> =
        Vec::with_capacity(prefix_context.len() + ops.len() + suffix_context.len());
    all.extend(prefix_context.iter().map(|line| (' ', *line)));
    all.append(&mut ops);
    all.extend(suffix_context.iter().map(|line| (' ', *line)));

    render_hunks(&all)
}

/// Cap a rendered diff at `max_lines`, appending a truncation note.
#[must_use]
pub fn clip(diff: &str, max_lines: usize) -> String {
    let total = diff.lines().count();
    if total <= max_lines {
        return diff.to_owned();
    }
    let mut clipped: Vec<&str> = diff.lines().take(max_lines).collect();
    let hidden = total - max_lines;
    let note = format!("… ({hidden} more diff lines)");
    clipped.push(&note);
    clipped.join("\n")
}

/// Minimal line ops between two slices via longest-common-subsequence.
fn lcs_ops<'a>(old: &[&'a str], new: &[&'a str]) -> Vec<(char, &'a str)> {
    let rows = old.len();
    let columns = new.len();
    let mut table = vec![vec![0u32; columns + 1]; rows + 1];
    for i in (0..rows).rev() {
        for j in (0..columns).rev() {
            table[i][j] = if old[i] == new[j] {
                table[i + 1][j + 1] + 1
            } else {
                table[i + 1][j].max(table[i][j + 1])
            };
        }
    }
    let mut ops = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < rows && j < columns {
        if old[i] == new[j] {
            ops.push((' ', old[i]));
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            ops.push(('-', old[i]));
            i += 1;
        } else {
            ops.push(('+', new[j]));
            j += 1;
        }
    }
    ops.extend(old[i..].iter().map(|line| ('-', *line)));
    ops.extend(new[j..].iter().map(|line| ('+', *line)));
    ops
}

/// Keep `CONTEXT` unchanged lines around each run of changes; longer
/// unchanged stretches become `···` separators.
fn render_hunks(ops: &[(char, &str)]) -> String {
    let change_positions: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(index, (kind, _))| (*kind != ' ').then_some(index))
        .collect();
    let Some(&first) = change_positions.first() else {
        return String::new();
    };
    let last = *change_positions.last().expect("non-empty");

    let mut out = String::new();
    let mut position = first.saturating_sub(CONTEXT);
    let end = (last + CONTEXT + 1).min(ops.len());
    let mut changes = change_positions.iter().peekable();
    while position < end {
        // Skip long unchanged stretches between hunks.
        if let Some(&&next_change) = changes.peek() {
            if position + CONTEXT < next_change.saturating_sub(CONTEXT) && ops[position].0 == ' ' {
                if !out.is_empty() {
                    out.push_str("···\n");
                }
                position = next_change - CONTEXT;
                continue;
            }
            if position > next_change {
                changes.next();
                continue;
            }
        }
        let (kind, line) = ops[position];
        if kind == ' ' {
            out.push_str("  ");
        } else {
            out.push(kind);
            out.push(' ');
        }
        out.push_str(line);
        out.push('\n');
        position += 1;
    }
    out.trim_end_matches('\n').to_owned()
}

#[cfg(test)]
mod tests {
    use super::{clip, unified};
    use std::fmt::Write as _;

    fn numbered_lines(prefix: &str, range: std::ops::Range<usize>) -> String {
        range.fold(String::new(), |mut text, n| {
            let _ = writeln!(text, "{prefix}{n}");
            text
        })
    }

    #[test]
    fn identical_content_yields_no_diff() {
        assert_eq!(unified("a\nb\n", "a\nb\n"), "");
    }

    #[test]
    fn changed_line_shows_removal_and_addition_with_context() {
        let old = "zero\none\ntwo\nthree\nfour\nfive\nsix\n";
        let new = "zero\none\ntwo\nTHREE\nfour\nfive\nsix\n";
        let diff = unified(old, new);
        assert!(diff.contains("- three"), "diff was: {diff}");
        assert!(diff.contains("+ THREE"), "diff was: {diff}");
        assert!(diff.contains("  two"), "context line missing: {diff}");
        assert!(diff.contains("  four"), "context line missing: {diff}");
        assert!(
            !diff.contains("zero") && !diff.contains("six"),
            "line outside the 2-line context window leaked: {diff}"
        );
    }

    #[test]
    fn new_file_is_all_additions() {
        let diff = unified("", "alpha\nbeta\n");
        assert_eq!(diff, "+ alpha\n+ beta");
    }

    #[test]
    fn distant_changes_split_into_hunks() {
        let old = numbered_lines("line", 1..21);
        let new = old
            .replace("line2\n", "LINE2\n")
            .replace("line19\n", "LINE19\n");
        let diff = unified(&old, &new);
        assert!(diff.contains("···"), "expected a hunk separator: {diff}");
        assert!(diff.contains("- line2"));
        assert!(diff.contains("+ LINE19"));
        assert!(
            !diff.contains("line10"),
            "unchanged middle must be elided: {diff}"
        );
    }

    #[test]
    fn oversized_inputs_fall_back_to_block_replacement() {
        let old = numbered_lines("old", 0..1_000);
        let new = numbered_lines("new", 0..1_000);
        let diff = unified(&old, &new);
        assert!(diff.contains("- old0"));
        assert!(diff.contains("+ new999"));
    }

    #[test]
    fn clip_truncates_with_a_note() {
        let diff = "+ a\n+ b\n+ c\n+ d";
        assert_eq!(clip(diff, 2), "+ a\n+ b\n… (2 more diff lines)");
        assert_eq!(clip(diff, 10), diff);
    }
}
