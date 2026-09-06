//! Shared "find a fenced or bare JSON array in a model reply" scanner. Every
//! role in this codebase that asks a model to end its reply with a
//! ```` ```json ```` fenced array (swarm task plans, abductive-reasoning
//! hypothesis/evidence arrays) uses this to locate it before handing the
//! slice to `serde_json`. Deliberately dumb: the first `[` after a
//! ` ```json ` fence (or the first `[` at all, if no fence is present) to the
//! last `]` in the whole reply. Callers own their own empty/parse-error
//! messages — this only locates text, it never validates or parses it.

/// The raw JSON-array slice of `text`, or `None` when no bracket pair can be
/// located at all.
#[must_use]
pub fn find_array(text: &str) -> Option<&str> {
    find_delimited(text, '[', ']')
}

/// Same idea as `find_array`, for a single fenced or bare JSON *object*
/// (`{...}`) instead of an array — used by builders that ask a model to end
/// its reply with one spec object (an agent or tool definition) rather than
/// a list.
#[must_use]
pub fn find_object(text: &str) -> Option<&str> {
    find_delimited(text, '{', '}')
}

fn find_delimited(text: &str, open: char, close: char) -> Option<&str> {
    let start = match text.find("```json") {
        Some(fence) => text[fence..].find(open).map(|offset| fence + offset),
        None => text.find(open),
    }?;
    let end = text.rfind(close)?;
    if start > end {
        return None;
    }
    Some(&text[start..=end])
}

#[cfg(test)]
mod tests {
    use super::find_array;

    #[test]
    fn finds_a_fenced_array_after_surrounding_prose() {
        let text = "Here is the plan:\n```json\n[{\"a\":1}]\n```\nDone.";
        assert_eq!(find_array(text), Some("[{\"a\":1}]"));
    }

    #[test]
    fn finds_a_bare_array_with_no_fence() {
        let text = "sure: [1, 2, 3] there you go";
        assert_eq!(find_array(text), Some("[1, 2, 3]"));
    }

    #[test]
    fn returns_none_when_there_is_no_bracket_pair() {
        assert_eq!(find_array("no json here"), None);
        assert_eq!(find_array("only an opening [ bracket"), None);
    }

    #[test]
    fn returns_none_when_brackets_are_reversed() {
        assert_eq!(find_array("] before ["), None);
    }

    #[test]
    fn finds_a_fenced_object() {
        let text =
            "Here's the spec:\n```json\n{\"name\":\"a\",\"tools\":[\"read_file\"]}\n```\nDone.";
        assert_eq!(
            super::find_object(text),
            Some("{\"name\":\"a\",\"tools\":[\"read_file\"]}")
        );
    }

    #[test]
    fn find_object_returns_none_with_no_brace_pair() {
        assert_eq!(super::find_object("no json here"), None);
    }
}
