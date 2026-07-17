//! The `fetch_url` tool: a bounded HTTP GET with HTML-to-text extraction,
//! completing the `web_search` loop so the model can read a result page.
//! Network-risk by policy — outside yolo every call needs an explicit
//! approval that shows the exact URL being fetched.

use std::io::Read as _;
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::{CONTENT_TYPE, USER_AGENT};

/// Bytes read from the response body at most, before text extraction.
const BODY_CAP: usize = 2 * 1024 * 1024;
pub const DEFAULT_MAX_CHARS: usize = 20_000;

/// Fetch a URL and return readable text: HTML is reduced to its text
/// content, other `text/*` and JSON bodies pass through, and everything is
/// truncated to `max_chars`.
///
/// # Errors
///
/// Returns an error for a non-HTTP(S) URL, transport failures, a
/// non-success status, or a non-text content type.
pub fn fetch_url(url: &str, max_chars: usize) -> Result<String, String> {
    let url = url.trim();
    if !url.starts_with("https://") && !url.starts_with("http://") {
        return Err("only http:// and https:// URLs can be fetched".to_owned());
    }
    let max_chars = max_chars.clamp(1_000, 100_000);
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|error| error.to_string())?;
    let response = client
        .get(url)
        .header(
            USER_AGENT,
            "Mozilla/5.0 (compatible) junebug-cli (+https://github.com/weeksdev/junebug-cli)",
        )
        .send()
        .map_err(|error| format!("fetch failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("fetch failed with HTTP {status}"));
    }
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let is_html = content_type.contains("html");
    let is_text = is_html
        || content_type.contains("text/")
        || content_type.contains("json")
        || content_type.contains("xml")
        || content_type.is_empty();
    if !is_text {
        return Err(format!(
            "unsupported content type for fetch: {content_type}"
        ));
    }
    // Read at most BODY_CAP bytes so a huge (or endless) body cannot
    // exhaust memory; truncation at this layer is reported below.
    let mut body = Vec::new();
    response
        .take(BODY_CAP as u64)
        .read_to_end(&mut body)
        .map_err(|error| format!("fetch body could not be read: {error}"))?;
    let text = String::from_utf8_lossy(&body);
    let extracted = if is_html {
        html_to_text(&text)
    } else {
        text.into_owned()
    };
    Ok(truncate_chars(&extracted, max_chars))
}

/// Reduce an HTML document to readable text: script/style/head noise
/// removed, block boundaries become newlines, tags stripped, entities
/// decoded, and whitespace collapsed.
fn html_to_text(html: &str) -> String {
    let without_blocks = remove_elements(html, &["script", "style", "noscript", "svg"]);
    let mut text = String::with_capacity(without_blocks.len() / 2);
    let mut rest = without_blocks.as_str();
    while let Some(open) = rest.find('<') {
        text.push_str(&rest[..open]);
        let Some(close) = rest[open..].find('>') else {
            break;
        };
        let tag = rest[open + 1..open + close].trim_start_matches('/');
        let tag_name: String = tag
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect();
        // Block-level boundaries become line breaks so paragraphs, list
        // items, headings, and table rows stay visually separate.
        if matches!(
            tag_name.to_ascii_lowercase().as_str(),
            "p" | "div"
                | "br"
                | "li"
                | "tr"
                | "h1"
                | "h2"
                | "h3"
                | "h4"
                | "h5"
                | "h6"
                | "ul"
                | "ol"
                | "table"
                | "section"
                | "article"
                | "header"
                | "footer"
                | "blockquote"
                | "pre"
                | "form"
                | "hr"
        ) {
            text.push('\n');
        }
        rest = &rest[open + close + 1..];
    }
    text.push_str(rest);
    let decoded = crate::websearch::decode_entities(&text);
    collapse_whitespace(&decoded)
}

/// Remove `<name …>…</name>` elements wholesale, case-insensitively.
fn remove_elements(html: &str, names: &[&str]) -> String {
    let mut output = html.to_owned();
    for name in names {
        let mut result = String::with_capacity(output.len());
        let lower = output.to_ascii_lowercase();
        let open_marker = format!("<{name}");
        let close_marker = format!("</{name}");
        let mut position = 0;
        while let Some(start) = lower[position..].find(&open_marker) {
            let start = position + start;
            result.push_str(&output[position..start]);
            let Some(end) = lower[start..].find(&close_marker) else {
                position = output.len();
                break;
            };
            let after = start + end;
            let Some(close) = lower[after..].find('>') else {
                position = output.len();
                break;
            };
            position = after + close + 1;
        }
        result.push_str(&output[position..]);
        output = result;
    }
    output
}

/// Collapse runs of spaces within lines and runs of blank lines, trimming
/// trailing whitespace — HTML source indentation otherwise dominates the
/// extracted text.
fn collapse_whitespace(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    for line in text.lines() {
        let collapsed = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if collapsed.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        output.push_str(&collapsed);
        output.push('\n');
    }
    output.trim().to_owned()
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let mut output: String = text.chars().take(max_chars).collect();
    output.push_str("\n…[truncated]");
    output
}

#[cfg(test)]
mod tests {
    use super::{collapse_whitespace, fetch_url, html_to_text, truncate_chars};

    #[test]
    fn html_reduces_to_readable_text() {
        let html = r#"<html><head><title>Docs</title><style>body { color: red; }</style></head>
        <body>
          <script>var tracking = "noise";</script>
          <h1>Getting&nbsp;Started</h1>
          <p>Install the tool with <code>cargo install</code> &amp; run it.</p>
          <ul><li>First</li><li>Second</li></ul>
          <noscript>Enable JS</noscript>
        </body></html>"#;
        let text = html_to_text(html);
        assert!(text.contains("Getting Started"), "got: {text}");
        assert!(
            text.contains("Install the tool with cargo install & run it."),
            "got: {text}"
        );
        let first = text.find("First").expect("First present");
        let second = text.find("Second").expect("Second present");
        assert!(
            first < second && text[first..second].contains('\n'),
            "list items must land on separate lines: {text}"
        );
        assert!(!text.contains("tracking"), "script must be removed: {text}");
        assert!(
            !text.contains("color: red"),
            "style must be removed: {text}"
        );
        assert!(
            !text.contains("Enable JS"),
            "noscript must be removed: {text}"
        );
    }

    #[test]
    fn whitespace_and_length_are_bounded() {
        assert_eq!(collapse_whitespace("a   b\n\n\n\n\nc  \n"), "a b\n\nc");
        let truncated = truncate_chars(&"x".repeat(50), 10);
        assert!(truncated.starts_with("xxxxxxxxxx\n…[truncated]"));
        assert_eq!(truncate_chars("short", 10), "short");
    }

    #[test]
    fn non_http_urls_are_rejected_without_network() {
        assert!(fetch_url("ftp://example.com", 5_000).is_err());
        assert!(fetch_url("file:///etc/passwd", 5_000).is_err());
        assert!(fetch_url("  javascript:alert(1)", 5_000).is_err());
    }

    /// Live network check, excluded from CI: `cargo test -- --ignored`.
    #[test]
    #[ignore = "requires network access"]
    fn live_fetch_returns_page_text() {
        let text = fetch_url("https://example.com", 5_000).expect("fetch");
        assert!(text.contains("Example Domain"), "got: {text}");
        assert!(!text.contains("<html"), "tags must be stripped: {text}");
    }
}
