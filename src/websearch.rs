//! The `web_search` tool: `DuckDuckGo`'s keyless HTML endpoint, parsed into
//! titles, URLs, and snippets. Network-risk by policy — outside yolo every
//! call needs an explicit approval that shows the outgoing query, because
//! the query text leaves the machine.

use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::USER_AGENT;

const ENDPOINT: &str = "https://html.duckduckgo.com/html/";
const MAX_QUERY_CHARS: usize = 500;
pub const DEFAULT_RESULTS: usize = 5;
const MAX_RESULTS: usize = 10;

pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Search the web and return a numbered plain-text result list.
///
/// # Errors
///
/// Returns an error for an empty/oversized query, transport failures, or a
/// non-success HTTP status.
pub fn web_search(query: &str, max_results: usize) -> Result<String, String> {
    let query = query.trim();
    if query.is_empty() {
        return Err("search query cannot be empty".to_owned());
    }
    if query.chars().count() > MAX_QUERY_CHARS {
        return Err(format!("search query exceeds {MAX_QUERY_CHARS} characters"));
    }
    let limit = max_results.clamp(1, MAX_RESULTS);
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|error| error.to_string())?;
    let response = client
        .get(ENDPOINT)
        .query(&[("q", query)])
        // DuckDuckGo serves the HTML endpoint to browser-identified agents;
        // the suffix keeps the client honest about what it is.
        .header(
            USER_AGENT,
            "Mozilla/5.0 (compatible) junebug-cli (+https://github.com/weeksdev/junebug-cli)",
        )
        .send()
        .map_err(|error| format!("web search request failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("web search failed with HTTP {status}"));
    }
    let html = response
        .text()
        .map_err(|error| format!("web search response could not be read: {error}"))?;
    let results = parse_results(&html, limit);
    if results.is_empty() {
        return Ok("no results".to_owned());
    }
    Ok(format_results(&results))
}

/// Extract result blocks from the `DuckDuckGo` HTML page. Each block carries
/// a `result__a` title anchor (whose href is a redirect wrapping the real
/// URL in a `uddg` parameter) and a `result__snippet` anchor. Manual
/// scanning, not an HTML parser — the page is simple and this avoids a
/// dependency; unit tests pin the expected shape.
fn parse_results(html: &str, limit: usize) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let mut rest = html;
    while results.len() < limit {
        let Some(anchor) = rest.find("class=\"result__a\"") else {
            break;
        };
        let tag_start = rest[..anchor].rfind("<a ").unwrap_or(0);
        let href = attribute(&rest[tag_start..], "href").unwrap_or_default();
        let after = &rest[anchor..];
        let (Some(open), Some(close)) = (after.find('>'), after.find("</a>")) else {
            break;
        };
        if open >= close {
            break;
        }
        let title = decode_entities(&strip_tags(&after[open + 1..close]));
        // Snippets belong to their own result block: stop looking at the
        // next result's title anchor so a snippetless result never steals
        // the following one's text.
        let block_end = after[1..]
            .find("class=\"result__a\"")
            .map_or(after.len(), |position| position + 1);
        let snippet = after[..block_end]
            .find("result__snippet")
            .and_then(|position| {
                let section = &after[position..block_end];
                let open = section.find('>')?;
                let close = section.find("</a>")?;
                (open < close).then(|| decode_entities(&strip_tags(&section[open + 1..close])))
            })
            .unwrap_or_default();
        let url = resolve_result_url(&decode_entities(&href));
        if !url.is_empty() && !title.is_empty() {
            results.push(SearchResult {
                title,
                url,
                snippet,
            });
        }
        rest = &after[close + 4..];
    }
    results
}

fn format_results(results: &[SearchResult]) -> String {
    use std::fmt::Write as _;
    let mut output = String::new();
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            output.push('\n');
        }
        let _ = write!(output, "{}. {}\n   {}", index + 1, result.title, result.url);
        if !result.snippet.is_empty() {
            let _ = write!(output, "\n   {}", result.snippet);
        }
        output.push('\n');
    }
    output
}

/// The value of `name="…"` inside the first tag of `tag`.
fn attribute(tag: &str, name: &str) -> Option<String> {
    let start = tag.find(&format!("{name}=\""))? + name.len() + 2;
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_owned())
}

/// `DuckDuckGo` result hrefs are redirects like
/// `//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F&rut=…`; unwrap
/// the `uddg` parameter to the real destination. Direct URLs pass through.
fn resolve_result_url(href: &str) -> String {
    if let Some(position) = href.find("uddg=") {
        let encoded = &href[position + 5..];
        let encoded = encoded.split('&').next().unwrap_or(encoded);
        return percent_decode(encoded);
    }
    if let Some(protocol_relative) = href.strip_prefix("//") {
        return format!("https://{protocol_relative}");
    }
    href.to_owned()
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let (Some(high), Some(low)) = (
                (bytes[index + 1] as char).to_digit(16),
                (bytes[index + 2] as char).to_digit(16),
            )
        {
            output.push(u8::try_from(high * 16 + low).unwrap_or(b'?'));
            index += 3;
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&output).into_owned()
}

fn strip_tags(value: &str) -> String {
    let mut output = String::new();
    let mut in_tag = false;
    for character in value.chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => output.push(character),
            _ => {}
        }
    }
    output.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn decode_entities(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_RESULTS, decode_entities, format_results, parse_results, percent_decode,
        resolve_result_url, web_search,
    };

    const FIXTURE: &str = r#"
    <div class="result results_links results_links_deep web-result ">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rust%2Dlang.org%2F&amp;rut=abc123">Rust Programming &amp; Language</a>
      </h2>
      <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rust%2Dlang.org%2F">A language empowering everyone to build <b>reliable</b> software.</a>
    </div>
    <div class="result">
      <a rel="nofollow" class="result__a" href="https://doc.rust-lang.org/book/">The Book</a>
    </div>
    <div class="result">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fcrates.io%2F&amp;rut=def">crates.io</a>
      <a class="result__snippet">The Rust community&#x27;s crate registry</a>
    </div>
    "#;

    #[test]
    fn parses_titles_urls_and_snippets_from_result_blocks() {
        let results = parse_results(FIXTURE, DEFAULT_RESULTS);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].title, "Rust Programming & Language");
        assert_eq!(results[0].url, "https://www.rust-lang.org/");
        assert_eq!(
            results[0].snippet,
            "A language empowering everyone to build reliable software."
        );
        // A direct (non-redirect) href passes through, and a result with no
        // snippet of its own must not steal the next block's snippet.
        assert_eq!(results[1].url, "https://doc.rust-lang.org/book/");
        assert_eq!(results[1].snippet, "");
        assert_eq!(results[2].title, "crates.io");
        assert_eq!(results[2].snippet, "The Rust community's crate registry");
    }

    #[test]
    fn result_limit_is_respected() {
        assert_eq!(parse_results(FIXTURE, 2).len(), 2);
        assert!(parse_results("<html>no results here</html>", 5).is_empty());
    }

    #[test]
    fn redirect_urls_unwrap_and_decode() {
        assert_eq!(
            resolve_result_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%20b&rut=x"),
            "https://example.com/a b"
        );
        assert_eq!(
            resolve_result_url("//lite.example.com/page"),
            "https://lite.example.com/page"
        );
        assert_eq!(
            resolve_result_url("https://direct.example"),
            "https://direct.example"
        );
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(
            decode_entities("a &amp;&lt;tag&gt; &#x27;x&#x27;"),
            "a &<tag> 'x'"
        );
    }

    #[test]
    fn formatting_numbers_results_with_indented_url_and_snippet() {
        let results = parse_results(FIXTURE, 1);
        let text = format_results(&results);
        assert!(text.starts_with("1. Rust Programming & Language\n   https://www.rust-lang.org/"));
        assert!(text.contains("\n   A language empowering"));
    }

    #[test]
    fn empty_and_oversized_queries_are_rejected_without_network() {
        assert!(web_search("   ", 5).is_err());
        assert!(web_search(&"q".repeat(501), 5).is_err());
    }

    /// Live network check, excluded from CI: `cargo test -- --ignored`.
    #[test]
    #[ignore = "requires network access"]
    fn live_search_returns_results() {
        let output = web_search("rust programming language", 3).expect("search");
        assert!(output.starts_with("1. "), "got: {output}");
        assert!(output.contains("http"), "got: {output}");
    }
}
