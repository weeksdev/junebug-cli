//! Deterministic, inspectable context budgeting for provider requests.

use serde_json::{Value, json};

/// Compact a conversation to a character budget while preserving the system
/// prompt and newest complete messages.
#[must_use]
pub fn compact(messages: &[Value], budget: usize) -> Vec<Value> {
    if serialized_len(messages) <= budget {
        return messages.to_vec();
    }
    let system = messages
        .first()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
        .cloned();
    let start = usize::from(system.is_some());
    let remaining_budget = budget.saturating_sub(system.as_ref().map_or(0, serialized_len_one));
    let cut = tail_cut(messages, start, remaining_budget);
    let omitted = cut - start;
    let mut compacted = Vec::new();
    if let Some(system) = system {
        compacted.push(system);
    }
    if omitted > 0 {
        compacted.push(json!({"role":"system", "content": format!("Context compaction: {omitted} earlier conversation messages were omitted. Do not assume their contents; inspect files or ask for clarification if needed.")}));
    }
    compacted.extend_from_slice(&messages[cut..]);
    compacted
}

/// Index of the oldest message to keep so the kept suffix (`messages[cut..]`)
/// fits within `budget` characters — the newest message is always kept even
/// when it alone exceeds the budget. Never lands on a `tool` message whose
/// preceding assistant `tool_calls` message got cut, since providers reject
/// that pairing. `start` bounds the search from below — e.g. past a leading
/// system message the caller keeps separately.
#[must_use]
pub fn tail_cut(messages: &[Value], start: usize, budget: usize) -> usize {
    let mut used = 0usize;
    let mut cut = messages.len();
    while cut > start {
        let length = serialized_len_one(&messages[cut - 1]);
        if used.saturating_add(length) > budget && cut < messages.len() {
            break;
        }
        used = used.saturating_add(length);
        cut -= 1;
    }
    while cut > start && messages[cut].get("role").and_then(Value::as_str) == Some("tool") {
        cut -= 1;
    }
    cut
}

#[must_use]
pub fn serialized_len(messages: &[Value]) -> usize {
    messages.iter().map(serialized_len_one).sum()
}

fn serialized_len_one(message: &Value) -> usize {
    serde_json::to_string(message).map_or(0, |encoded| encoded.len())
}

#[cfg(test)]
mod tests {
    use super::{compact, tail_cut};
    use serde_json::{Value, json};
    #[test]
    fn preserves_system_and_recent_message() {
        let messages = vec![
            json!({"role":"system", "content":"policy"}),
            json!({"role":"user", "content":"old old old old old old old"}),
            json!({"role":"assistant", "content":"new"}),
        ];
        let compacted = compact(&messages, 80);
        assert_eq!(
            compacted
                .first()
                .and_then(|message| message.get("content"))
                .and_then(|content| content.as_str()),
            Some("policy")
        );
        assert_eq!(
            compacted
                .last()
                .and_then(|message| message.get("content"))
                .and_then(|content| content.as_str()),
            Some("new")
        );
    }

    #[test]
    fn compaction_never_orphans_tool_results() {
        let messages = vec![
            json!({"role":"system", "content":"sys"}),
            json!({"role":"user", "content":"please do a fairly long thing with lots of words"}),
            json!({"role":"assistant", "content":null, "tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}]}),
            json!({"role":"tool", "tool_call_id":"c1", "content":"long tool result padding xxxxxxxxxxxxxxxxxxxxxxx"}),
            json!({"role":"assistant", "content":"done"}),
        ];
        // Sweep budgets so every possible cut point is exercised.
        for budget in 1..600 {
            let compacted = compact(&messages, budget);
            for (index, message) in compacted.iter().enumerate() {
                if message.get("role").and_then(Value::as_str) == Some("tool") {
                    assert!(
                        compacted[..index]
                            .iter()
                            .any(|earlier| earlier.get("tool_calls").is_some()),
                        "budget {budget} produced a tool message without its tool_calls parent"
                    );
                }
            }
        }
    }

    #[test]
    fn tail_cut_keeps_a_small_verbatim_tail_out_of_a_large_history() {
        let messages = vec![
            json!({"role":"system", "content":"sys"}),
            json!({"role":"user", "content":"old old old old old old old"}),
            json!({"role":"assistant", "content":"old reply old reply old reply"}),
            json!({"role":"user", "content":"recent"}),
            json!({"role":"assistant", "content":"newest"}),
        ];
        // A budget too small for the whole history but big enough for the
        // newest message keeps just that.
        let cut = tail_cut(&messages, 1, 40);
        assert_eq!(cut, 4, "should cut right before the newest message");
        // A budget covering everything after the system message keeps it all.
        let cut = tail_cut(&messages, 1, 10_000);
        assert_eq!(cut, 1);
    }
}
