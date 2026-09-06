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
/// when it alone exceeds the budget. Always lands on a genuine `user`
/// message (a real prompt, never a `tool` result or a bare `assistant`
/// reply with no visible antecedent): landing on `tool` orphans it from the
/// `assistant` `tool_calls` message that opened it, which every provider
/// rejects, and Anthropic's Messages API additionally requires the very
/// first message to have role `user` outright — landing on `assistant`
/// there produces exactly the same class of 400 (confirmed live against
/// Z.ai's Anthropic-compatible endpoint: "req contains no user message" /
/// "[1214] The messages parameter is illegal", immediately after a
/// mid-conversation compaction). `start` bounds the search from below —
/// e.g. past a leading system message the caller keeps separately. In
/// Junebug's canonical message shape a `user`-role entry is always a real
/// typed prompt (tool results are their own `tool` role, only folded into
/// Anthropic's `user`-with-`tool_result` shape by the provider-specific
/// translator), so this is always a safe restart point.
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
    while cut > start && messages[cut].get("role").and_then(Value::as_str) != Some("user") {
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
    fn compaction_never_starts_the_kept_suffix_on_a_non_user_message() {
        // Reproduces a live failure: Anthropic-format providers (real
        // Anthropic, and Z.ai's Anthropic-compatible endpoint) reject a
        // request whose first message isn't role `user`. The old `tail_cut`
        // only guarded against landing on `tool`, so a cut that landed on
        // an `assistant` message (with or without tool_calls) sailed
        // through and produced exactly that 400 on the very next turn.
        let messages = vec![
            json!({"role":"system", "content":"sys"}),
            json!({"role":"user", "content":"long ago user turn padding padding padding"}),
            json!({"role":"assistant", "content":"long ago reply padding padding padding"}),
            json!({"role":"user", "content":"look at this project"}),
            json!({"role":"assistant", "content":null, "tool_calls":[{"id":"c1","type":"function","function":{"name":"list_dir","arguments":"{\"path\":\".\"}"}}]}),
            json!({"role":"tool", "tool_call_id":"c1", "content":"a\nb\nc"}),
            json!({"role":"assistant", "content":"looked around"}),
            json!({"role":"assistant", "content":null, "tool_calls":[{"id":"c2","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a\"}"}}]}),
            json!({"role":"tool", "tool_call_id":"c2", "content":"contents of a"}),
        ];
        for budget in 1..1200 {
            let compacted = compact(&messages, budget);
            let first_kept = compacted
                .iter()
                .find(|message| message.get("role").and_then(Value::as_str) != Some("system"));
            if let Some(first_kept) = first_kept {
                assert_eq!(
                    first_kept.get("role").and_then(Value::as_str),
                    Some("user"),
                    "budget {budget} started the kept history on a non-user message: {first_kept:?}"
                );
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
        // newest message alone would cut right before it (index 4) — but
        // that message is an `assistant` reply, and a compacted history
        // must always restart on a genuine `user` message (see `tail_cut`),
        // so it keeps one more message back to the preceding `user` turn.
        let cut = tail_cut(&messages, 1, 40);
        assert_eq!(
            cut, 3,
            "should widen past a lone trailing assistant reply to the user turn before it"
        );
        // A budget covering everything after the system message keeps it all.
        let cut = tail_cut(&messages, 1, 10_000);
        assert_eq!(cut, 1);
    }
}
