//! Deterministic tool-approval decisions.
//!
//! This is the `PolicyEngine` contract from `PLAN.md`: given a tool's risk
//! class, the active permission mode, and whether the agent is in plan mode,
//! produce an `allow`/`deny`/`ask` decision. It must stay pure and must never
//! consult model output, tool arguments, or other untrusted signals — those
//! could otherwise be used to talk the policy into a looser decision.

use crate::PermissionMode;
use crate::tool::ToolRisk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone, Copy)]
pub struct PolicyEngine {
    permission: PermissionMode,
    plan_mode: bool,
}

impl PolicyEngine {
    #[must_use]
    pub const fn new(permission: PermissionMode, plan_mode: bool) -> Self {
        Self {
            permission,
            plan_mode,
        }
    }

    /// Decide how a tool call of the given risk class should be handled.
    ///
    /// Plan mode is a hard guard: it denies every write/execute/network risk
    /// regardless of the configured permission mode, independent of whichever
    /// tools happen to be offered to the model.
    #[must_use]
    pub const fn evaluate(&self, risk: ToolRisk) -> Decision {
        match risk {
            ToolRisk::Read => Decision::Allow,
            ToolRisk::Write => {
                if self.plan_mode {
                    return Decision::Deny;
                }
                match self.permission {
                    PermissionMode::ReadOnly => Decision::Deny,
                    PermissionMode::WorkspaceWrite => Decision::Allow,
                    PermissionMode::Ask => Decision::Ask,
                }
            }
            ToolRisk::Execute | ToolRisk::Network => {
                if self.plan_mode {
                    Decision::Deny
                } else {
                    // Commands and network access always require an explicit
                    // approval; no permission mode pre-approves them.
                    Decision::Ask
                }
            }
        }
    }
}

/// Parse an interactive approval answer. Only an exact, case-insensitive
/// affirmative approves; anything else (including empty input or a read
/// failure represented as an empty string) denies.
#[must_use]
pub fn parse_approval_answer(answer: &str) -> bool {
    matches!(answer.trim(), "y" | "Y" | "yes" | "YES")
}

#[cfg(test)]
mod tests {
    use super::{Decision, PolicyEngine, parse_approval_answer};
    use crate::PermissionMode;
    use crate::tool::ToolRisk;

    #[test]
    fn read_is_always_allowed() {
        for permission in [
            PermissionMode::ReadOnly,
            PermissionMode::Ask,
            PermissionMode::WorkspaceWrite,
        ] {
            for plan_mode in [false, true] {
                let policy = PolicyEngine::new(permission, plan_mode);
                assert_eq!(policy.evaluate(ToolRisk::Read), Decision::Allow);
            }
        }
    }

    #[test]
    fn write_follows_permission_mode() {
        assert_eq!(
            PolicyEngine::new(PermissionMode::ReadOnly, false).evaluate(ToolRisk::Write),
            Decision::Deny
        );
        assert_eq!(
            PolicyEngine::new(PermissionMode::Ask, false).evaluate(ToolRisk::Write),
            Decision::Ask
        );
        assert_eq!(
            PolicyEngine::new(PermissionMode::WorkspaceWrite, false).evaluate(ToolRisk::Write),
            Decision::Allow
        );
    }

    #[test]
    fn execute_and_network_always_ask_outside_plan_mode() {
        for permission in [
            PermissionMode::ReadOnly,
            PermissionMode::Ask,
            PermissionMode::WorkspaceWrite,
        ] {
            let policy = PolicyEngine::new(permission, false);
            assert_eq!(policy.evaluate(ToolRisk::Execute), Decision::Ask);
            assert_eq!(policy.evaluate(ToolRisk::Network), Decision::Ask);
        }
    }

    #[test]
    fn plan_mode_denies_every_write_execute_and_network_risk() {
        for permission in [
            PermissionMode::ReadOnly,
            PermissionMode::Ask,
            PermissionMode::WorkspaceWrite,
        ] {
            let policy = PolicyEngine::new(permission, true);
            assert_eq!(policy.evaluate(ToolRisk::Write), Decision::Deny);
            assert_eq!(policy.evaluate(ToolRisk::Execute), Decision::Deny);
            assert_eq!(policy.evaluate(ToolRisk::Network), Decision::Deny);
            assert_eq!(policy.evaluate(ToolRisk::Read), Decision::Allow);
        }
    }

    #[test]
    fn approval_answer_requires_exact_affirmative() {
        assert!(parse_approval_answer("y"));
        assert!(parse_approval_answer("yes\n"));
        assert!(parse_approval_answer("  YES  "));
        assert!(!parse_approval_answer("n"));
        assert!(!parse_approval_answer(""));
        assert!(!parse_approval_answer("sure"));
    }
}
