//! Deterministic tool-approval decisions.
//!
//! This is the `PolicyEngine` contract from `PLAN.md`: given a tool's risk
//! class, the active permission mode, and whether the agent is in plan mode,
//! produce an `allow`/`deny`/`ask` decision. It must stay pure and must never
//! consult model output, tool arguments, or other untrusted signals — those
//! could otherwise be used to talk the policy into a looser decision.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::PermissionMode;
use crate::tool::ToolRisk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
    Ask,
}

/// Shared permission mode used by the REPL and an in-flight agent turn.
#[derive(Debug, Clone)]
pub struct PermissionState {
    value: Arc<AtomicU8>,
}

impl PermissionState {
    #[must_use]
    pub fn new(permission: PermissionMode) -> Self {
        Self {
            value: Arc::new(AtomicU8::new(permission_code(permission))),
        }
    }

    #[must_use]
    pub fn get(&self) -> PermissionMode {
        permission_from_code(self.value.load(Ordering::Relaxed))
    }

    pub fn set(&self, permission: PermissionMode) {
        self.value
            .store(permission_code(permission), Ordering::Relaxed);
    }

    /// Cycle read-only → ask → workspace-write → yolo → read-only.
    #[must_use]
    pub fn cycle(&self) -> PermissionMode {
        let next = match self.get() {
            PermissionMode::ReadOnly => PermissionMode::Ask,
            PermissionMode::Ask => PermissionMode::WorkspaceWrite,
            PermissionMode::WorkspaceWrite => PermissionMode::Yolo,
            PermissionMode::Yolo => PermissionMode::ReadOnly,
        };
        self.set(next);
        next
    }
}

const fn permission_code(permission: PermissionMode) -> u8 {
    match permission {
        PermissionMode::ReadOnly => 0,
        PermissionMode::Ask => 1,
        PermissionMode::WorkspaceWrite => 2,
        PermissionMode::Yolo => 3,
    }
}

const fn permission_from_code(code: u8) -> PermissionMode {
    match code {
        1 => PermissionMode::Ask,
        2 => PermissionMode::WorkspaceWrite,
        3 => PermissionMode::Yolo,
        _ => PermissionMode::ReadOnly,
    }
}

#[derive(Debug, Clone)]
pub struct PolicyEngine {
    permission: PermissionMode,
    shared_permission: Option<PermissionState>,
    plan_mode: bool,
}

impl PolicyEngine {
    #[must_use]
    pub const fn new(permission: PermissionMode, plan_mode: bool) -> Self {
        Self {
            permission,
            shared_permission: None,
            plan_mode,
        }
    }

    #[must_use]
    pub fn with_state(permission: PermissionState, plan_mode: bool) -> Self {
        Self {
            permission: permission.get(),
            shared_permission: Some(permission),
            plan_mode,
        }
    }

    #[must_use]
    pub fn permission(&self) -> PermissionMode {
        self.shared_permission
            .as_ref()
            .map_or(self.permission, PermissionState::get)
    }

    /// Freeze the currently effective permission for one tool-call boundary.
    #[must_use]
    pub fn snapshot(&self) -> Self {
        Self::new(self.permission(), self.plan_mode)
    }

    #[must_use]
    pub const fn plan_mode(&self) -> bool {
        self.plan_mode
    }

    /// Whether tools may leave the startup workspace, access normally
    /// protected paths, and inherit the launching environment. Plan mode
    /// always keeps the boundary intact.
    #[must_use]
    pub fn unrestricted_access(&self) -> bool {
        !self.plan_mode && matches!(self.permission(), PermissionMode::Yolo)
    }

    /// Decide how a tool call of the given risk class should be handled.
    ///
    /// Plan mode is a hard guard: it denies every write/execute/network risk
    /// regardless of the configured permission mode, independent of whichever
    /// tools happen to be offered to the model.
    #[must_use]
    pub fn evaluate(&self, risk: ToolRisk) -> Decision {
        let permission = self.permission();
        match risk {
            ToolRisk::Read => Decision::Allow,
            ToolRisk::Write => {
                if self.plan_mode {
                    return Decision::Deny;
                }
                match permission {
                    PermissionMode::ReadOnly => Decision::Deny,
                    PermissionMode::WorkspaceWrite | PermissionMode::Yolo => Decision::Allow,
                    PermissionMode::Ask => Decision::Ask,
                }
            }
            ToolRisk::Execute | ToolRisk::Network => {
                if self.plan_mode {
                    Decision::Deny
                } else if matches!(permission, PermissionMode::Yolo) {
                    // Yolo mode pre-approves commands and network access too.
                    Decision::Allow
                } else {
                    // Otherwise commands and network access always require an
                    // explicit approval; no other mode pre-approves them.
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
    use super::{Decision, PermissionState, PolicyEngine, parse_approval_answer};
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
            PermissionMode::Yolo,
        ] {
            let policy = PolicyEngine::new(permission, true);
            assert_eq!(policy.evaluate(ToolRisk::Write), Decision::Deny);
            assert_eq!(policy.evaluate(ToolRisk::Execute), Decision::Deny);
            assert_eq!(policy.evaluate(ToolRisk::Network), Decision::Deny);
            assert_eq!(policy.evaluate(ToolRisk::Read), Decision::Allow);
        }
    }

    #[test]
    fn yolo_pre_approves_writes_and_commands_but_not_in_plan_mode() {
        let policy = PolicyEngine::new(PermissionMode::Yolo, false);
        assert_eq!(policy.evaluate(ToolRisk::Write), Decision::Allow);
        assert_eq!(policy.evaluate(ToolRisk::Execute), Decision::Allow);
        assert_eq!(policy.evaluate(ToolRisk::Network), Decision::Allow);
        assert_eq!(policy.evaluate(ToolRisk::Read), Decision::Allow);
        assert!(policy.unrestricted_access());
        assert!(!PolicyEngine::new(PermissionMode::Yolo, true).unrestricted_access());
        assert!(!PolicyEngine::new(PermissionMode::WorkspaceWrite, false).unrestricted_access());
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

    #[test]
    fn shared_permission_cycles_and_changes_live_policy_decisions() {
        let state = PermissionState::new(PermissionMode::ReadOnly);
        let policy = PolicyEngine::with_state(state.clone(), false);
        assert_eq!(policy.evaluate(ToolRisk::Write), Decision::Deny);
        assert_eq!(state.cycle(), PermissionMode::Ask);
        assert_eq!(policy.evaluate(ToolRisk::Write), Decision::Ask);
        assert_eq!(state.cycle(), PermissionMode::WorkspaceWrite);
        assert_eq!(policy.evaluate(ToolRisk::Write), Decision::Allow);
        assert_eq!(state.cycle(), PermissionMode::Yolo);
        assert!(policy.unrestricted_access());
        assert_eq!(state.cycle(), PermissionMode::ReadOnly);
        assert_eq!(policy.evaluate(ToolRisk::Execute), Decision::Ask);
    }
}
