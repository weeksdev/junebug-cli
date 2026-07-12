//! Core types for Febo CLI. Public contracts live here so future TUI, CLI and
//! app-server interfaces all use the same provider and policy boundaries.

pub mod agent;
pub mod context;
pub mod editor;
pub mod hooks;
pub mod instructions;
pub mod markdown;
pub mod mcp;
pub mod policy;
pub mod provider;
pub mod session;
pub mod tool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    ReadOnly,
    Ask,
    WorkspaceWrite,
    /// Approve every write and command without prompting ("yolo"). Plan mode
    /// still overrides this to read-only.
    Yolo,
}

impl PermissionMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::Ask => "ask",
            Self::WorkspaceWrite => "workspace-write",
            Self::Yolo => "yolo",
        }
    }

    /// Parse a permission-mode name, accepting a couple of friendly aliases.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` is not a recognized mode.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "read-only" | "read" => Ok(Self::ReadOnly),
            "ask" => Ok(Self::Ask),
            "workspace-write" | "write" => Ok(Self::WorkspaceWrite),
            "yolo" | "approve-all" => Ok(Self::Yolo),
            _ => Err("permission must be read-only, ask, workspace-write, or yolo".to_owned()),
        }
    }
}
