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
}

impl PermissionMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::Ask => "ask",
            Self::WorkspaceWrite => "workspace-write",
        }
    }
}
