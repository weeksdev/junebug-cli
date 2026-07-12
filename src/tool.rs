//! Workspace tools. The caller must enforce approval policy before invoking
//! any write capability.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRisk {
    Read,
    Write,
    Execute,
    Network,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: &'static str,
    pub risk: ToolRisk,
}

pub const BUILTIN_TOOLS: [ToolDefinition; 7] = [
    ToolDefinition {
        name: "list_dir",
        risk: ToolRisk::Read,
    },
    ToolDefinition {
        name: "read_file",
        risk: ToolRisk::Read,
    },
    ToolDefinition {
        name: "search",
        risk: ToolRisk::Read,
    },
    ToolDefinition {
        name: "write_file",
        risk: ToolRisk::Write,
    },
    ToolDefinition {
        name: "run_command",
        risk: ToolRisk::Execute,
    },
    ToolDefinition {
        name: "git_status",
        risk: ToolRisk::Read,
    },
    ToolDefinition {
        name: "git_diff",
        risk: ToolRisk::Read,
    },
];

#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            root: root.canonicalize().unwrap_or(root),
        }
    }

    fn checked_path(&self, requested: &Path) -> Result<PathBuf, String> {
        if requested.is_absolute()
            || requested
                .components()
                .any(|part| matches!(part, Component::ParentDir))
        {
            return Err(
                "path must be relative to the workspace and cannot contain '..'".to_owned(),
            );
        }
        // Component-based so `.git\config` on Windows and nested entries like
        // `vendor/.env` are protected too, not only `/`-separated root paths.
        for component in requested.components() {
            if let Component::Normal(name) = component {
                let name = name.to_string_lossy();
                if name == ".git" || name == ".febo" || name == ".env" || name.starts_with(".env.")
                {
                    return Err("path is protected by Febo policy".to_owned());
                }
            }
        }
        let candidate = self.root.join(requested);
        // An existing target (including a dangling symlink) is resolved in
        // full: that covers symlinks in the target itself and in any parent,
        // so a link inside the workspace can never alias content outside it.
        if candidate.symlink_metadata().is_ok() {
            let canonical = candidate
                .canonicalize()
                .map_err(|_| "path is a symbolic link that cannot be resolved".to_owned())?;
            if !canonical.starts_with(&self.root) {
                return Err("path escapes workspace through a symlink".to_owned());
            }
            return Ok(candidate);
        }
        // The target does not exist yet (a new file): resolve the nearest
        // existing ancestor instead. Note candidate.parent() operates on
        // normalized components, so for "." it would skip past the root —
        // that case cannot reach here because the root itself exists.
        let mut ancestor = candidate.parent().ok_or("workspace path has no parent")?;
        while !ancestor.exists() {
            ancestor = ancestor.parent().ok_or("workspace path has no parent")?;
        }
        let canonical_ancestor = ancestor.canonicalize().map_err(|error| error.to_string())?;
        if !canonical_ancestor.starts_with(&self.root) {
            return Err("path escapes workspace through a symlink".to_owned());
        }
        Ok(candidate)
    }

    /// # Errors
    ///
    /// Returns an error for protected/escaping paths, non-files, oversized files, or I/O failures.
    pub fn read_file(&self, requested: &Path) -> Result<String, String> {
        let path = self.checked_path(requested)?;
        let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
        if !metadata.is_file() {
            return Err("requested path is not a regular file".to_owned());
        }
        if metadata.len() > 256 * 1024 {
            return Err("requested file exceeds the 256 KiB read limit".to_owned());
        }
        fs::read_to_string(path).map_err(|error| error.to_string())
    }

    /// # Errors
    ///
    /// Returns an error when ripgrep cannot run or emits excessive output.
    pub fn search(&self, query: &str) -> Result<String, String> {
        if query.is_empty() {
            return Err("search query cannot be empty".to_owned());
        }
        let output = Command::new("rg")
            .args([
                "--line-number",
                "--max-count",
                "100",
                "--glob",
                "!.git",
                "--",
                query,
                ".",
            ])
            .current_dir(&self.root)
            .env_clear()
            .env("PATH", default_path())
            .output()
            .map_err(|error| error.to_string())?;
        let text = String::from_utf8_lossy(&output.stdout).into_owned();
        if text.len() > 256 * 1024 {
            return Err("search output exceeds the 256 KiB limit".to_owned());
        }
        if output.status.success() || output.status.code() == Some(1) {
            Ok(text)
        } else {
            Err(String::from_utf8_lossy(&output.stderr).into_owned())
        }
    }

    /// # Errors
    ///
    /// Returns an error for protected/escaping paths or directory I/O failures.
    pub fn list_dir(&self, requested: &Path) -> Result<Vec<String>, String> {
        let path = self.checked_path(requested)?;
        let mut entries = fs::read_dir(path)
            .map_err(|error| error.to_string())?
            .map(|entry| entry.map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        entries.sort_unstable();
        Ok(entries)
    }

    /// # Errors
    ///
    /// Returns an error for protected/escaping paths, symbolic links, oversized contents, or I/O failures.
    pub fn write_file(&self, requested: &Path, contents: &str) -> Result<(), String> {
        if contents.len() > 1024 * 1024 {
            return Err("write exceeds the 1 MiB content limit".to_owned());
        }
        let path = self.checked_path(requested)?;
        // symlink_metadata does not follow links, so this also refuses
        // dangling symlinks, which fs::write would otherwise follow.
        if let Ok(metadata) = path.symlink_metadata()
            && metadata.file_type().is_symlink()
        {
            return Err("refusing to overwrite a symbolic link".to_owned());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        fs::write(path, contents).map_err(|error| error.to_string())
    }

    /// # Errors
    ///
    /// Returns an error when the command cannot be started, fails, or emits more than 1 MiB of output.
    pub fn run_command(&self, command: &str) -> Result<String, String> {
        if command.trim().is_empty() {
            return Err("command cannot be empty".to_owned());
        }
        // Destructive-pattern classification happens in the approval layer
        // (`is_dangerous_command` adds a warning to the approval prompt);
        // by the time this runs the user has explicitly approved the exact
        // command text, and commands are never runnable without approval.
        let mut process = if cfg!(windows) {
            let mut process = Command::new("cmd");
            process.args(["/C", command]);
            process
        } else {
            let mut process = Command::new("/bin/sh");
            process.args(["-c", command]);
            process
        };
        process
            .current_dir(&self.root)
            .env_clear()
            .env("PATH", default_path());
        apply_windows_environment(&mut process);
        let output = process.output().map_err(|error| error.to_string())?;
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        if text.len() > 1024 * 1024 {
            return Err("command output exceeds the 1 MiB limit".to_owned());
        }
        if output.status.success() {
            Ok(text)
        } else {
            Err(format!("command exited with {}: {text}", output.status))
        }
    }

    /// # Errors
    ///
    /// Returns an error if Git cannot report workspace status.
    pub fn git_status(&self) -> Result<String, String> {
        self.run_git(["status", "--short"])
    }

    /// # Errors
    ///
    /// Returns an error if Git cannot produce a workspace diff.
    pub fn git_diff(&self) -> Result<String, String> {
        self.run_git(["diff", "--no-ext-diff"])
    }

    fn run_git<const N: usize>(&self, arguments: [&str; N]) -> Result<String, String> {
        let mut process = Command::new("git");
        process
            .arg("--no-pager")
            .args(arguments)
            .current_dir(&self.root)
            .env_clear()
            .env("PATH", default_path());
        apply_windows_environment(&mut process);
        let output = process.output().map_err(|error| error.to_string())?;
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        if output.status.success() {
            Ok(text)
        } else {
            Err(format!("git exited with {}: {text}", output.status))
        }
    }
}

const fn default_path() -> &'static str {
    if cfg!(windows) {
        "C:\\Windows\\System32"
    } else {
        "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    }
}

/// After `env_clear`, cmd.exe pipelines and many Windows programs fail
/// without `SystemRoot` and `ComSpec`; restore just those two.
pub(crate) fn apply_windows_environment(process: &mut Command) {
    if cfg!(windows)
        && let Ok(system_root) = std::env::var("SystemRoot")
    {
        process.env("ComSpec", format!("{system_root}\\System32\\cmd.exe"));
        process.env("SystemRoot", system_root);
    }
}

/// Detect `>`/`>>` redirection that targets a real file. Redirects to
/// `/dev/null` or to another descriptor (`2>&1`, `>&2`) cannot truncate
/// workspace files and are common in harmless commands.
fn has_file_redirection(command: &str) -> bool {
    let bytes = command.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'>' {
            let mut next = index + 1;
            if next < bytes.len() && bytes[next] == b'>' {
                next += 1;
            }
            while next < bytes.len() && bytes[next].is_ascii_whitespace() {
                next += 1;
            }
            let target = &command[next..];
            if !(target.starts_with('&') || target.starts_with("/dev/null")) {
                return true;
            }
            index = next.max(index + 1);
        } else {
            index += 1;
        }
    }
    false
}

/// Classify destructive/network command patterns. Matching commands are not
/// blocked outright: every command already requires an explicit interactive
/// approval, and this classification adds a visible warning to that prompt.
/// Matching is by whole command token (split on whitespace and shell
/// separators), so `git add .` is not flagged by `dd` while `rm<TAB>-rf`
/// and `echo hi;rm -rf /` still are. Shell expansion can always evade
/// static parsing; the approval prompt showing the exact text is the gate.
#[must_use]
pub fn is_dangerous_command(command: &str) -> bool {
    let normalized = command.to_ascii_lowercase();
    if has_file_redirection(&normalized) {
        return true;
    }
    let tokens = normalized
        .split(|character: char| {
            character.is_whitespace()
                || matches!(character, ';' | '|' | '&' | '(' | ')' | '<' | '`')
        })
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let blocked_word = |token: &str| {
        let basename = token.rsplit('/').next().unwrap_or(token);
        matches!(
            basename,
            "rm" | "sudo" | "dd" | "shutdown" | "reboot" | "curl" | "wget" | "nc" | "chmod"
        ) || basename == "mkfs"
            || basename.starts_with("mkfs.")
    };
    tokens.iter().any(|token| blocked_word(token))
        || tokens
            .windows(3)
            .any(|window| window == ["git", "reset", "--hard"])
        || tokens.windows(2).any(|window| window == ["git", "clean"])
}

#[cfg(test)]
mod tests {
    use super::Workspace;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_workspace() -> PathBuf {
        let name = format!(
            "febo-tool-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        );
        let path = std::env::temp_dir().join(name);
        fs::create_dir_all(&path).expect("temp workspace");
        path
    }

    #[test]
    fn traversal_is_rejected() {
        let workspace = Workspace::new(PathBuf::from("."));
        assert!(workspace.read_file(Path::new("../secret")).is_err());
    }

    #[test]
    fn protected_paths_are_rejected() {
        let workspace = Workspace::new(PathBuf::from("."));
        assert!(workspace.write_file(Path::new(".env"), "secret").is_err());
        assert!(
            workspace
                .write_file(Path::new(".git/config"), "nope")
                .is_err()
        );
    }

    #[test]
    fn writes_and_reads_regular_workspace_file() {
        let root = temporary_workspace();
        let workspace = Workspace::new(root.clone());
        workspace
            .write_file(Path::new("hello.txt"), "hello")
            .expect("write");
        assert_eq!(
            workspace.read_file(Path::new("hello.txt")).expect("read"),
            "hello"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn runs_non_destructive_command_in_workspace() {
        let root = temporary_workspace();
        let workspace = Workspace::new(root.clone());
        let command = if cfg!(windows) {
            "echo|set /p=febo"
        } else {
            "printf febo"
        };
        assert_eq!(workspace.run_command(command).expect("run"), "febo");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn dangerous_matcher_has_no_common_false_positives() {
        for command in [
            "git add .",
            "cargo fmt",
            "ls -la sudoku/",
            "echo confirm",
            "git diff --stat",
            "git log --oneline -5 2>/dev/null || echo none",
            "cargo test 2>&1",
            "echo warning >&2",
        ] {
            assert!(
                !super::is_dangerous_command(command),
                "{command} should not be flagged"
            );
        }
    }

    #[test]
    fn dangerous_matcher_catches_separator_and_path_variants() {
        for command in [
            "rm\t-rf x",
            "echo hi;rm -rf /",
            "true&&rm -rf /",
            "/bin/rm -rf /",
            "git  reset   --hard",
            "cat x > .bashrc",
            "echo data >> notes.txt",
            "mkfs.ext4 /dev/sda1",
        ] {
            assert!(
                super::is_dangerous_command(command),
                "{command} should be flagged"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn runs_command_with_dev_null_redirect() {
        let root = temporary_workspace();
        let workspace = Workspace::new(root.clone());
        assert_eq!(
            workspace
                .run_command("printf febo 2>/dev/null")
                .expect("run"),
            "febo"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn nested_protected_paths_are_rejected() {
        let workspace = Workspace::new(PathBuf::from("."));
        assert!(workspace.read_file(Path::new("vendor/.env")).is_err());
        assert!(
            workspace
                .write_file(Path::new("sub/.git/config"), "nope")
                .is_err()
        );
        assert!(
            workspace
                .read_file(Path::new(".febo/sessions/x.jsonl"))
                .is_err()
        );
    }

    #[test]
    fn list_dir_accepts_workspace_root_dot() {
        let root = temporary_workspace();
        fs::write(root.join("visible.txt"), "x").expect("file");
        let workspace = Workspace::new(root.clone());
        let entries = workspace.list_dir(Path::new(".")).expect("list root");
        assert!(entries.contains(&"visible.txt".to_owned()));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn write_creates_nested_directories() {
        let root = temporary_workspace();
        let workspace = Workspace::new(root.clone());
        workspace
            .write_file(Path::new("src/deep/module.rs"), "pub fn x() {}")
            .expect("nested write");
        assert_eq!(
            workspace
                .read_file(Path::new("src/deep/module.rs"))
                .expect("read"),
            "pub fn x() {}"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escaping_workspace_is_rejected_for_read_and_write() {
        let root = temporary_workspace();
        let outside = temporary_workspace();
        fs::write(outside.join("secret.txt"), "outside").expect("outside file");
        std::os::unix::fs::symlink(outside.join("secret.txt"), root.join("alias.txt"))
            .expect("symlink");
        std::os::unix::fs::symlink(outside.join("missing.txt"), root.join("dangling.txt"))
            .expect("dangling symlink");
        let workspace = Workspace::new(root.clone());
        assert!(
            workspace.read_file(Path::new("alias.txt")).is_err(),
            "reading through an escaping symlink must fail"
        );
        assert!(
            workspace.write_file(Path::new("alias.txt"), "x").is_err(),
            "writing through an escaping symlink must fail"
        );
        assert!(
            workspace
                .write_file(Path::new("dangling.txt"), "x")
                .is_err(),
            "writing through a dangling symlink must fail"
        );
        assert!(
            !outside.join("missing.txt").exists(),
            "no file may be created outside the workspace"
        );
        fs::remove_dir_all(root).expect("cleanup");
        fs::remove_dir_all(outside).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_within_workspace_is_still_readable() {
        let root = temporary_workspace();
        fs::write(root.join("real.txt"), "inside").expect("file");
        std::os::unix::fs::symlink(root.join("real.txt"), root.join("link.txt")).expect("symlink");
        let workspace = Workspace::new(root.clone());
        assert_eq!(
            workspace.read_file(Path::new("link.txt")).expect("read"),
            "inside"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }
}
