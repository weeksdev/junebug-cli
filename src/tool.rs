//! Workspace tools. The caller must enforce approval policy before invoking
//! any write capability.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Seconds a command may run when the model does not set `timeout_seconds`.
pub const DEFAULT_COMMAND_TIMEOUT_SECS: u64 = 120;
/// Largest `timeout_seconds` a model may request.
pub const MAX_COMMAND_TIMEOUT_SECS: u64 = 3600;

/// Combined stdout+stderr size beyond which a command fails.
const COMMAND_OUTPUT_CAP: usize = 1024 * 1024;

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

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn checked_path(&self, requested: &Path, unrestricted: bool) -> Result<PathBuf, String> {
        if unrestricted {
            return Ok(if requested.is_absolute() {
                requested.to_path_buf()
            } else {
                self.root.join(requested)
            });
        }
        if requested.is_absolute() {
            return Err(
                "absolute paths need yolo permission — use a path relative to the workspace"
                    .to_owned(),
            );
        }
        if requested
            .components()
            .any(|part| matches!(part, Component::ParentDir))
        {
            return Err("path cannot contain '..'".to_owned());
        }
        // Component-based so `.git\config` on Windows and nested entries like
        // `vendor/.env` are protected too, not only `/`-separated root paths.
        for component in requested.components() {
            if let Component::Normal(name) = component {
                let name = name.to_string_lossy();
                if name == ".git"
                    || name == ".junebug"
                    || name == ".febo"
                    || name == ".env"
                    || name.starts_with(".env.")
                {
                    return Err("path is protected by Junebug policy".to_owned());
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
        self.read_file_with_access(requested, false)
    }

    /// Read a file with optional yolo filesystem access.
    ///
    /// # Errors
    ///
    /// Returns an error for disallowed paths, non-files, oversized files, or I/O failures.
    pub fn read_file_with_access(
        &self,
        requested: &Path,
        unrestricted: bool,
    ) -> Result<String, String> {
        let path = self.checked_path(requested, unrestricted)?;
        // Name the failing path: the raw OS error ("No such file or
        // directory (os error 2)") does not say which path failed, and the
        // model needs that to correct a guessed filename.
        let metadata =
            fs::metadata(&path).map_err(|error| format!("{}: {error}", requested.display()))?;
        if metadata.is_dir() {
            return Err(
                "requested path is a directory, not a file — use list_dir to see its entries"
                    .to_owned(),
            );
        }
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
        self.search_at(query, Path::new("."), false)
    }

    /// Search under `requested`, which may be outside the startup workspace
    /// only when unrestricted yolo access is active.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid paths, ripgrep failures, or excessive output.
    pub fn search_at(
        &self,
        query: &str,
        requested: &Path,
        unrestricted: bool,
    ) -> Result<String, String> {
        if query.is_empty() {
            return Err("search query cannot be empty".to_owned());
        }
        let path = self.checked_path(requested, unrestricted)?;
        // Models regularly pass a file as the search path; search within it
        // rather than failing with rg's opaque "Not a directory" spawn error.
        let (working_dir, target) = if path.is_file() {
            let file = path.file_name().map_or_else(
                || ".".to_owned(),
                |name| name.to_string_lossy().into_owned(),
            );
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .map_or_else(|| self.root.clone(), Path::to_path_buf);
            (parent, file)
        } else {
            (path, ".".to_owned())
        };
        let output = Command::new("rg")
            .args([
                "--line-number",
                "--max-count",
                "100",
                "--glob",
                "!.git",
                "--",
                query,
                &target,
            ])
            .current_dir(working_dir)
            .env_clear()
            .env("PATH", default_path())
            .output()
            .map_err(|error| describe_rg_spawn_error(&error))?;
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
        self.list_dir_with_access(requested, false)
    }

    /// List a directory with optional yolo filesystem access.
    ///
    /// # Errors
    ///
    /// Returns an error for disallowed paths or directory I/O failures.
    pub fn list_dir_with_access(
        &self,
        requested: &Path,
        unrestricted: bool,
    ) -> Result<Vec<String>, String> {
        let path = self.checked_path(requested, unrestricted)?;
        let mut entries = fs::read_dir(path)
            .map_err(|error| format!("{}: {error}", requested.display()))?
            .map(|entry| entry.map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|entry| {
                let mut name = entry.file_name().to_string_lossy().into_owned();
                // Mark directories the way `ls -p` does; without the marker
                // the model cannot tell a directory from an extensionless
                // file and wastes turns calling read_file on directories.
                if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                    name.push('/');
                }
                name
            })
            .collect::<Vec<_>>();
        entries.sort_unstable();
        Ok(entries)
    }

    /// # Errors
    ///
    /// Returns an error for protected/escaping paths, symbolic links, oversized contents, or I/O failures.
    pub fn write_file(&self, requested: &Path, contents: &str) -> Result<(), String> {
        self.write_file_with_access(requested, contents, false)
    }

    /// Write a file with optional yolo filesystem access.
    ///
    /// # Errors
    ///
    /// Returns an error for disallowed paths, oversized contents, or I/O failures.
    pub fn write_file_with_access(
        &self,
        requested: &Path,
        contents: &str,
        unrestricted: bool,
    ) -> Result<(), String> {
        if contents.len() > 1024 * 1024 {
            return Err("write exceeds the 1 MiB content limit".to_owned());
        }
        let path = self.checked_path(requested, unrestricted)?;
        // symlink_metadata does not follow links, so this also refuses
        // dangling symlinks, which fs::write would otherwise follow.
        if !unrestricted
            && let Ok(metadata) = path.symlink_metadata()
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
    /// Returns an error when the command cannot be started, fails, emits more
    /// than 1 MiB of output, or runs past the default timeout.
    pub fn run_command(&self, command: &str) -> Result<String, String> {
        self.run_command_with_access(
            command,
            false,
            Duration::from_secs(DEFAULT_COMMAND_TIMEOUT_SECS),
        )
    }

    /// Run a command, inheriting the launching environment only for
    /// unrestricted yolo access. Other modes retain the sanitized environment.
    /// The command and every descendant are killed at `timeout`, so a
    /// long-running foreground process (e.g. a server) cannot hang the turn.
    ///
    /// # Errors
    ///
    /// Returns an error when the command cannot start, fails, emits excessive
    /// output, or exceeds `timeout`.
    pub fn run_command_with_access(
        &self,
        command: &str,
        unrestricted: bool,
        timeout: Duration,
    ) -> Result<String, String> {
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
        process.current_dir(&self.root);
        if !unrestricted {
            apply_sanitized_environment(&mut process);
        }
        process
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            // Lead a fresh process group so a timeout can kill the whole
            // command tree, not just the shell.
            use std::os::unix::process::CommandExt as _;
            process.process_group(0);
        }
        let mut child = process.spawn().map_err(|error| error.to_string())?;
        let stdout = drain_stream(child.stdout.take());
        let stderr = drain_stream(child.stderr.take());
        let started = Instant::now();
        let status = loop {
            match child.try_wait() {
                Err(error) => return Err(error.to_string()),
                Ok(Some(status)) => break status,
                Ok(None) => {}
            }
            if started.elapsed() >= timeout {
                kill_command_tree(&mut child);
                let _ = child.wait();
                let (partial, _) = collect_output(&stdout, &stderr);
                let tail = tail_chars(&partial, 2000);
                return Err(if tail.is_empty() {
                    format!(
                        "command timed out after {}s and was killed",
                        timeout.as_secs()
                    )
                } else {
                    format!(
                        "command timed out after {}s and was killed; output so far:\n{tail}",
                        timeout.as_secs()
                    )
                });
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        let (text, total) = collect_output(&stdout, &stderr);
        if total > COMMAND_OUTPUT_CAP {
            return Err("command output exceeds the 1 MiB limit".to_owned());
        }
        if status.success() {
            Ok(text)
        } else {
            Err(format!("command exited with {status}: {text}"))
        }
    }

    /// # Errors
    ///
    /// Returns an error if Git cannot report workspace status.
    pub fn git_status(&self) -> Result<String, String> {
        self.git_status_at(Path::new("."), false)
    }

    /// Git status at an explicit target directory.
    ///
    /// # Errors
    ///
    /// Returns an error for a disallowed path or Git failure.
    pub fn git_status_at(&self, requested: &Path, unrestricted: bool) -> Result<String, String> {
        self.run_git_or_explain_non_repository(
            ["status", "--short"],
            requested,
            unrestricted,
            "Git status",
        )
    }

    /// # Errors
    ///
    /// Returns an error if Git cannot produce a workspace diff.
    pub fn git_diff(&self) -> Result<String, String> {
        self.git_diff_at(Path::new("."), false)
    }

    /// Git diff at an explicit target directory.
    ///
    /// # Errors
    ///
    /// Returns an error for a disallowed path or Git failure.
    pub fn git_diff_at(&self, requested: &Path, unrestricted: bool) -> Result<String, String> {
        self.run_git_or_explain_non_repository(
            ["diff", "--no-ext-diff"],
            requested,
            unrestricted,
            "Git diff",
        )
    }

    fn run_git_or_explain_non_repository<const N: usize>(
        &self,
        arguments: [&str; N],
        requested: &Path,
        unrestricted: bool,
        operation: &str,
    ) -> Result<String, String> {
        let path = self.checked_path(requested, unrestricted)?;
        if !Self::is_git_work_tree(&path)? {
            return Ok(format!(
                "This directory is not a Git repository; {operation} is unavailable.\n"
            ));
        }
        Self::run_git_in(&path, &arguments)
    }

    fn is_git_work_tree(path: &Path) -> Result<bool, String> {
        let output = Self::git_command(path, &["rev-parse", "--is-inside-work-tree"])?;
        if output.status.success() {
            return Ok(String::from_utf8_lossy(&output.stdout).trim() == "true");
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("not a git repository") {
            Ok(false)
        } else {
            Err(format!("git repository check failed: {}", stderr.trim()))
        }
    }

    fn run_git_in(path: &Path, arguments: &[&str]) -> Result<String, String> {
        let output = Self::git_command(path, arguments)?;
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        if output.status.success() {
            Ok(text)
        } else {
            Err(format!("git exited with {}: {text}", output.status))
        }
    }

    fn git_command(path: &Path, arguments: &[&str]) -> Result<std::process::Output, String> {
        let mut process = Command::new("git");
        process.arg("--no-pager").args(arguments).current_dir(path);
        apply_sanitized_environment(&mut process);
        // Pinned after the allowlist passthrough so output stays parseable
        // regardless of the launching locale.
        process.env("LC_ALL", "C");
        process.output().map_err(|error| error.to_string())
    }
}

/// Output captured from one child stream. `total` keeps the true byte count
/// even after `bytes` stops growing at the cap, so the 1 MiB error is exact
/// while memory stays bounded.
struct StreamCapture {
    bytes: Vec<u8>,
    total: usize,
}

/// Captured stream state plus an EOF signal for timed joins.
struct DrainedStream {
    state: Arc<Mutex<StreamCapture>>,
    eof: mpsc::Receiver<()>,
}

/// Read a child stream to EOF on its own thread, storing at most the output
/// cap. The EOF channel lets the caller wait with a timeout: an orphaned
/// grandchild can hold the pipe open long after the shell exits, and the
/// command must not hang on it.
fn drain_stream(stream: Option<impl std::io::Read + Send + 'static>) -> DrainedStream {
    let state = Arc::new(Mutex::new(StreamCapture {
        bytes: Vec::new(),
        total: 0,
    }));
    let (eof_tx, eof) = mpsc::channel();
    if let Some(mut stream) = stream {
        let state = Arc::clone(&state);
        std::thread::spawn(move || {
            let mut chunk = [0_u8; 8192];
            loop {
                match stream.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(count) => {
                        if let Ok(mut capture) = state.lock() {
                            capture.total += count;
                            let room = (COMMAND_OUTPUT_CAP + 1).saturating_sub(capture.bytes.len());
                            capture.bytes.extend_from_slice(&chunk[..count.min(room)]);
                        }
                    }
                }
            }
            let _ = eof_tx.send(());
        });
    } else {
        let _ = eof_tx.send(());
    }
    DrainedStream { state, eof }
}

/// Concatenate captured stdout then stderr, waiting briefly for EOF on each
/// stream first. Returns the text plus the true combined byte count.
fn collect_output(stdout: &DrainedStream, stderr: &DrainedStream) -> (String, usize) {
    let mut text = String::new();
    let mut total = 0;
    for stream in [stdout, stderr] {
        let _ = stream.eof.recv_timeout(Duration::from_secs(2));
        if let Ok(capture) = stream.state.lock() {
            text.push_str(&String::from_utf8_lossy(&capture.bytes));
            total += capture.total;
        }
    }
    (text, total)
}

/// Kill the command and every descendant. On unix the child leads its own
/// process group (see `process_group(0)` at spawn), so the group is
/// signalled; on Windows `taskkill /T` walks the process tree.
fn kill_command_tree(child: &mut Child) {
    let id = child.id();
    if cfg!(windows) {
        let _ = Command::new("taskkill")
            .args(["/T", "/F", "/PID", &id.to_string()])
            .output();
    } else {
        let _ = Command::new("/bin/sh")
            .args(["-c", &format!("kill -9 -- -{id} 2>/dev/null")])
            .output();
    }
    let _ = child.kill();
}

/// The last `cap` characters of `text`, trimmed.
fn tail_chars(text: &str, cap: usize) -> String {
    let count = text.chars().count();
    let skipped: String = text.chars().skip(count.saturating_sub(cap)).collect();
    skipped.trim().to_owned()
}

/// Whether ripgrep is reachable on the sanitized PATH the search tool uses.
/// The REPL warns at startup when it is not, because every `search` call
/// will fail until it is installed.
#[must_use]
pub fn ripgrep_available() -> bool {
    ripgrep_available_on(default_path())
}

fn ripgrep_available_on(path: &str) -> bool {
    let binary = if cfg!(windows) { "rg.exe" } else { "rg" };
    std::env::split_paths(&std::ffi::OsString::from(path)).any(|dir| dir.join(binary).is_file())
}

/// A search failure the model (and user) can act on: a missing `rg` binary
/// otherwise surfaces as a bare "No such file or directory (os error 2)"
/// that looks like the searched path is at fault.
fn describe_rg_spawn_error(error: &std::io::Error) -> String {
    if error.kind() == std::io::ErrorKind::NotFound {
        "ripgrep (rg) is not installed or not on Junebug's sanitized PATH; the search tool \
         requires it (e.g. brew install ripgrep / apt install ripgrep)"
            .to_owned()
    } else {
        format!("cannot run rg: {error}")
    }
}

const fn default_path() -> &'static str {
    if cfg!(windows) {
        "C:\\Windows\\System32"
    } else {
        // Include the standard Apple Silicon Homebrew location. Read-only
        // subprocesses still get a fixed, sanitized PATH, but `rg` installed
        // by Homebrew must remain discoverable for the search tool.
        "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    }
}

/// Environment variables passed through to sanitized subprocesses.
/// Operational variables only: home/temp locations, locale, timezone,
/// proxies, the ssh agent socket, and toolchain homes — the settings common
/// developer commands (`git`, `cargo`, `npm`, `pip`, cloud CLIs) need to
/// behave normally. API keys and other secrets in the launching environment
/// are deliberately absent: nothing outside this list survives `env_clear`,
/// so they can never reach an approved command, the model, or a session log.
const SANITIZED_ENV_ALLOWLIST: &[&str] = &[
    "HOME",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "TZ",
    "TERM",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR",
    // The agent socket grants key *use*, not key material; every command
    // already requires an explicit interactive approval before it runs.
    "SSH_AUTH_SOCK",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "GOPATH",
    "GOROOT",
    "JAVA_HOME",
    // Windows equivalents; absent elsewhere, so passing them is a no-op.
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "ALLUSERSPROFILE",
    "TEMP",
    "TMP",
    "PATHEXT",
    "SystemDrive",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramW6432",
    "CommonProgramFiles",
    "USERNAME",
    "windir",
];

/// Clear the environment, then rebuild it from the fixed sanitized `PATH`
/// and the operational-variable allowlist. Used for every non-yolo
/// subprocess so approved commands work normally while environment secrets
/// stay out of reach.
pub(crate) fn apply_sanitized_environment(process: &mut Command) {
    process.env_clear().env("PATH", sanitized_path());
    for name in SANITIZED_ENV_ALLOWLIST {
        if let Some(value) = std::env::var_os(name) {
            process.env(name, value);
        }
    }
    apply_windows_environment(process);
}

/// The fixed sanitized `PATH`, extended with the user's `~/.cargo/bin` when
/// a home directory is known: rustup installs toolchains only there, and a
/// `cargo` the agent cannot find defeats the point of preserving its
/// environment.
fn sanitized_path() -> std::ffi::OsString {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" });
    let Some(home) = home else {
        return default_path().into();
    };
    let cargo_bin = Path::new(&home).join(".cargo").join("bin");
    let base = std::ffi::OsString::from(default_path());
    std::env::join_paths(std::env::split_paths(&base).chain([cargo_bin]))
        .unwrap_or_else(|_| default_path().into())
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
    #[cfg(unix)]
    use super::DEFAULT_COMMAND_TIMEOUT_SECS;
    use super::Workspace;
    use std::fs;
    use std::path::{Path, PathBuf};
    #[cfg(unix)]
    use std::time::{Duration, Instant};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_workspace() -> PathBuf {
        let name = format!(
            "junebug-tool-test-{}",
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
    fn unrestricted_access_can_read_protected_and_outside_paths() {
        let root = temporary_workspace();
        let outside = temporary_workspace();
        fs::write(root.join(".env"), "LOCAL_SECRET=test-only").expect("seed protected file");
        fs::write(outside.join("outside.txt"), "outside").expect("seed outside file");
        let workspace = Workspace::new(root.clone());

        assert!(workspace.read_file(Path::new(".env")).is_err());
        assert_eq!(
            workspace
                .read_file_with_access(Path::new(".env"), true)
                .expect("yolo protected read"),
            "LOCAL_SECRET=test-only"
        );
        assert_eq!(
            workspace
                .read_file_with_access(&outside.join("outside.txt"), true)
                .expect("yolo outside read"),
            "outside"
        );
        workspace
            .write_file_with_access(&outside.join("created.txt"), "created", true)
            .expect("yolo outside write");
        assert_eq!(
            fs::read_to_string(outside.join("created.txt")).expect("outside write exists"),
            "created"
        );

        fs::remove_dir_all(root).expect("cleanup root");
        fs::remove_dir_all(outside).expect("cleanup outside");
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
        // `echo` emits a trailing newline (\r\n on Windows); trim it rather
        // than using `set /p`, which returns errorlevel 1 at EOF on Windows.
        assert_eq!(
            workspace
                .run_command("echo junebug")
                .expect("run")
                .trim_end(),
            "junebug"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn timed_out_command_is_killed_with_its_children_and_reports_partial_output() {
        let root = temporary_workspace();
        let workspace = Workspace::new(root.clone());
        let started = Instant::now();
        // The `sleep` is a grandchild of the spawned shell once the compound
        // command forks; the process-group kill must take it down too.
        let error = workspace
            .run_command_with_access("echo started; sleep 30", false, Duration::from_secs(1))
            .expect_err("must time out");
        assert!(started.elapsed() < Duration::from_secs(10), "kill was slow");
        assert!(error.contains("timed out after 1s"), "got: {error}");
        assert!(error.contains("started"), "partial output missing: {error}");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn command_returns_when_a_background_child_holds_the_output_pipe() {
        let root = temporary_workspace();
        let workspace = Workspace::new(root.clone());
        let started = Instant::now();
        // The shell exits immediately while `sleep` keeps the inherited
        // stdout pipe open; the old `Command::output` path blocked on it
        // until the background child exited.
        let output = workspace
            .run_command("sleep 30 & echo done")
            .expect("command must not hang on the orphaned pipe");
        assert!(started.elapsed() < Duration::from_secs(10), "join was slow");
        assert_eq!(output.trim_end(), "done");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn git_tools_explain_when_workspace_is_not_a_repository() {
        let root = temporary_workspace();
        let workspace = Workspace::new(root.clone());

        let status = workspace.git_status().expect("non-repo status is graceful");
        let diff = workspace.git_diff().expect("non-repo diff is graceful");

        assert!(status.contains("not a Git repository"), "got: {status}");
        assert!(diff.contains("not a Git repository"), "got: {diff}");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn search_can_find_ripgrep_in_homebrew_path() {
        if !Path::new("/opt/homebrew/bin/rg").is_file() {
            return;
        }
        let root = temporary_workspace();
        fs::write(root.join("needle.txt"), "homebrew-rg-marker").expect("seed search file");
        let workspace = Workspace::new(root.clone());
        let result = workspace
            .search("homebrew-rg-marker")
            .expect("search with sanitized PATH");
        assert!(result.contains("needle.txt"), "got: {result}");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn search_with_a_file_path_searches_within_that_file() {
        if !Path::new("/opt/homebrew/bin/rg").is_file() && !Path::new("/usr/bin/rg").is_file() {
            return;
        }
        let root = temporary_workspace();
        fs::create_dir(root.join("src")).expect("dir");
        fs::write(root.join("src/main.py"), "in_target_file_marker").expect("target");
        fs::write(root.join("src/other.py"), "in_target_file_marker").expect("other");
        let workspace = Workspace::new(root.clone());
        let result = workspace
            .search_at("in_target_file_marker", Path::new("src/main.py"), false)
            .expect("file-path search");
        assert!(result.contains("in_target_file_marker"), "got: {result}");
        assert!(
            !result.contains("other.py"),
            "must not search siblings: {result}"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn sanitized_commands_keep_operational_variables_but_not_the_rest() {
        let root = temporary_workspace();
        let workspace = Workspace::new(root.clone());
        // HOME is on the allowlist: git identity, cargo, and npm need it.
        assert_eq!(
            workspace
                .run_command("printf %s \"${HOME-unset}\"")
                .expect("sanitized command"),
            std::env::var("HOME").expect("test runs with HOME set")
        );
        // CARGO_PKG_NAME is set by the cargo test harness but is not on the
        // allowlist; anything off the list must never reach the child.
        assert_eq!(
            workspace
                .run_command("printf %s \"${CARGO_PKG_NAME-unset}\"")
                .expect("sanitized command"),
            "unset"
        );
        assert_eq!(
            workspace
                .run_command_with_access(
                    "printf %s \"${CARGO_PKG_NAME-unset}\"",
                    true,
                    Duration::from_secs(DEFAULT_COMMAND_TIMEOUT_SECS),
                )
                .expect("yolo command"),
            "junebug-cli",
            "yolo commands inherit the full launching environment"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn sanitized_git_sees_the_users_global_configuration() {
        let root = temporary_workspace();
        let workspace = Workspace::new(root.clone());
        // `git config --global` resolves through HOME; before the allowlist
        // this failed or reported nothing under an emptied environment.
        // The command must at least run without an environment error even
        // when the host has no global config (empty output, exit 0 or 1).
        let result = workspace.run_command("git config --global --get user.name || true");
        assert!(result.is_ok(), "got: {result:?}");
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
                .run_command("printf junebug 2>/dev/null")
                .expect("run"),
            "junebug"
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
                .read_file(Path::new(".junebug/sessions/x.jsonl"))
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
    fn ripgrep_detection_follows_the_given_path() {
        let root = temporary_workspace();
        assert!(!super::ripgrep_available_on(&root.to_string_lossy()));
        let binary = if cfg!(windows) { "rg.exe" } else { "rg" };
        fs::write(root.join(binary), "stub").expect("stub rg");
        assert!(super::ripgrep_available_on(&root.to_string_lossy()));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn missing_paths_and_missing_ripgrep_produce_actionable_errors() {
        let root = temporary_workspace();
        let workspace = Workspace::new(root.clone());
        let error = workspace
            .read_file(Path::new("requirements-dev.txt"))
            .expect_err("missing file");
        assert!(
            error.contains("requirements-dev.txt"),
            "error must name the path: {error}"
        );
        let error = workspace
            .search_at("query", Path::new("/outside"), false)
            .expect_err("absolute path");
        assert!(error.contains("yolo"), "got: {error}");
        assert!(!error.contains(".."), "'..' is not the problem: {error}");
        let not_found = std::io::Error::from(std::io::ErrorKind::NotFound);
        let described = super::describe_rg_spawn_error(&not_found);
        assert!(described.contains("ripgrep"), "got: {described}");
        assert!(described.contains("install"), "got: {described}");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn list_dir_marks_directories_and_read_file_explains_them() {
        let root = temporary_workspace();
        fs::create_dir(root.join("tests")).expect("dir");
        fs::write(root.join("tests.rs"), "x").expect("file");
        let workspace = Workspace::new(root.clone());
        let entries = workspace.list_dir(Path::new(".")).expect("list root");
        assert!(entries.contains(&"tests/".to_owned()), "got: {entries:?}");
        assert!(entries.contains(&"tests.rs".to_owned()), "got: {entries:?}");
        let error = workspace
            .read_file(Path::new("tests"))
            .expect_err("directories are not readable");
        assert!(error.contains("directory"), "got: {error}");
        assert!(error.contains("list_dir"), "got: {error}");
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
