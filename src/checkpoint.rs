//! Workspace checkpoints in a shadow Git repository.
//!
//! Junebug snapshots the workspace before each prompt and before every mutating
//! tool call, without touching the user's own repository: the shadow repo
//! lives under `~/.junebug/checkpoints/<workspace-id>` and uses the workspace as
//! its work tree, so it works in non-Git workspaces too. `/rewind` restores
//! files from any checkpoint; the state just before a restore is checkpointed
//! first, so a restore is always undoable. Snapshots respect the workspace's
//! `.gitignore` plus a fixed exclude list, so secrets (`.env*`) and Junebug's own
//! state (`.junebug/`) are never captured or rewound.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Never snapshot or restore these. `.env*` keeps secrets out of the shadow
/// repo; current and legacy state directories keep sessions and hooks from
/// being rewound; `.git/` cannot be tracked anyway; build caches are skipped.
const EXCLUDES: &str = ".git/\n.junebug/\n.febo/\n.env\n.env.*\ntarget/\nnode_modules/\n";

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub tag: String,
    pub label: String,
    pub created: SystemTime,
}

pub struct Checkpointer {
    git_dir: PathBuf,
    workspace: PathBuf,
    /// `-c core.hooksPath=<absent dir>` so user-configured global Git hooks
    /// never run against the shadow repository.
    hooks_override: String,
}

impl Checkpointer {
    /// Open (or initialize) the shadow repository for `workspace` under the
    /// user's `~/.junebug/checkpoints` directory.
    ///
    /// # Errors
    ///
    /// Returns an error when no home directory is known, or Git is missing
    /// or fails to initialize the shadow repository.
    pub fn new(workspace: &Path) -> Result<Self, String> {
        let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
            .ok_or("cannot locate a home directory")?;
        // Canonicalize only for the identity hash so `.` and the absolute
        // path map to the same shadow repo; keep the original path for Git.
        let identity = workspace
            .canonicalize()
            .unwrap_or_else(|_| workspace.to_path_buf());
        let mut hasher = DefaultHasher::new();
        identity.hash(&mut hasher);
        let home = PathBuf::from(home);
        let suffix = format!("{:016x}", hasher.finish());
        let current = home.join(".junebug").join("checkpoints").join(&suffix);
        let legacy = home.join(".febo").join("checkpoints").join(suffix);
        // Existing workspaces keep using their old shadow repo so `/rewind`
        // history survives the product rename. New workspaces use Junebug.
        let git_dir = if current.join("HEAD").exists() || !legacy.join("HEAD").exists() {
            current
        } else {
            legacy
        };
        Self::with_git_dir(workspace.to_path_buf(), git_dir)
    }

    /// Like [`Checkpointer::new`] with an explicit shadow-repo location
    /// (used by tests).
    ///
    /// # Errors
    ///
    /// Returns an error when Git is missing or initialization fails.
    pub fn with_git_dir(workspace: PathBuf, git_dir: PathBuf) -> Result<Self, String> {
        let hooks_override = format!(
            "core.hooksPath={}",
            git_dir.join("hooks-disabled").display()
        );
        let checkpointer = Self {
            git_dir,
            workspace,
            hooks_override,
        };
        if !checkpointer.git_dir.join("HEAD").exists() {
            std::fs::create_dir_all(&checkpointer.git_dir).map_err(|error| error.to_string())?;
            checkpointer.git(&["init", "-q"])?;
        }
        let info = checkpointer.git_dir.join("info");
        std::fs::create_dir_all(&info).map_err(|error| error.to_string())?;
        std::fs::write(info.join("exclude"), EXCLUDES).map_err(|error| error.to_string())?;
        Ok(checkpointer)
    }

    /// Snapshot the current workspace state. Returns the new checkpoint tag,
    /// or `None` when nothing changed since the last checkpoint (the previous
    /// checkpoint already represents this state).
    ///
    /// # Errors
    ///
    /// Returns an error when Git fails.
    pub fn snapshot(&self, label: &str) -> Result<Option<String>, String> {
        self.git(&["add", "-A", "."])?;
        let has_head = self
            .git(&["rev-parse", "--verify", "--quiet", "HEAD"])
            .is_ok();
        if has_head && self.git_succeeds(&["diff", "--cached", "--quiet"])? {
            return Ok(None);
        }
        // --allow-empty gives an empty workspace a baseline to rewind to.
        let mut commit: Vec<&str> = vec!["commit", "-q", "-m", label];
        if !has_head {
            commit.push("--allow-empty");
        }
        self.git(&commit)?;
        let mut millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_millis();
        let mut last_error = String::new();
        for _ in 0..5 {
            let tag = format!("cp-{millis}");
            match self.git(&["tag", &tag]) {
                Ok(_) => return Ok(Some(tag)),
                Err(error) => {
                    last_error = error;
                    millis += 1;
                }
            }
        }
        Err(last_error)
    }

    /// All checkpoints, newest first.
    ///
    /// # Errors
    ///
    /// Returns an error when Git fails.
    pub fn list(&self) -> Result<Vec<Checkpoint>, String> {
        let output = self.git(&[
            "for-each-ref",
            "--format=%(refname:short)\t%(creatordate:unix)\t%(subject)",
            "refs/tags/cp-*",
        ])?;
        let mut checkpoints: Vec<Checkpoint> = output
            .lines()
            .filter_map(|line| {
                let mut parts = line.splitn(3, '\t');
                let tag = parts.next()?.to_owned();
                let seconds: u64 = parts.next()?.parse().ok()?;
                let label = parts.next().unwrap_or("").to_owned();
                Some(Checkpoint {
                    tag,
                    label,
                    created: UNIX_EPOCH + Duration::from_secs(seconds),
                })
            })
            .collect();
        // Tags encode creation milliseconds (`cp-<millis>`), so they order
        // reliably where Git's whole-second creatordate ties.
        checkpoints.sort_by(|a, b| (b.tag.len(), &b.tag).cmp(&(a.tag.len(), &a.tag)));
        Ok(checkpoints)
    }

    /// Restore workspace files to `tag`. The state just before restoring is
    /// checkpointed first, so the restore itself can be rewound. Excluded
    /// paths (`.env*`, `.junebug/`, ignored files) are never touched.
    ///
    /// # Errors
    ///
    /// Returns an error when the tag is unknown or Git fails.
    pub fn restore(&self, tag: &str) -> Result<(), String> {
        self.git(&[
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("{tag}^{{commit}}"),
        ])
        .map_err(|_| format!("unknown checkpoint: {tag}"))?;
        self.snapshot("before rewind restore")?;
        self.git(&["reset", "--hard", "-q", tag])?;
        Ok(())
    }

    /// Return machine-readable changes since the most recent checkpoint.
    /// This is the non-Git-workspace baseline used by the read-only
    /// `/changes` browser.
    ///
    /// # Errors
    ///
    /// Returns an error when the shadow Git repository cannot be inspected.
    pub fn changes_status(&self) -> Result<String, String> {
        self.git(&["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    }

    /// Return a per-file diff against the most recent checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error when the shadow Git repository cannot be inspected.
    pub fn diff_from_head(&self, path: &Path) -> Result<String, String> {
        self.git(&[
            "diff",
            "--no-ext-diff",
            "HEAD",
            "--",
            path.to_string_lossy().as_ref(),
        ])
    }

    /// Run git against the shadow repo with the workspace as the work tree.
    /// Identity and hooks are pinned per invocation so nothing depends on
    /// (or leaks into) the user's Git configuration.
    fn git(&self, args: &[&str]) -> Result<String, String> {
        let output = self.command(args)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "git {} failed: {}",
                args.first().unwrap_or(&""),
                stderr.trim()
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    /// Like `git`, but a non-zero exit is a normal answer (`false`), not an
    /// error — used for `diff --quiet` style status checks.
    fn git_succeeds(&self, args: &[&str]) -> Result<bool, String> {
        Ok(self.command(args)?.status.success())
    }

    fn command(&self, args: &[&str]) -> Result<std::process::Output, String> {
        Command::new("git")
            .arg("--git-dir")
            .arg(&self.git_dir)
            .arg("--work-tree")
            .arg(&self.workspace)
            .args([
                "-c",
                "user.name=Junebug",
                "-c",
                "user.email=junebug@localhost",
                "-c",
                "commit.gpgsign=false",
                "-c",
                self.hooks_override.as_str(),
            ])
            .args(args)
            .current_dir(&self.workspace)
            .output()
            .map_err(|error| format!("could not run git: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_checkpointer(label: &str) -> (PathBuf, Checkpointer) {
        let base = std::env::temp_dir().join(format!(
            "junebug-checkpoint-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let workspace = base.join("workspace");
        fs::create_dir_all(&workspace).expect("workspace");
        let checkpointer =
            Checkpointer::with_git_dir(workspace.clone(), base.join("shadow")).expect("init");
        (base, checkpointer)
    }

    #[test]
    fn snapshot_restore_roundtrip() {
        let (base, checkpointer) = temp_checkpointer("roundtrip");
        let workspace = base.join("workspace");
        fs::write(workspace.join("a.txt"), "one").expect("write");

        let first = checkpointer
            .snapshot("before prompt: test")
            .expect("snapshot")
            .expect("first snapshot creates a checkpoint");
        assert!(
            checkpointer
                .snapshot("unchanged")
                .expect("snapshot")
                .is_none(),
            "an unchanged workspace must not create a new checkpoint"
        );

        fs::write(workspace.join("a.txt"), "two").expect("modify");
        fs::write(workspace.join("b.txt"), "new").expect("create");
        checkpointer.restore(&first).expect("restore");

        assert_eq!(
            fs::read_to_string(workspace.join("a.txt")).expect("read"),
            "one"
        );
        assert!(
            !workspace.join("b.txt").exists(),
            "files created after the checkpoint must be removed on restore"
        );

        let checkpoints = checkpointer.list().expect("list");
        assert!(
            checkpoints.len() >= 2,
            "restore must checkpoint prior state"
        );
        assert_eq!(
            checkpoints[0].label, "before rewind restore",
            "newest first, and the pre-restore safety checkpoint exists"
        );
        assert!(checkpoints.iter().any(|c| c.tag == first));

        fs::remove_dir_all(base).expect("cleanup");
    }

    #[test]
    fn restore_is_undoable_via_the_safety_checkpoint() {
        let (base, checkpointer) = temp_checkpointer("undo-restore");
        let workspace = base.join("workspace");
        fs::write(workspace.join("a.txt"), "one").expect("write");
        let first = checkpointer.snapshot("s1").expect("snapshot").expect("tag");
        fs::write(workspace.join("a.txt"), "two").expect("modify");

        checkpointer.restore(&first).expect("restore");
        assert_eq!(
            fs::read_to_string(workspace.join("a.txt")).expect("read"),
            "one"
        );

        let safety = &checkpointer.list().expect("list")[0];
        checkpointer.restore(&safety.tag).expect("undo the restore");
        assert_eq!(
            fs::read_to_string(workspace.join("a.txt")).expect("read"),
            "two"
        );

        fs::remove_dir_all(base).expect("cleanup");
    }

    #[test]
    fn secrets_and_junebug_state_are_never_captured_or_restored() {
        let (base, checkpointer) = temp_checkpointer("excludes");
        let workspace = base.join("workspace");
        fs::write(workspace.join("a.txt"), "code").expect("write");
        fs::write(workspace.join(".env"), "SECRET=original").expect("env");
        fs::create_dir_all(workspace.join(".junebug/sessions")).expect("junebug dir");
        fs::write(workspace.join(".junebug/sessions/s.jsonl"), "log").expect("session");
        fs::create_dir_all(workspace.join(".febo/sessions")).expect("legacy dir");
        fs::write(workspace.join(".febo/sessions/s.jsonl"), "legacy").expect("legacy session");

        let first = checkpointer.snapshot("s1").expect("snapshot").expect("tag");
        let tracked = checkpointer
            .git(&["ls-tree", "-r", "--name-only", &first])
            .expect("ls");
        assert!(
            !tracked.contains(".env"),
            "secrets must not enter the shadow repo"
        );
        assert!(
            !tracked.contains(".junebug"),
            "junebug state must not enter the shadow repo"
        );
        assert!(
            !tracked.contains(".febo"),
            "legacy state must not enter the shadow repo"
        );

        fs::write(workspace.join(".env"), "SECRET=changed").expect("env");
        fs::write(workspace.join("a.txt"), "changed").expect("modify");
        checkpointer.restore(&first).expect("restore");

        assert_eq!(
            fs::read_to_string(workspace.join(".env")).expect("read"),
            "SECRET=changed",
            "restore must never touch .env"
        );
        assert_eq!(
            fs::read_to_string(workspace.join(".junebug/sessions/s.jsonl")).expect("read"),
            "log"
        );
        assert_eq!(
            fs::read_to_string(workspace.join(".febo/sessions/s.jsonl")).expect("read"),
            "legacy"
        );
        assert_eq!(
            fs::read_to_string(workspace.join("a.txt")).expect("read"),
            "code"
        );

        fs::remove_dir_all(base).expect("cleanup");
    }

    #[test]
    fn workspace_gitignore_is_respected() {
        let (base, checkpointer) = temp_checkpointer("gitignore");
        let workspace = base.join("workspace");
        fs::write(workspace.join(".gitignore"), "build/\n").expect("gitignore");
        fs::create_dir_all(workspace.join("build")).expect("build dir");
        fs::write(workspace.join("build/artifact.bin"), "big").expect("artifact");
        fs::write(workspace.join("a.txt"), "code").expect("write");

        let tag = checkpointer.snapshot("s1").expect("snapshot").expect("tag");
        let tracked = checkpointer
            .git(&["ls-tree", "-r", "--name-only", &tag])
            .expect("ls");
        assert!(tracked.contains("a.txt"));
        assert!(tracked.contains(".gitignore"));
        assert!(
            !tracked.contains("artifact.bin"),
            "ignored build output must not be snapshotted"
        );

        fs::remove_dir_all(base).expect("cleanup");
    }

    #[test]
    fn empty_workspace_gets_a_baseline_checkpoint() {
        let (base, checkpointer) = temp_checkpointer("empty");
        let tag = checkpointer.snapshot("baseline").expect("snapshot");
        assert!(tag.is_some(), "a fresh shadow repo must create a baseline");
        assert!(
            checkpointer.restore(&tag.expect("tag")).is_ok(),
            "restoring the baseline must work"
        );
        fs::remove_dir_all(base).expect("cleanup");
    }

    #[test]
    fn unknown_tag_is_rejected_without_side_effects() {
        let (base, checkpointer) = temp_checkpointer("unknown-tag");
        let workspace = base.join("workspace");
        fs::write(workspace.join("a.txt"), "one").expect("write");
        checkpointer.snapshot("s1").expect("snapshot");
        let before = checkpointer.list().expect("list").len();
        assert!(checkpointer.restore("cp-does-not-exist").is_err());
        assert_eq!(
            checkpointer.list().expect("list").len(),
            before,
            "a failed restore must not create checkpoints"
        );
        fs::remove_dir_all(base).expect("cleanup");
    }
}
