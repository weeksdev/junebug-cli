//! Project instruction discovery for `AGENTS.md` files.

use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_INSTRUCTION_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionFile {
    pub path: PathBuf,
    pub contents: String,
}

/// Discover `AGENTS.md` files from the filesystem root through the workspace.
/// Parent instructions are returned first, allowing closer files to refine them.
///
/// # Errors
///
/// Returns an error when an instruction file cannot be read or exceeds the size limit.
pub fn discover(workspace: &Path) -> Result<Vec<InstructionFile>, String> {
    let mut directories = workspace.ancestors().collect::<Vec<_>>();
    directories.reverse();
    let mut files = Vec::new();
    for directory in directories {
        let path = directory.join("AGENTS.md");
        if !path.is_file() {
            continue;
        }
        let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
        if metadata.len() > MAX_INSTRUCTION_BYTES {
            return Err(format!(
                "{} exceeds the 64 KiB instruction limit",
                path.display()
            ));
        }
        files.push(InstructionFile {
            contents: fs::read_to_string(&path).map_err(|error| error.to_string())?,
            path,
        });
    }
    Ok(files)
}

/// Format discovered instructions as untrusted project guidance for the model.
#[must_use]
pub fn render(files: &[InstructionFile]) -> String {
    let mut output = String::new();
    for file in files {
        write!(
            output,
            "\n<project-instructions path=\"{}\">\n{}\n</project-instructions>",
            file.path.display(),
            file.contents
        )
        .expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::discover;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn finds_instruction_file() {
        let root = std::env::temp_dir().join(format!(
            "junebug-instructions-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("directory");
        fs::write(root.join("AGENTS.md"), "run tests").expect("instructions");
        let files = discover(&root).expect("discover");
        assert!(
            files
                .iter()
                .any(|file| file.path == root.join("AGENTS.md") && file.contents == "run tests")
        );
        fs::remove_dir_all(root).expect("cleanup");
    }
}
