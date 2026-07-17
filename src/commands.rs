//! User-defined slash commands: Markdown prompt templates in
//! `~/.junebug/commands/` and the workspace's `.junebug/commands/`. Each
//! `<name>.md` becomes `/<name>`; its content is submitted as a regular
//! prompt, with `$ARGUMENTS` replaced by whatever follows the command.
//! Built-in commands always win a name collision.

use std::fs;
use std::path::{Path, PathBuf};

pub struct CustomCommand {
    pub name: String,
    /// First line of the template, shown in the completion menu.
    pub description: String,
    pub template: String,
}

/// Commands for `workspace`: user-level definitions overlaid by workspace
/// definitions on a name clash, sorted by name. Unreadable or oddly named
/// files are skipped rather than failing startup.
#[must_use]
pub fn load(workspace: &Path) -> Vec<CustomCommand> {
    let mut directories = Vec::new();
    if let Some(home) = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
        directories.push(PathBuf::from(home).join(".junebug").join("commands"));
    }
    directories.push(workspace.join(".junebug").join("commands"));
    load_from(&directories)
}

fn load_from(directories: &[PathBuf]) -> Vec<CustomCommand> {
    let mut commands: Vec<CustomCommand> = Vec::new();
    // Later directories override earlier ones, so the workspace wins.
    for directory in directories {
        let Ok(entries) = fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("md") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            let name = stem.to_ascii_lowercase();
            if name.is_empty()
                || !name
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || "-_".contains(character))
            {
                continue;
            }
            let Ok(template) = fs::read_to_string(&path) else {
                continue;
            };
            let template = template.trim().to_owned();
            if template.is_empty() {
                continue;
            }
            let description: String = template
                .lines()
                .next()
                .unwrap_or("")
                .trim_start_matches('#')
                .trim()
                .chars()
                .take(60)
                .collect();
            commands.retain(|existing| existing.name != name);
            commands.push(CustomCommand {
                name,
                description,
                template,
            });
        }
    }
    commands.sort_by(|left, right| left.name.cmp(&right.name));
    commands
}

/// The prompt a command submits: `$ARGUMENTS` replaced when the template
/// declares it, otherwise any arguments appended after the template.
#[must_use]
pub fn expand(template: &str, arguments: &str) -> String {
    if template.contains("$ARGUMENTS") {
        template.replace("$ARGUMENTS", arguments)
    } else if arguments.is_empty() {
        template.to_owned()
    } else {
        format!("{template}\n\n{arguments}")
    }
}

#[cfg(test)]
mod tests {
    use super::{expand, load_from};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "junebug-commands-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&path).expect("directory");
        path
    }

    #[test]
    fn workspace_commands_override_user_commands_and_bad_names_are_skipped() {
        let user = temp_dir("user");
        let workspace = temp_dir("workspace");
        fs::write(user.join("review.md"), "# User review\nDo a user review.").expect("user");
        fs::write(user.join("deploy.md"), "# Deploy\nShip it.").expect("user deploy");
        fs::write(
            workspace.join("review.md"),
            "# Team review\nUse the team checklist.",
        )
        .expect("workspace");
        fs::write(workspace.join("bad name!.md"), "nope").expect("bad name");
        fs::write(workspace.join("empty.md"), "   ").expect("empty");
        fs::write(workspace.join("notes.txt"), "not a command").expect("txt");

        let commands = load_from(&[user.clone(), workspace.clone()]);
        let names: Vec<&str> = commands.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["deploy", "review"]);
        let review = &commands[1];
        assert_eq!(review.description, "Team review");
        assert!(review.template.contains("team checklist"));

        fs::remove_dir_all(user).expect("cleanup");
        fs::remove_dir_all(workspace).expect("cleanup");
    }

    #[test]
    fn expansion_substitutes_or_appends_arguments() {
        assert_eq!(
            expand("Review $ARGUMENTS carefully.", "src/tool.rs"),
            "Review src/tool.rs carefully."
        );
        assert_eq!(expand("Fixed prompt.", ""), "Fixed prompt.");
        assert_eq!(
            expand("Fixed prompt.", "extra detail"),
            "Fixed prompt.\n\nextra detail"
        );
        assert_eq!(expand("A $ARGUMENTS B $ARGUMENTS", "x"), "A x B x");
    }
}
