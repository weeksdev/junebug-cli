use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::router::{Band, Route};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub routing: RoutingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RoutingConfig {
    #[serde(default)]
    pub mode: RoutingMode,
    #[serde(default = "default_api_url")]
    pub api_url: String,
    #[serde(default)]
    pub send_prompt: bool,
    #[serde(default)]
    pub routes: BTreeMap<Band, Route>,
    #[serde(default)]
    pub min_band: Option<Band>,
    #[serde(default)]
    pub max_band: Option<Band>,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            mode: RoutingMode::Off,
            api_url: default_api_url(),
            send_prompt: false,
            routes: BTreeMap::new(),
            min_band: None,
            max_band: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RoutingMode {
    #[default]
    Off,
    Auto,
}

fn default_api_url() -> String {
    "http://127.0.0.1:8791".to_owned()
}

/// Load user config, then overlay a workspace config when present.
///
/// # Errors
/// Returns malformed JSON or file I/O errors (missing files are ignored).
pub fn load(workspace: &Path) -> Result<Config, String> {
    let mut config = Config::default();
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        merge_file(&mut config, &home.join(".febo/config.json"))?;
        merge_file(&mut config, &home.join(".junebug/config.json"))?;
    }
    merge_file(&mut config, &workspace.join(".febo/config.json"))?;
    merge_file(&mut config, &workspace.join(".junebug/config.json"))?;
    Ok(config)
}

fn merge_file(config: &mut Config, path: &Path) -> Result<(), String> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let overlay: Config =
        serde_json::from_str(&contents).map_err(|error| format!("{}: {error}", path.display()))?;
    config.routing = overlay.routing;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn defaults_are_opt_out() {
        let config: Config = serde_json::from_str("{}").expect("config");
        assert_eq!(config.routing.mode, RoutingMode::Off);
        assert!(!config.routing.send_prompt);
        assert_eq!(config.routing.api_url, "http://127.0.0.1:8791");
    }

    #[test]
    fn workspace_legacy_config_is_loaded_and_current_config_wins() {
        let root = std::env::temp_dir().join(format!(
            "junebug-config-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join(".febo")).expect("legacy directory");
        std::fs::write(
            root.join(".febo/config.json"),
            r#"{"routing":{"mode":"auto"}}"#,
        )
        .expect("legacy config");
        assert_eq!(
            load(&root).expect("legacy load").routing.mode,
            RoutingMode::Auto
        );

        std::fs::create_dir_all(root.join(".junebug")).expect("current directory");
        std::fs::write(root.join(".junebug/config.json"), r#"{"routing":{}}"#)
            .expect("current config");
        assert_eq!(
            load(&root).expect("current load").routing.mode,
            RoutingMode::Off,
            "current Junebug config must override legacy Febo config"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
