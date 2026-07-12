//! Minimal local stdio MCP client. Remote HTTP/OAuth is intentionally separate.

use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
}

/// # Errors
///
/// Returns an error when `.febo/mcp.json` cannot be read or validated.
pub fn load_config(workspace: &Path) -> Result<Vec<ServerConfig>, String> {
    let path = workspace.join(".febo").join("mcp.json");
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let config: Value =
        serde_json::from_str(&fs::read_to_string(path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let servers = config
        .get("servers")
        .and_then(Value::as_object)
        .ok_or("mcp.json requires a servers object")?;
    servers
        .iter()
        .map(|(name, server)| {
            let command = server
                .get("command")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("MCP server {name} requires command"))?
                .to_owned();
            let args = match server.get("args") {
                None => Vec::new(),
                Some(args) => args
                    .as_array()
                    .ok_or_else(|| format!("MCP server {name} args must be an array"))?
                    .iter()
                    .map(|arg| {
                        arg.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| format!("MCP server {name} args must be strings"))
                    })
                    .collect::<Result<_, _>>()?,
            };
            Ok(ServerConfig {
                name: name.to_owned(),
                command,
                args,
            })
        })
        .collect()
}

pub struct Client {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Client {
    /// # Errors
    ///
    /// Returns an error when the server cannot start or rejects initialization.
    pub fn connect(config: &ServerConfig) -> Result<Self, String> {
        let mut child = Command::new(&config.command)
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("could not start MCP server {}: {error}", config.name))?;
        let stdin = BufWriter::new(child.stdin.take().ok_or("MCP stdin unavailable")?);
        let stdout = BufReader::new(child.stdout.take().ok_or("MCP stdout unavailable")?);
        let mut client = Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        };
        let initialize = json!({"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"febo-cli","version":env!("CARGO_PKG_VERSION")}});
        client.request("initialize", &initialize)?;
        client.notify("notifications/initialized", &json!({}))?;
        Ok(client)
    }
    /// # Errors
    /// Returns an error when the server request fails or returns malformed JSON-RPC.
    pub fn request(&mut self, method: &str, params: &Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        writeln!(
            self.stdin,
            "{}",
            json!({"jsonrpc":"2.0","id":id,"method":method,"params":params})
        )
        .map_err(|error| error.to_string())?;
        self.stdin.flush().map_err(|error| error.to_string())?;
        let mut line = String::new();
        loop {
            line.clear();
            if self
                .stdout
                .read_line(&mut line)
                .map_err(|error| error.to_string())?
                == 0
            {
                return Err("MCP server closed stdout".to_owned());
            }
            let response: Value = serde_json::from_str(line.trim())
                .map_err(|error| format!("invalid MCP response: {error}"))?;
            if response.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = response.get("error") {
                return Err(format!("MCP {method} failed: {error}"));
            }
            return response
                .get("result")
                .cloned()
                .ok_or("MCP response lacks result".to_owned());
        }
    }
    fn notify(&mut self, method: &str, params: &Value) -> Result<(), String> {
        writeln!(
            self.stdin,
            "{}",
            json!({"jsonrpc":"2.0","method":method,"params":params})
        )
        .map_err(|error| error.to_string())?;
        self.stdin.flush().map_err(|error| error.to_string())
    }
    /// # Errors
    /// Returns an error if the server does not list valid tools.
    pub fn tools(&mut self) -> Result<Vec<Value>, String> {
        Ok(self
            .request("tools/list", &json!({}))?
            .get("tools")
            .and_then(Value::as_array)
            .ok_or("MCP tools/list lacks tools")?
            .clone())
    }
    /// # Errors
    /// Returns an error if the tool call fails.
    pub fn call(&mut self, name: &str, arguments: &Value) -> Result<Value, String> {
        self.request("tools/call", &json!({"name":name,"arguments":arguments}))
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Convert an MCP tool to an OpenAI-style, namespaced function definition.
#[must_use]
pub fn provider_tool(server: &str, tool: &Value) -> Option<Value> {
    let name = tool.get("name")?.as_str()?;
    Some(
        json!({"type":"function","function":{"name":format!("mcp__{server}__{name}"),"description":tool.get("description").and_then(Value::as_str).unwrap_or("MCP tool"),"parameters":tool.get("inputSchema").cloned().unwrap_or_else(|| json!({"type":"object","properties":{}}))}}),
    )
}

#[cfg(test)]
mod tests {
    use super::{load_config, provider_tool};
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};
    #[test]
    fn loads_config_and_namespaces_tool() {
        let root = std::env::temp_dir().join(format!(
            "febo-mcp-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(root.join(".febo")).expect("dir");
        fs::write(
            root.join(".febo/mcp.json"),
            r#"{"servers":{"docs":{"command":"echo","args":["hi"]}}}"#,
        )
        .expect("config");
        assert_eq!(load_config(&root).expect("load")[0].name, "docs");
        assert_eq!(
            provider_tool(
                "docs",
                &json!({"name":"lookup","inputSchema":{"type":"object"}})
            )
            .expect("tool")["function"]["name"],
            "mcp__docs__lookup"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }
}
