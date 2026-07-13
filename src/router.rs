use std::collections::BTreeMap;
use std::time::Duration;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::agent::{ModelSource, Selection, TurnState};
use crate::config::RoutingConfig;
use crate::provider::{ModelProvider, ProviderKind, ProviderRegistry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Band {
    Simple,
    Standard,
    Complex,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Route {
    pub provider: String,
    pub model: String,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Indicators {
    pub security: bool,
    pub concurrency: bool,
    pub migration: bool,
    pub architecture: bool,
    pub refactor: bool,
    pub trivial: bool,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct TaskSignals {
    pub prompt_chars: usize,
    pub attached_file_count: usize,
    pub attached_content_chars: usize,
    pub conversation_message_count: usize,
    pub plan_mode: bool,
    pub languages: Vec<String>,
    pub indicators: Indicators,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Task {
    pub signals: TaskSignals,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Execution {
    pub turn_index: usize,
    pub turns_remaining: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    pub consecutive_tool_failures: usize,
    pub tests_failing: bool,
    pub provider_error: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_band: Option<Band>,
    pub turns_on_route: usize,
    pub switches_so_far: usize,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Preferences {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_band: Option<Band>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_band: Option<Band>,
    pub minimum_turns_on_route: usize,
    pub max_switches: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RouteRequest {
    pub task: Task,
    pub execution: Execution,
    pub routes: BTreeMap<Band, Route>,
    pub preferences: Preferences,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RouteDecision {
    pub route: Route,
    pub band: Band,
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub switch: bool,
    #[serde(default)]
    pub reasons: Vec<String>,
    #[serde(default = "default_recheck")]
    pub recheck_after_turns: usize,
}

const fn default_recheck() -> usize {
    1
}

pub struct RouterClient {
    url: String,
    client: Client,
}

impl RouterClient {
    /// # Errors
    /// Returns an error if the HTTP client cannot be constructed.
    pub fn new(base_url: &str) -> Result<Self, String> {
        let client = Client::builder()
            .connect_timeout(Duration::from_millis(750))
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            url: format!("{}/v1/route", base_url.trim_end_matches('/')),
            client,
        })
    }

    /// # Errors
    /// Returns transport, status, and response-decoding errors.
    pub fn route(&self, request: &RouteRequest) -> Result<RouteDecision, String> {
        let response = self
            .client
            .post(&self.url)
            .json(request)
            .send()
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!("routing API returned {}", response.status()));
        }
        response.json().map_err(|e| e.to_string())
    }
}

#[must_use]
pub fn derive_indicators(prompt: &str) -> Indicators {
    let text = prompt.to_ascii_lowercase();
    let has = |words: &[&str]| words.iter().any(|word| text.contains(word));
    Indicators {
        security: has(&["security", "vulnerability", "auth", "credential", "secret"]),
        concurrency: has(&[
            "concurrency",
            "concurrent",
            "race condition",
            "deadlock",
            "thread",
        ]),
        migration: has(&["migration", "migrate", "schema change", "backfill"]),
        architecture: has(&["architecture", "architect", "system design", "redesign"]),
        refactor: has(&["refactor", "restructure", "rewrite"]),
        trivial: has(&["typo", "rename", "one line", "small change", "simple"]),
    }
}

#[must_use]
pub fn local_route(request: &RouteRequest) -> Option<RouteDecision> {
    if request.routes.is_empty() {
        return None;
    }
    let i = &request.task.signals.indicators;
    let mut band = if request.execution.consecutive_tool_failures >= 2
        || i.security
        || i.concurrency
        || i.migration
        || i.architecture
    {
        Band::Complex
    } else if i.refactor || request.task.signals.prompt_chars > 1200 {
        Band::Standard
    } else {
        Band::Simple
    };
    if let Some(min) = request.preferences.min_band {
        band = band.max(min);
    }
    if let Some(max) = request.preferences.max_band {
        band = band.min(max);
    }
    if let Some(previous) = request.execution.previous_band
        && previous != band
        && request.execution.consecutive_tool_failures < 2
        && request.execution.turns_on_route < request.preferences.minimum_turns_on_route
    {
        band = previous;
    }
    let route = request
        .routes
        .get(&band)
        .cloned()
        .or_else(|| {
            request
                .routes
                .range(band..)
                .next()
                .map(|(_, route)| route.clone())
        })
        .or_else(|| {
            request
                .routes
                .range(..band)
                .next_back()
                .map(|(_, route)| route.clone())
        })?;
    Some(RouteDecision {
        route,
        band,
        score: 0.0,
        confidence: 0.5,
        switch: request
            .execution
            .previous_band
            .is_some_and(|old| old != band),
        reasons: vec![
            if request.execution.consecutive_tool_failures >= 2 {
                "repeated tool failures during repair"
            } else {
                "local task classification"
            }
            .to_owned(),
        ],
        recheck_after_turns: 1,
    })
}

pub struct RoutedModel {
    registry: ProviderRegistry,
    client: RouterClient,
    config: RoutingConfig,
    task: Task,
    current: Option<RouteDecision>,
    turns_on_route: usize,
    switches: usize,
    api_fallback_notice: bool,
}

impl RoutedModel {
    /// Construct an automatic source, dropping routes without credentials.
    ///
    /// # Errors
    /// Returns an error when no configured route has a usable credential.
    pub fn new(
        config: RoutingConfig,
        prompt: &str,
        message_count: usize,
        plan: bool,
    ) -> Result<Self, String> {
        let available = ProviderKind::all()
            .into_iter()
            .filter(|kind| kind.has_credential())
            .map(ProviderKind::name)
            .collect::<Vec<_>>();
        let mut config = config;
        config
            .routes
            .retain(|_, route| available.contains(&route.provider.as_str()));
        if config.routes.is_empty() {
            return Err("no configured routing provider has a credential".to_owned());
        }
        let client = RouterClient::new(&config.api_url)?;
        let task = Task {
            signals: TaskSignals {
                prompt_chars: prompt.chars().count(),
                conversation_message_count: message_count,
                plan_mode: plan,
                indicators: derive_indicators(prompt),
                ..TaskSignals::default()
            },
            prompt: config.send_prompt.then(|| prompt.to_owned()),
        };
        Ok(Self {
            registry: ProviderRegistry::from_available(),
            client,
            config,
            task,
            current: None,
            turns_on_route: 0,
            switches: 0,
            api_fallback_notice: false,
        })
    }

    #[must_use]
    pub fn current(&self) -> Option<&RouteDecision> {
        self.current.as_ref()
    }
    #[must_use]
    pub const fn switches(&self) -> usize {
        self.switches
    }
    #[must_use]
    pub const fn used_local_fallback(&self) -> bool {
        self.api_fallback_notice
    }
}

impl ModelSource for RoutedModel {
    fn next(&mut self, state: &TurnState) -> Result<Selection<'_>, String> {
        let request = RouteRequest {
            task: self.task.clone(),
            execution: Execution {
                turn_index: state.turn_index,
                turns_remaining: state.turns_remaining,
                consecutive_tool_failures: state.consecutive_tool_failures,
                previous_band: self.current.as_ref().map(|decision| decision.band),
                turns_on_route: self.turns_on_route,
                switches_so_far: self.switches,
                ..Execution::default()
            },
            routes: self.config.routes.clone(),
            preferences: Preferences {
                min_band: self.config.min_band,
                max_band: self.config.max_band,
                minimum_turns_on_route: 2,
                max_switches: 5,
            },
        };
        let mut decision = if let Ok(decision) = self.client.route(&request) {
            decision
        } else {
            self.api_fallback_notice = true;
            let mut decision = local_route(&request).ok_or("local router found no usable route")?;
            decision
                .reasons
                .insert(0, "routing API unreachable — using local rules".to_owned());
            decision
        };
        let changed = self
            .current
            .as_ref()
            .is_some_and(|old| old.route != decision.route);
        let first = self.current.is_none();
        decision.switch = changed;
        if changed {
            self.switches += 1;
            self.turns_on_route = 0;
        } else {
            self.turns_on_route += 1;
        }
        self.current = Some(decision.clone());
        let provider = self
            .registry
            .get(&decision.route.provider)
            .ok_or_else(|| format!("provider {} is unavailable", decision.route.provider))?;
        Ok(Selection {
            provider,
            provider_name: provider.name(),
            model: &self.current.as_ref().expect("set above").route.model,
            decision: (first || changed).then_some(decision),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn derives_high_risk_signals() {
        let i = derive_indicators("Refactor authentication concurrency handling");
        assert!(i.refactor && i.security && i.concurrency);
    }
    #[test]
    fn failures_floor_local_route_at_complex() {
        let mut request = RouteRequest {
            task: Task::default(),
            execution: Execution::default(),
            routes: BTreeMap::new(),
            preferences: Preferences {
                minimum_turns_on_route: 2,
                max_switches: 5,
                ..Preferences::default()
            },
        };
        request.routes.insert(
            Band::Complex,
            Route {
                provider: "openai".into(),
                model: "strong".into(),
            },
        );
        request.execution.consecutive_tool_failures = 2;
        assert_eq!(local_route(&request).expect("route").band, Band::Complex);
    }
}
