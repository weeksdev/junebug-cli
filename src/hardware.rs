//! Best-effort local hardware profiling so Junebug can recommend a local
//! model size that will actually run, instead of the user guessing and
//! hitting OOM or swap thrash on Ollama/`local-openai`.
//!
//! Detection is deliberately approximate: it estimates a usable memory
//! budget from unified memory (Apple Silicon), discrete VRAM (`nvidia-smi`),
//! or total system RAM as a last resort, then maps that budget to a
//! parameter-count tier assuming Q4_K_M-class quantization (the Ollama
//! default), which GGUF sizes in the wild cluster around 0.6 GB per billion
//! parameters.

use std::process::Command;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Human-readable description of what was detected, e.g.
    /// "Apple M3 Max · 36 GB unified memory".
    pub summary: String,
    /// Estimated GB safely usable for a local model's weights plus
    /// context/runtime overhead, after leaving headroom for the OS and
    /// other running apps.
    pub budget_gb: f64,
    /// Whether a GPU (integrated or discrete) was actually detected, as
    /// opposed to falling back to a bare system-RAM guess.
    pub gpu_detected: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SizeTier {
    pub label: &'static str,
    /// Parameter count (billions) a model in this tier tops out around.
    pub max_params_b: f64,
    pub examples: &'static [&'static str],
}

const TIERS: &[SizeTier] = &[
    SizeTier {
        label: "~3B",
        max_params_b: 3.5,
        examples: &["qwen2.5:3b", "llama3.2:3b"],
    },
    SizeTier {
        label: "~8B",
        max_params_b: 9.0,
        examples: &["qwen3:8b", "llama3.1:8b"],
    },
    SizeTier {
        label: "~14B",
        max_params_b: 15.0,
        examples: &["qwen3:14b", "phi4:14b"],
    },
    SizeTier {
        label: "~32B",
        max_params_b: 33.0,
        examples: &["qwen3:32b", "qwq:32b"],
    },
    SizeTier {
        label: "~70B",
        max_params_b: 72.0,
        examples: &["llama3.3:70b", "qwen2.5:72b"],
    },
    SizeTier {
        label: "~235B (MoE)",
        max_params_b: 236.0,
        examples: &["qwen3:235b-a22b"],
    },
];

/// GB of Q4_K_M-quantized weights per billion parameters. Derived from
/// observed GGUF sizes (7B/8B/13B/70B models cluster around 0.58-0.6 GB/B).
const BYTES_PER_PARAM_GB: f64 = 0.6;
/// Flat allowance for the runtime, KV cache, and context at moderate
/// context lengths. Approximate by design — long-context sessions need more.
const RUNTIME_OVERHEAD_GB: f64 = 2.0;

/// Detect once per process; hardware doesn't change mid-session.
pub fn profile() -> &'static HardwareProfile {
    static PROFILE: OnceLock<HardwareProfile> = OnceLock::new();
    PROFILE.get_or_init(detect)
}

fn detect() -> HardwareProfile {
    if cfg!(target_os = "macos")
        && let Some(profile) = detect_macos()
    {
        return profile;
    }
    if let Some(profile) = detect_nvidia() {
        return profile;
    }
    detect_fallback_ram()
}

fn run_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_owned())
}

#[allow(clippy::cast_precision_loss)]
fn detect_macos() -> Option<HardwareProfile> {
    let memsize_bytes: u64 = run_stdout("sysctl", &["-n", "hw.memsize"])?.parse().ok()?;
    let brand = run_stdout("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default();
    // Real-world RAM sizes are far below 2^52 bytes, so f64's mantissa has
    // ample precision here.
    let total_gb = memsize_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    if brand.contains("Apple") {
        Some(HardwareProfile {
            summary: format!("{} · {total_gb:.0} GB unified memory", brand.trim()),
            // Unified memory is shared with the OS and every other running
            // app, so budget conservatively.
            budget_gb: total_gb * 0.65,
            gpu_detected: true,
        })
    } else {
        // Intel Mac: no unified-memory GPU story to lean on.
        Some(HardwareProfile {
            summary: format!("Intel Mac · {total_gb:.0} GB RAM (no discrete GPU detected)"),
            budget_gb: total_gb * 0.4,
            gpu_detected: false,
        })
    }
}

fn detect_nvidia() -> Option<HardwareProfile> {
    let raw = run_stdout(
        "nvidia-smi",
        &["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    )?;
    let total_mib: f64 = raw
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .sum();
    if total_mib <= 0.0 {
        return None;
    }
    let total_gb = total_mib / 1024.0;
    Some(HardwareProfile {
        summary: format!("NVIDIA GPU · {total_gb:.0} GB VRAM"),
        // Dedicated VRAM isn't shared with the OS, so less headroom is
        // needed than on unified memory.
        budget_gb: total_gb * 0.85,
        gpu_detected: true,
    })
}

fn detect_fallback_ram() -> HardwareProfile {
    let total_gb = linux_meminfo_gb().unwrap_or(8.0);
    HardwareProfile {
        summary: format!(
            "no dedicated GPU detected · {total_gb:.0} GB system RAM (CPU inference will be slow)"
        ),
        budget_gb: total_gb * 0.4,
        gpu_detected: false,
    }
}

fn linux_meminfo_gb() -> Option<f64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib: f64 = contents
        .lines()
        .find(|line| line.starts_with("MemTotal:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kib / 1024.0 / 1024.0)
}

/// Largest size tier expected to run comfortably within `profile`'s budget.
#[must_use]
pub fn recommend(profile: &HardwareProfile) -> &'static SizeTier {
    let max_params_b = (profile.budget_gb - RUNTIME_OVERHEAD_GB).max(0.0) / BYTES_PER_PARAM_GB;
    TIERS
        .iter()
        .rev()
        .find(|tier| tier.max_params_b <= max_params_b)
        .unwrap_or(&TIERS[0])
}

/// Parse a leading `<number>b` parameter count out of an Ollama-style model
/// tag (`qwen3:8b` -> `8.0`, `llama3.1:70b` -> `70.0`). Returns `None` for
/// MoE-per-expert tags like `mixtral:8x7b` where the trailing number
/// doesn't reflect resident memory, and for tags with no discernible size.
#[must_use]
pub fn parse_param_billions(model_name: &str) -> Option<f64> {
    let tag = model_name.rsplit(':').next().unwrap_or(model_name);
    for (index, char) in tag.char_indices() {
        if char != 'b' && char != 'B' {
            continue;
        }
        let head = &tag[..index];
        let digits_start = head
            .rfind(|candidate: char| !candidate.is_ascii_digit() && candidate != '.')
            .map_or(0, |position| position + 1);
        let number = &head[digits_start..];
        if number.is_empty() || head[..digits_start].ends_with('x') {
            continue;
        }
        if let Ok(value) = number.parse::<f64>() {
            return Some(value);
        }
    }
    None
}

/// Fit hint for a specific installed model against `profile`, or `None`
/// when it's a comfortable fit (or its size can't be determined), so the
/// picker only calls out models actually worth a second thought.
#[must_use]
pub fn fit_hint(profile: &HardwareProfile, model_name: &str) -> Option<&'static str> {
    let params_b = parse_param_billions(model_name)?;
    let needed_gb = params_b.mul_add(BYTES_PER_PARAM_GB, RUNTIME_OVERHEAD_GB);
    if needed_gb <= profile.budget_gb {
        None
    } else if needed_gb <= profile.budget_gb * 1.3 {
        Some("tight — may be slow or swap")
    } else {
        Some("likely too large for this machine")
    }
}

#[cfg(test)]
mod tests {
    use super::{HardwareProfile, TIERS, fit_hint, parse_param_billions, recommend};

    #[test]
    fn parses_simple_param_tags() {
        assert_eq!(parse_param_billions("qwen3:8b"), Some(8.0));
        assert_eq!(parse_param_billions("llama3.1:70b"), Some(70.0));
        assert_eq!(parse_param_billions("phi3:3.8b"), Some(3.8));
        assert_eq!(parse_param_billions("qwen3:235b-a22b"), Some(235.0));
    }

    #[test]
    fn skips_moe_expert_counts_and_unsized_tags() {
        assert_eq!(parse_param_billions("mixtral:8x7b"), None);
        assert_eq!(parse_param_billions("qwen3:latest"), None);
        assert_eq!(parse_param_billions("codellama"), None);
    }

    #[test]
    fn recommends_larger_tiers_for_bigger_budgets() {
        let small = HardwareProfile {
            summary: String::new(),
            budget_gb: 5.2,
            gpu_detected: true,
        };
        let large = HardwareProfile {
            summary: String::new(),
            budget_gb: 23.4,
            gpu_detected: true,
        };
        assert_eq!(recommend(&small).label, "~3B");
        assert_eq!(recommend(&large).label, "~32B");
    }

    #[test]
    fn recommend_never_panics_on_zero_budget() {
        let empty = HardwareProfile {
            summary: String::new(),
            budget_gb: 0.0,
            gpu_detected: false,
        };
        assert_eq!(recommend(&empty).label, TIERS[0].label);
    }

    #[test]
    fn flags_oversized_models_only() {
        let profile = HardwareProfile {
            summary: String::new(),
            budget_gb: 10.0,
            gpu_detected: true,
        };
        assert_eq!(fit_hint(&profile, "qwen3:8b"), None);
        assert_eq!(
            fit_hint(&profile, "qwen3:14b"),
            Some("tight — may be slow or swap")
        );
        assert_eq!(
            fit_hint(&profile, "llama3.3:70b"),
            Some("likely too large for this machine")
        );
        assert_eq!(fit_hint(&profile, "codellama"), None);
    }
}
