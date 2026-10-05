// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sandbox supervisor SPIFFE ID template.

use std::sync::LazyLock;

use regex::Regex;

/// Matches the per-sandbox ClusterSPIFFEID in the OpenShell Helm SPIRE overlay.
/// Keycloak matches the SVID subject exactly, so this must equal the template
/// SPIRE uses to issue supervisor SVIDs.
pub const DEFAULT_TEMPLATE: &str = "{trustDomain}/openshell/sandbox/{sandboxID}";

/// Only values present in both CreateSandbox and DeleteSandbox responses are
/// allowed, so every registered ID can also be deregistered.
const KNOWN_PLACEHOLDERS: &[&str] = &["{trustDomain}", "{sandboxID}"];

static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\{[^{}]*\}").unwrap());

#[derive(Debug, Clone)]
pub struct SpiffeIdTemplate {
    template: String,
}

impl SpiffeIdTemplate {
    /// Validates a template. An empty template selects [`DEFAULT_TEMPLATE`].
    pub fn new(template: &str) -> Result<Self, String> {
        let template = if template.is_empty() {
            DEFAULT_TEMPLATE
        } else {
            template
        };
        for placeholder in PLACEHOLDER.find_iter(template) {
            if !KNOWN_PLACEHOLDERS.contains(&placeholder.as_str()) {
                return Err(format!(
                    "SPIFFE ID template {template:?}: unknown placeholder {}",
                    placeholder.as_str()
                ));
            }
        }
        if !template.contains("{sandboxID}") {
            return Err(format!(
                "SPIFFE ID template {template:?} must contain {{sandboxID}}"
            ));
        }
        Ok(Self {
            template: template.to_string(),
        })
    }

    pub fn render(&self, trust_domain: &str, sandbox_id: &str) -> String {
        self.template
            .replace("{trustDomain}", trust_domain.trim_end_matches('/'))
            .replace("{sandboxID}", sandbox_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_template_matches_openshell_cluster_spiffe_id() {
        let template = SpiffeIdTemplate::new("").unwrap();
        assert_eq!(
            template.render("spiffe://openshell.local", "9c4daea2"),
            "spiffe://openshell.local/openshell/sandbox/9c4daea2"
        );
    }

    #[test]
    fn renders_custom_template_and_trims_trailing_slash() {
        let template = SpiffeIdTemplate::new("{trustDomain}/workloads/{sandboxID}").unwrap();
        assert_eq!(
            template.render("spiffe://td/", "id-1"),
            "spiffe://td/workloads/id-1"
        );
    }

    #[test]
    fn rejects_name_placeholder() {
        // DeleteSandbox only returns the sandbox ID, so a name-based ID could
        // be registered but never deregistered.
        let err = SpiffeIdTemplate::new("{trustDomain}/{name}/{sandboxID}").unwrap_err();
        assert!(err.contains("{name}"), "{err}");
    }

    #[test]
    fn rejects_template_without_sandbox_id() {
        let err = SpiffeIdTemplate::new("{trustDomain}/openshell/static").unwrap_err();
        assert!(err.contains("{sandboxID}"), "{err}");
    }

    #[test]
    fn rejects_unknown_placeholder() {
        // {namespace} is not known to the gateway API, unlike the Kubernetes controller.
        let err = SpiffeIdTemplate::new("{trustDomain}/{namespace}/{sandboxID}").unwrap_err();
        assert!(err.contains("{namespace}"), "{err}");
    }
}
