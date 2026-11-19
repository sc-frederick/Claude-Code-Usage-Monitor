//! Usage for accounts pooled by a CLIProxyAPI hub
//! (<https://github.com/router-for-me/CLIProxyAPI>).
//!
//! The hub holds each account's OAuth login. Its management API lists the
//! accounts (`GET auth-files`) and makes upstream requests with an account's
//! stored token (`POST api-call`, which replaces `$TOKEN$` in headers). The
//! monitor asks each provider's own usage endpoint that way, then parses the
//! body exactly as it would a direct response.
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::{build_agent, claude, codex, grok, windows_credentials, PollError};
use crate::accounts::CliProxySettings;
use crate::diagnose;
use crate::models::UsageData;
use crate::providers::ProviderId;

/// Sent with Claude requests: the usage endpoint rate-limits other clients.
const CLAUDE_USER_AGENT: &str = "claude-code/2.1.0";
/// Profiles of one hub poll concurrently; list its accounts once per cycle.
const ACCOUNT_LIST_TTL: Duration = Duration::from_secs(20);

/// One account the hub can use, from `GET auth-files`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HubAccount {
    /// Auth file name, stable across restarts. Profiles store this.
    pub auth: String,
    /// Opaque handle `api-call` uses to pick the account's token.
    pub auth_index: String,
    pub provider: Option<ProviderId>,
    /// Email or label for display.
    pub label: String,
    pub disabled: bool,
    /// ChatGPT workspace, needed by the Codex usage endpoint.
    pub chatgpt_account_id: Option<String>,
}

#[derive(Deserialize)]
struct AuthFilesResponse {
    #[serde(default)]
    files: Vec<AuthFile>,
}

#[derive(Deserialize)]
struct AuthFile {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    auth_index: Option<serde_json::Value>,
    #[serde(default)]
    provider: String,
    #[serde(default, rename = "type")]
    type_: String,
    #[serde(default)]
    email: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    disabled: bool,
    #[serde(default)]
    id_token: Option<IdTokenClaims>,
}

#[derive(Deserialize)]
struct IdTokenClaims {
    #[serde(default)]
    chatgpt_account_id: Option<String>,
}

#[derive(Deserialize)]
struct ApiCallResponse {
    status_code: u16,
    #[serde(default)]
    body: String,
}

fn hub_provider(name: &str) -> Option<ProviderId> {
    match name.to_ascii_lowercase().as_str() {
        "claude" => Some(ProviderId::Claude),
        "codex" => Some(ProviderId::Codex),
        "xai" | "grok" => Some(ProviderId::Grok),
        _ => None,
    }
}

fn accounts_from_response(response: AuthFilesResponse) -> Vec<HubAccount> {
    response
        .files
        .into_iter()
        .filter_map(|file| {
            let auth = if file.id.is_empty() {
                file.name
            } else {
                file.id
            };
            let auth_index = match file.auth_index? {
                serde_json::Value::String(value) => value,
                serde_json::Value::Number(value) => value.to_string(),
                _ => return None,
            };
            if auth.is_empty() || auth_index.is_empty() {
                return None;
            }
            let provider_name = if file.provider.is_empty() {
                file.type_
            } else {
                file.provider
            };
            let label = [file.email, file.label]
                .into_iter()
                .find(|value| !value.trim().is_empty())
                .unwrap_or_else(|| auth.clone());
            Some(HubAccount {
                provider: hub_provider(&provider_name),
                label,
                disabled: file.disabled,
                chatgpt_account_id: file
                    .id_token
                    .and_then(|claims| claims.chatgpt_account_id)
                    .map(|id| id.trim().to_string())
                    .filter(|id| !id.is_empty()),
                auth,
                auth_index,
            })
        })
        .collect()
}

/// The management key, read from where the settings say it lives.
fn management_key(settings: &CliProxySettings) -> Result<String, PollError> {
    let key = if !settings.key_credential.trim().is_empty() {
        windows_credentials::read_generic(settings.key_credential.trim())
    } else {
        let path = crate::accounts::expand_path(std::path::Path::new(settings.key_file.trim()));
        path.and_then(|path| std::fs::read_to_string(path).ok())
    };
    key.map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
        .ok_or_else(|| {
            diagnose::log("CLIProxyAPI management key not found");
            PollError::NoCredentials
        })
}

fn management_request<T: serde::de::DeserializeOwned>(
    settings: &CliProxySettings,
    path: &str,
    body: Option<serde_json::Value>,
) -> Result<T, PollError> {
    let base = settings.management_base().ok_or_else(|| {
        diagnose::log("CLIProxyAPI hub URL must start with https:// or http://");
        PollError::NoCredentials
    })?;
    let key = management_key(settings)?;
    let url = format!("{base}/{path}");
    let authorization = format!("Bearer {key}");
    let agent = build_agent()?;
    let result = match body {
        Some(body) => agent
            .post(&url)
            .header("Authorization", &authorization)
            .send_json(body),
        None => agent
            .get(&url)
            .header("Authorization", &authorization)
            .call(),
    };
    let mut response = match result.and_then(super::check_http_status) {
        Ok(response) => response,
        Err(ureq::Error::StatusCode(code)) => {
            diagnose::log(format!("CLIProxyAPI {path} returned status {code}"));
            return Err(PollError::HttpStatus(code));
        }
        Err(error) => {
            diagnose::log_error(&format!("CLIProxyAPI {path} request failed"), error);
            return Err(PollError::NetworkError);
        }
    };
    response.body_mut().read_json().map_err(|error| {
        diagnose::log_error(&format!("unexpected CLIProxyAPI {path} response"), error);
        PollError::UnexpectedResponse
    })
}

/// Every account in the hub, including disabled and unsupported ones.
pub fn list_accounts(settings: &CliProxySettings) -> Result<Vec<HubAccount>, PollError> {
    management_request::<AuthFilesResponse>(settings, "auth-files", None)
        .map(accounts_from_response)
}

/// Result of matching profiles against the hub's account list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HubSync {
    /// Enabled hub accounts of a provider that supports profiles.
    pub supported: usize,
    /// Profiles created for accounts no profile tracked yet.
    pub added: usize,
}

/// Add a monitored profile for every enabled, supported hub account that no
/// profile tracks. A profile the user disabled still counts as tracking.
pub fn add_missing_profiles(
    settings: &mut crate::accounts::AccountSettings,
    hub_accounts: &[HubAccount],
) -> HubSync {
    let mut sync = HubSync::default();
    for account in hub_accounts.iter().filter(|account| !account.disabled) {
        let Some(profiles) = account
            .provider
            .and_then(|provider| settings.get_mut(provider))
        else {
            continue;
        };
        sync.supported += 1;
        sync.added += profiles.add_cliproxy(&account.auth, &account.label) as usize;
    }
    sync
}

/// Profiles for hub accounts signed in since the last check, so a new hub
/// account is monitored without visiting Settings. Uses the poll cycle's
/// cached account list.
pub fn sync_profiles(
    settings: &mut crate::accounts::AccountSettings,
) -> Result<HubSync, PollError> {
    if !settings.cliproxy.is_configured() {
        return Ok(HubSync::default());
    }
    let hub_accounts = cached_accounts(&settings.cliproxy)?;
    Ok(add_missing_profiles(settings, &hub_accounts))
}

type AccountCache = Option<(CliProxySettings, Instant, Vec<HubAccount>)>;

fn cached_accounts(settings: &CliProxySettings) -> Result<Vec<HubAccount>, PollError> {
    static CACHE: Mutex<AccountCache> = Mutex::new(None);
    // Holding the lock while listing makes concurrent profiles share one call.
    let mut cache = CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((cached_settings, at, accounts)) = cache.as_ref() {
        if cached_settings == settings && at.elapsed() < ACCOUNT_LIST_TTL {
            return Ok(accounts.clone());
        }
    }
    let accounts = list_accounts(settings)?;
    *cache = Some((settings.clone(), Instant::now(), accounts.clone()));
    Ok(accounts)
}

struct UpstreamRequest {
    url: &'static str,
    headers: Vec<(&'static str, String)>,
}

fn upstream_request(provider: ProviderId, account: &HubAccount) -> Option<UpstreamRequest> {
    let bearer = ("Authorization", "Bearer $TOKEN$".to_string());
    match provider {
        ProviderId::Claude => Some(UpstreamRequest {
            url: claude::USAGE_URL,
            headers: vec![
                bearer,
                ("anthropic-beta", "oauth-2025-04-20".into()),
                ("User-Agent", CLAUDE_USER_AGENT.into()),
            ],
        }),
        ProviderId::Codex => {
            let mut headers = vec![bearer, ("User-Agent", "codex-cli".into())];
            if let Some(id) = &account.chatgpt_account_id {
                headers.push(("ChatGPT-Account-Id", id.clone()));
            }
            Some(UpstreamRequest {
                url: codex::CODEX_USAGE_URL,
                headers,
            })
        }
        ProviderId::Grok => Some(UpstreamRequest {
            url: grok::GROK_BILLING_URL,
            headers: vec![
                bearer,
                ("X-XAI-Token-Auth", grok::GROK_TOKEN_AUTH.into()),
                ("x-grok-client-mode", grok::GROK_CLIENT_MODE.into()),
                ("x-grok-client-version", grok::grok_client_version()),
            ],
        }),
        _ => None,
    }
}

/// Usage for the hub account whose auth file is `auth`.
pub(super) fn poll_account(
    settings: &CliProxySettings,
    provider: ProviderId,
    auth: &str,
) -> Result<UsageData, PollError> {
    let account = cached_accounts(settings)?
        .into_iter()
        .find(|account| account.auth == auth)
        .ok_or_else(|| {
            diagnose::log(format!("CLIProxyAPI has no account {auth}"));
            PollError::NoCredentials
        })?;
    if account.disabled {
        diagnose::log(format!("CLIProxyAPI account {auth} is disabled"));
        return Err(PollError::NoCredentials);
    }
    if account.provider != Some(provider) {
        diagnose::log(format!(
            "CLIProxyAPI account {auth} is not a {} account",
            provider.descriptor().display_name
        ));
        return Err(PollError::NoCredentials);
    }
    let request = upstream_request(provider, &account).ok_or(PollError::NoCredentials)?;
    let header: serde_json::Map<String, serde_json::Value> = request
        .headers
        .into_iter()
        .map(|(name, value)| (name.to_string(), serde_json::Value::String(value)))
        .collect();
    let response: ApiCallResponse = management_request(
        settings,
        "api-call",
        Some(serde_json::json!({
            "auth_index": account.auth_index,
            "method": "GET",
            "url": request.url,
            "header": header,
        })),
    )?;
    if !(200..300).contains(&response.status_code) {
        diagnose::log(format!(
            "{} usage through CLIProxyAPI returned status {}",
            provider.descriptor().display_name,
            response.status_code
        ));
        // The hub owns and refreshes this login, so an upstream auth failure
        // is not something signing in on this machine can fix. Keep polling.
        return Err(match response.status_code {
            401 | 403 => PollError::RequestFailed,
            code => PollError::HttpStatus(code),
        });
    }
    match provider {
        ProviderId::Claude => claude::usage_from_json(&response.body),
        ProviderId::Codex => codex::usage_from_json(
            &response.body,
            account.chatgpt_account_id.as_deref(),
            &credit_state_key(settings, auth),
        ),
        ProviderId::Grok => grok::usage_from_json(&response.body),
        _ => Err(PollError::NoCredentials),
    }
}

/// Codex credit baselines are stored per source. A hub account has no file,
/// so it is keyed by hub and auth file instead.
fn credit_state_key(settings: &CliProxySettings, auth: &str) -> PathBuf {
    PathBuf::from(format!(
        "cliproxy|{}|{auth}",
        settings.management_base().unwrap_or_default()
    ))
}

/// Changes whenever the hub, the key location, or the account changes, so a
/// paused failure resumes after the user fixes the settings.
pub(super) fn source_signature(settings: &CliProxySettings, auth: &str) -> String {
    crate::accounts::fingerprint(&format!(
        "cliproxy|{}|{}|{}|{auth}",
        settings.management_base().unwrap_or_default(),
        settings.key_credential.trim(),
        settings.key_file.trim(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_files_map_to_accounts() {
        let response: AuthFilesResponse = serde_json::from_str(
            r#"{"files":[
                {"id":"claude-a.json","auth_index":"abc","provider":"claude","type":"claude",
                 "email":"a@example.com","label":"a@example.com","disabled":false},
                {"id":"codex-b.json","auth_index":"def","provider":"codex",
                 "email":"","label":"Work","id_token":{"chatgpt_account_id":" acct-1 ","plan_type":"pro"}},
                {"id":"xai-c.json","auth_index":7,"type":"xai","disabled":true},
                {"id":"gemini-d.json","auth_index":"ghi","provider":"gemini-cli"},
                {"id":"no-index.json","provider":"claude"}
            ],"observed_at":"2026-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let accounts = accounts_from_response(response);
        assert_eq!(accounts.len(), 4);
        assert_eq!(accounts[0].provider, Some(ProviderId::Claude));
        assert_eq!(accounts[0].label, "a@example.com");
        assert_eq!(accounts[1].label, "Work");
        assert_eq!(accounts[1].chatgpt_account_id.as_deref(), Some("acct-1"));
        assert_eq!(accounts[2].provider, Some(ProviderId::Grok));
        assert_eq!(accounts[2].auth_index, "7");
        assert_eq!(accounts[2].label, "xai-c.json");
        assert!(accounts[2].disabled);
        assert_eq!(accounts[3].provider, None);
    }

    fn hub_account(auth: &str, provider: Option<ProviderId>, disabled: bool) -> HubAccount {
        HubAccount {
            auth: auth.into(),
            auth_index: auth.into(),
            provider,
            label: format!("{auth} label"),
            disabled,
            chatgpt_account_id: None,
        }
    }

    #[test]
    fn new_hub_accounts_get_profiles_once() {
        let mut settings = crate::accounts::AccountSettings::default();
        settings.normalize();
        let mut hub = vec![
            hub_account("claude-a.json", Some(ProviderId::Claude), false),
            hub_account("claude-b.json", Some(ProviderId::Claude), false),
            hub_account("codex-c.json", Some(ProviderId::Codex), false),
            hub_account("claude-off.json", Some(ProviderId::Claude), true),
            hub_account("gemini-d.json", None, false),
        ];
        let first = add_missing_profiles(&mut settings, &hub);
        assert_eq!(
            first,
            HubSync {
                supported: 3,
                added: 3
            }
        );

        // A user-disabled profile still tracks its account.
        settings.claude.profiles[1].enabled = false;
        hub.push(hub_account(
            "claude-e.json",
            Some(ProviderId::Claude),
            false,
        ));
        let second = add_missing_profiles(&mut settings, &hub);
        assert_eq!(
            second,
            HubSync {
                supported: 4,
                added: 1
            }
        );
        let claude: Vec<_> = settings
            .claude
            .profiles
            .iter()
            .filter(|profile| profile.uses_cliproxy())
            .map(|profile| (profile.cliproxy_auth.as_str(), profile.name.as_str()))
            .collect();
        assert_eq!(
            claude,
            [
                ("claude-a.json", "claude-a.json label"),
                ("claude-b.json", "claude-b.json label"),
                ("claude-e.json", "claude-e.json label"),
            ]
        );
        assert_eq!(add_missing_profiles(&mut settings, &hub).added, 0);
    }

    #[test]
    fn management_base_accepts_origin_or_full_path() {
        let mut settings = CliProxySettings {
            url: "https://hub.example:8318/".into(),
            ..Default::default()
        };
        assert_eq!(
            settings.management_base().as_deref(),
            Some("https://hub.example:8318/v0/management")
        );
        settings.url = "https://hub.example:8318/v8/management".into();
        assert_eq!(
            settings.management_base().as_deref(),
            Some("https://hub.example:8318/v0/management")
        );
        settings.url = "hub.example:8318".into();
        assert_eq!(settings.management_base(), None);
    }

    #[test]
    fn codex_requests_carry_the_workspace_and_a_token_placeholder() {
        let account = HubAccount {
            auth: "codex.json".into(),
            auth_index: "1".into(),
            provider: Some(ProviderId::Codex),
            label: "x".into(),
            disabled: false,
            chatgpt_account_id: Some("acct".into()),
        };
        let request = upstream_request(ProviderId::Codex, &account).unwrap();
        assert!(request
            .headers
            .contains(&("Authorization", "Bearer $TOKEN$".into())));
        assert!(request
            .headers
            .contains(&("ChatGPT-Account-Id", "acct".into())));
    }

    #[test]
    fn signature_changes_with_key_location() {
        let mut settings = CliProxySettings {
            url: "https://hub.example".into(),
            key_credential: "one".into(),
            ..Default::default()
        };
        let first = source_signature(&settings, "a.json");
        settings.key_credential = "two".into();
        assert_ne!(first, source_signature(&settings, "a.json"));
    }
}
